# diagnose_exit_meta.py
import json
import numpy as np
import pandas as pd
import joblib
import warnings

# Optional: plotting (comment out if running headless)
import matplotlib.pyplot as plt

def load_meta_results(path="meta_results.json") -> pd.DataFrame:
    with open(path, "r") as f:
        meta_results = json.load(f)

    rows = []
    for trade in meta_results:
        for cand in trade.get("meta_candidates", []) or []:
            feat = (cand.get("features", {}) or {}).copy()

            row = {
                **feat,
                # labeling from candidate (with defaults)
                "group_id": cand.get("group_id") or f"{trade.get('entry_date')}_{trade.get('ticker')}",
                "target":   int(cand.get("target", 0)) if cand.get("target") is not None else 0,
                "rel":      int(cand.get("rel", 0))    if cand.get("rel")    is not None else 0,
                "utility":  cand.get("utility"),

                # utility components (new names)
                "gain_lock_ratio": cand.get("gain_lock_ratio"),
                "future_drawdown": cand.get("future_drawdown"),
                "future_regret":   cand.get("future_regret"),
                "momentum_signal": cand.get("momentum_signal"),
                "momentum_penalty": cand.get("momentum_penalty"),
                "pct_from_peak":   cand.get("pct_from_peak"),
                "peak_proximity":  cand.get("peak_proximity"),
                "time_penalty":    cand.get("time_penalty"),
                "regret_penalty":  cand.get("regret_penalty"),
                "is_best_candidate": int(cand.get("is_best_candidate", 0) or 0),

                # candidate + trade context
                "exit_date":  cand.get("exit_date"),
                "exit_price": cand.get("exit_price"),
                "ticker":     trade.get("ticker"),
                "entry_date": trade.get("entry_date"),
                "entry_price": trade.get("entry_price"),
                "final_return": trade.get("final_return", 0.0),
                "regime":       trade.get("entry_regime"),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    # Keep only rows with core fields
    df = df.dropna(subset=["group_id", "entry_price", "exit_price"]).copy()

    # Cast types
    for col in ["target", "rel", "is_best_candidate"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0).astype(int)

    # Dates & days held
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"]  = pd.to_datetime(df["exit_date"])
    df["days_held"]  = (df["exit_date"] - df["entry_date"]).dt.days

    # Return delta (how much leaving on table vs final_return you logged)
    df["return_delta"] = ((df["exit_price"] - df["entry_price"]) / df["entry_price"]) - df["final_return"]

    # Backward compatibility for old plots: fabricate momentum_decay from new fields if needed
    if "momentum_decay" not in df.columns:
        # negative signal (decay) -> positive penalty; just provide a signed proxy for diagnostics
        df["momentum_decay"] = -df.get("momentum_signal", pd.Series(0.0, index=df.index)).fillna(0.0)

    return df

def ensure_rel_from_utility(df: pd.DataFrame,
                            near_best_1=0.010, near_best_2=0.020, near_best_3=0.035):
    """If rel is missing or all zeros, reconstruct graded relevance from utility proximity per group."""
    if "utility" not in df.columns or df["utility"].isna().all():
        print("[warn] Cannot reconstruct rel: 'utility' missing or all NaN.")
        return df

    need_reconstruct = ("rel" not in df.columns) or (df["rel"].sum() == 0)
    if not need_reconstruct:
        return df

    def _assign_rel(g):
        best = g["utility"].max()
        d = best - g["utility"]
        rel = np.select(
            [d <= near_best_1, d <= near_best_2, d <= near_best_3],
            [3, 2, 1],
            default=0
        )
        return pd.Series(rel, index=g.index)

    df["rel"] = df.groupby("group_id", group_keys=False).apply(_assign_rel).astype(int)
    df["target"] = (df["rel"] > 0).astype(int)
    print("[info] Reconstructed 'rel' from utility proximity bands.")
    return df

def load_model_and_predict(df: pd.DataFrame, model_path="exit_meta_model.pkl"):
    model = joblib.load(model_path)

    # Pull exact feature names from the trained model
    try:
        model_features = list(model.booster_.feature_name())
        gains = model.booster_.feature_importance(importance_type="gain")
    except AttributeError:
        model_features = getattr(model, "feature_name_", None)
        gains = getattr(model, "feature_importances_", None)
    if not model_features:
        raise RuntimeError("Model has no feature names. Retrain passing DataFrames, not .values.")

    # Add any missing features as zeros; order columns exactly
    missing = [c for c in model_features if c not in df.columns]
    if missing:
        print(f"[warn] Creating {len(missing)} missing model features as 0.0: {missing[:10]}{'...' if len(missing)>10 else ''}")
        for c in missing:
            df[c] = 0.0

    X = df[model_features].astype(float)
    na_mask = X.isna().any(axis=1)
    if na_mask.any():
        print(f"[warn] Dropping {na_mask.sum()} rows with NaNs in model features.")
        X = X.loc[~na_mask]
        df = df.loc[~na_mask]

    df["score"] = model.predict(X)

    return df, model, model_features, gains

def compute_topk_overlap(df: pd.DataFrame, truth_col="rel", ks=(1,3,5)):
    # drop singleton groups
    gsz = df.groupby("group_id").size()
    keep = gsz[gsz >= 2].index
    df = df[df["group_id"].isin(keep)].copy()

    # ranks
    df["pred_rank"] = df.groupby("group_id")["score"].rank("first", ascending=False)
    df["true_rank"] = df.groupby("group_id")[truth_col].rank("first", ascending=False)

    def topk(g, k):
        pred_top = g.nlargest(k, "score").index
        true_top = g.nlargest(k, truth_col).index
        return len(set(pred_top) & set(true_top)) / max(1, len(set(true_top)))

    out = {}
    for k in ks:
        out[k] = df.groupby("group_id").apply(topk, k=k).mean()
    return out, df

def compute_ndcg(df: pd.DataFrame, truth_col="rel", ks=(3,5)):
    try:
        from sklearn.metrics import ndcg_score
    except Exception:
        print("[info] sklearn not available; skipping NDCG.")
        return {}

    scores = {}
    for k in ks:
        vals = []
        for _, g in df.groupby("group_id"):
            y_true = g[[truth_col]].to_numpy().T
            y_score = g[["score"]].to_numpy().T
            vals.append(ndcg_score(y_true, y_score, k=k))
        scores[f"ndcg@{k}"] = float(np.mean(vals)) if vals else np.nan
    return scores

def summarize(df: pd.DataFrame):
    print("\n=== Label coverage ===")
    print("groups:", df["group_id"].nunique(), "| rows:", len(df))
    print("rel unique:", sorted(df["rel"].unique().tolist()), "| rel>0 rows:", int((df["rel"]>0).sum()))
    print("target unique:", sorted(df["target"].unique().tolist()), "| target=1 rows:", int((df["target"]==1).sum()))

    print("\n=== Positives per group (target) ===")
    print(df.groupby("group_id")["target"].sum().describe())

    print("\n=== Score dispersion per group ===")
    spread = df.groupby("group_id")["score"].std()
    print(spread.describe())
    flat_groups = (spread.fillna(0) < 1e-6).mean()
    print(f"Flat-score groups (std < 1e-6): {flat_groups:.2%}")

    # Correlation of score vs utility (sanity)
    if "utility" in df.columns and not df["utility"].isna().all():
        valid = df[["score", "utility"]].dropna()
        if len(valid) > 0:
            corr = valid["score"].corr(valid["utility"])
            print(f"\nCorr(score, utility): {corr:.3f}")
        else:
            print("\nCorr(score, utility): n/a (no valid rows)")
    else:
        print("\nNo utility column to correlate with score.")

    # Sample where model top pick has rel=0 (bad cases)
    print("\n=== Sample groups where model's top has rel=0 ===")
    bad = []
    for gid, g in df.groupby("group_id"):
        top_idx = g["score"].idxmax()
        if g.loc[top_idx, "rel"] == 0:
            bad.append((gid, top_idx))
        if len(bad) >= 5:
            break
    if bad:
        cols = ["group_id", "exit_date", "exit_price", "score", "rel", "utility"]
        print(df.loc[[idx for _, idx in bad], cols])
    else:
        print("(none in first 5 hits)")

def main():
    warnings.filterwarnings("ignore")

    # 1) Load & clean
    df = load_meta_results("meta_results.json")

    # 2) Rebuild rel from utility if needed
    df = ensure_rel_from_utility(df)

    # 3) Predict using trained model
    df, model, model_features, gains = load_model_and_predict(df, "exit_meta_model.pkl")

    # 4) Metrics on graded labels (preferred)
    truth_col = "rel" if "rel" in df.columns else "target"
    topk, df_ranked = compute_topk_overlap(df.copy(), truth_col=truth_col, ks=(1,3,5))
    ndcg = compute_ndcg(df_ranked, truth_col=truth_col, ks=(3,5))

    print("\n=== Accuracy (graded truth) ===")
    for k, v in topk.items():
        print(f"Top-{k} overlap: {v:.2%}")
    for k, v in ndcg.items():
        print(f"{k}: {v:.3f}")

    # 5) Summary & sanity
    summarize(df_ranked)

    # 6) Optional plots (comment out if running headless)
    try:
        # Per-group score std histogram
        spread = df_ranked.groupby("group_id")["score"].std()
        spread.hist(bins=50, edgecolor="black")
        plt.title("Score Std Dev per Group")
        plt.xlabel("Std( score | group )")
        plt.ylabel("#Groups")
        plt.tight_layout()
        plt.show()

        # Score vs utility
        if "utility" in df_ranked.columns:
            plt.figure()
            plt.scatter(df_ranked["utility"], df_ranked["score"], s=6, alpha=0.3)
            plt.title("Score vs Utility")
            plt.xlabel("Utility")
            plt.ylabel("Predicted Score")
            plt.tight_layout()
            plt.show()

    except Exception as e:
        print(f"[plot warn] {e}")

    # 7) Feature importances (names from model)
    try:
        if gains is not None and model_features is not None:
            if len(gains) == len(model_features):
                s = pd.Series(gains, index=model_features).sort_values()
                print("\n=== Top 15 features by gain ===")
                print(s.tail(15))
                try:
                    s.plot.barh(figsize=(8, 6), title="EXIT Meta Ranker Feature Importances (gain)")
                    plt.tight_layout(); plt.show()
                except Exception:
                    pass
            else:
                print(f"[warn] feature/importances length mismatch: {len(model_features)} vs {len(gains)}")
    except Exception as e:
        print(f"[feat warn] {e}")


    

# ==== ADD BELOW YOUR EXISTING CODE (or replace main with this) ====
import argparse

def check_missingness(df: pd.DataFrame):
    print("\n=== Missingness (key fields) ===")
    keys = ["utility","gain_lock_ratio","future_drawdown","future_regret",
            "momentum_signal","momentum_penalty","pct_from_peak","time_penalty"]
    miss = df[keys].isna().mean().sort_values(ascending=False)
    print((miss*100).round(2).to_string())
    if "utility" in df.columns:
        nunique_gid = df[df["utility"].isna()]["group_id"].nunique()
        print(f"Groups with ANY NaN utility rows: {nunique_gid}")

def label_balance_vs_spread(df_ranked: pd.DataFrame):
    print("\n=== Label density vs score spread ===")
    g = df_ranked.groupby("group_id")
    pos = g["rel"].apply(lambda s: (s>0).sum())
    std = g["score"].std()
    out = pd.DataFrame({"positives": pos, "score_std": std}).fillna(0)
    print(out.describe())
    corr = out.corr()
    print("\nCorr matrix (positives vs score_std):")
    print(corr)

    try:
        plt.figure(figsize=(6,5))
        plt.scatter(out["positives"], out["score_std"], s=10, alpha=0.4)
        plt.xlabel("#positives in group (rel>0)")
        plt.ylabel("std(score | group)")
        plt.title("Score dispersion vs label density")
        plt.tight_layout(); plt.show()
    except Exception:
        pass

def score_distributions(df: pd.DataFrame):
    print("\n=== Score distributions by label ===")
    pos = df[df["rel"]>0]["score"]
    neg = df[df["rel"]==0]["score"]
    print("pos count:", len(pos), "| neg count:", len(neg))
    print("pos score describe:\n", pos.describe())
    print("neg score describe:\n", neg.describe())
    try:
        plt.figure(figsize=(7,4))
        pos.hist(bins=60, alpha=0.6, label="rel>0")
        neg.hist(bins=60, alpha=0.6, label="rel=0")
        plt.legend(); plt.title("Score histogram by label")
        plt.tight_layout(); plt.show()
    except Exception:
        pass

def precision_at_k(df: pd.DataFrame, truth_col="rel", ks=(1,3,5)):
    prec = {}
    for k in ks:
        hits = 0; total=0
        for _, g in df.groupby("group_id"):
            g = g.sort_values("score", ascending=False)
            if len(g)==0: 
                continue
            topk = g.head(k)
            hits += (topk[truth_col] > 0).sum()
            total += len(topk)
        prec[k] = hits / max(1,total)
    return prec

def feature_redundancy(df: pd.DataFrame, feature_cols):
    exists = [c for c in feature_cols if c in df.columns]
    if len(exists) < 2:
        print("\n[feat] Not enough features present to compute redundancy.")
        return
    corr = df[exists].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = (upper.stack()
                   .reset_index()
                   .rename(columns={"level_0":"f1","level_1":"f2",0:"corr"})
                   .sort_values("corr", ascending=False))
    print("\n=== Most correlated feature pairs (|r|>=0.85) ===")
    print(pairs[pairs["corr"]>=0.85].head(20).to_string(index=False))

def quick_ablation(df: pd.DataFrame, drop_feats, truth_col="rel", headless=False):
    """
    Fast ablation: retrain a small LGBMRanker on current df dropping `drop_feats`.
    """
    import lightgbm as lgb
    # minimal safety: need at least 2 per group and truth column
    keep_gid = df.groupby("group_id").size()
    df2 = df[df["group_id"].isin(keep_gid[keep_gid>=2].index)].copy()
    y = df2[truth_col].astype(int)
    # features = model features if present, else all numeric except meta columns
    meta_cols = {"group_id","entry_date","exit_date","ticker",truth_col}
    cand_feats = [c for c in df2.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(df2[c])]
    cand_feats = [c for c in cand_feats if c not in drop_feats]
    if len(cand_feats)==0:
        print("[ablate] No features left after drop.")
        return None

    # split groups (80/20 by group blocks)
    gids = df2["group_id"].unique()
    cut = int(0.8 * len(gids))
    trn_gid = set(gids[:cut]); val_gid = set(gids[cut:])
    trn = df2[df2["group_id"].isin(trn_gid)]
    val = df2[df2["group_id"].isin(val_gid)]

    Xtr, ytr = trn[cand_feats].astype(float).values, trn[truth_col].values
    Xva, yva = val[cand_feats].astype(float).values, val[truth_col].values
    gtr = trn.groupby("group_id").size().to_list()
    gva = val.groupby("group_id").size().to_list()

    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=1200,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_data_in_leaf=50,
        random_state=42,
    )
    model.fit(
        Xtr, ytr,
        group=gtr,
        eval_set=[(Xva, yva)],
        eval_group=[gva],
        eval_at=[1,3,5],
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )

    # scores & ndcg on val
    val = val.copy()
    val["score_ablate"] = model.predict(val[cand_feats].astype(float))
    topk, ranked = compute_topk_overlap(val, truth_col=truth_col, ks=(1,3,5))
    ndcg = compute_ndcg(ranked, truth_col=truth_col, ks=(3,5))
    print(f"\n[ablation] Dropped {drop_feats}")
    for k,v in topk.items(): print(f"  overlap@{k}: {v:.3%}")
    for k,v in ndcg.items(): print(f"  {k}: {v:.3f}")
    return {"topk":topk, "ndcg":ndcg}

def forward_outcome_sanity(df: pd.DataFrame):
    """Optional: check forward return/drawdown if available."""
    has_ret = "future_return_5" in df.columns
    has_dd  = "future_drawdown" in df.columns
    if not (has_ret or has_dd):
        print("\n[forward] Skipping; no 'future_return_5' or 'future_drawdown'.")
        return
    print("\n=== Forward outcome sanity ===")
    if has_ret:
        print("future_return_5 by rel:\n", df.groupby("rel")["future_return_5"].describe())
    if has_dd:
        print("\nfuture_drawdown by rel:\n", df.groupby("rel")["future_drawdown"].describe())

def _pick_truth_col_for_best(df: pd.DataFrame) -> str:
    # Prefer utility (continuous), else rel (graded), else target
    if "utility" in df.columns and not df["utility"].isna().all():
        return "utility"
    if "rel" in df.columns:
        return "rel"
    return "target"

def _best_idx_by_peak(g: pd.DataFrame):
    # "Closest to peak" = minimize pct_from_peak (needs that column)
    if "pct_from_peak" in g.columns and not g["pct_from_peak"].isna().all():
        return g["pct_from_peak"].idxmin()
    # Fallback: minimal entry_to_peak_drawdown if available
    if "entry_to_peak_drawdown" in g.columns and not g["entry_to_peak_drawdown"].isna().all():
        return g["entry_to_peak_drawdown"].idxmin()
    return None

def evaluate_peak_capture(df_ranked: pd.DataFrame, truth_col: str = None, near_peak_band=0.03):
    """
    For each trade (group_id), compare the model's top-ranked exit with:
      (a) the best-by-utility candidate, and
      (b) the closest-to-peak candidate (min pct_from_peak).
    Returns a per-group DataFrame of gaps + a summary dict.
    """
    if truth_col is None:
        truth_col = _pick_truth_col_for_best(df_ranked)

    rows = []
    for gid, g in df_ranked.groupby("group_id"):
        if len(g) < 2:
            continue
        # Model pick
        model_idx = g["score"].idxmax()
        # Best by truth (utility or rel)
        best_truth_idx = g[truth_col].idxmax()
        # Best by peak proximity
        best_peak_idx = _best_idx_by_peak(g)

        model = g.loc[model_idx]
        best_truth = g.loc[best_truth_idx]

        # Defaults to NaN-safe pulls
        def get(grow, col, default=np.nan):
            return grow[col] if col in grow and pd.notna(grow[col]) else default

        model_pct  = get(model, "pct_from_peak")
        truth_pct  = get(best_truth, "pct_from_peak")
        peak_pct   = get(g.loc[best_peak_idx], "pct_from_peak") if best_peak_idx is not None else np.nan

        model_util = get(model, "utility")
        truth_util = get(best_truth, "utility")

        # Gaps (lower is better)
        gap_to_truth_util = (truth_util - model_util) if pd.notna(truth_util) and pd.notna(model_util) else np.nan
        gap_to_peak_pct   = (model_pct - peak_pct) if pd.notna(model_pct) and pd.notna(peak_pct) else np.nan
        gap_to_truth_pct  = (model_pct - truth_pct) if pd.notna(model_pct) and pd.notna(truth_pct) else np.nan

        # Optional forward-risk readouts if present
        model_dd   = get(model, "future_drawdown")
        model_reg  = get(model, "future_regret")

        rows.append({
            "group_id": gid,
            "ticker": g["ticker"].iloc[0] if "ticker" in g.columns else None,
            "model_idx": model_idx,
            "best_truth_idx": best_truth_idx,
            "best_peak_idx": best_peak_idx,
            "model_pct_from_peak": model_pct,
            "best_peak_pct_from_peak": peak_pct,
            "best_truth_pct_from_peak": truth_pct,
            "gap_to_peak_pct": gap_to_peak_pct,
            "gap_to_truth_pct": gap_to_truth_pct,
            "model_utility": model_util,
            "best_truth_utility": truth_util,
            "gap_to_truth_utility": gap_to_truth_util,
            "model_future_drawdown": model_dd,
            "model_future_regret": model_reg,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        print("[peaks] No eligible groups for peak capture evaluation.")
        return out, {}

    # Summaries
    def q(x, p): 
        return float(np.nanquantile(x, p)) if x.notna().any() else np.nan
    within_1 = float((out["gap_to_peak_pct"] <= 0.01).mean())
    within_2 = float((out["gap_to_peak_pct"] <= 0.02).mean())
    within_3 = float((out["gap_to_peak_pct"] <= 0.03).mean())
    exact_peak_pick = float((out["model_idx"] == out["best_peak_idx"]).mean())
    exact_truth_pick = float((out["model_idx"] == out["best_truth_idx"]).mean())

    summary = {
        "median_gap_to_peak_pct": q(out["gap_to_peak_pct"], 0.5),
        "p75_gap_to_peak_pct": q(out["gap_to_peak_pct"], 0.75),
        "mean_gap_to_peak_pct": float(np.nanmean(out["gap_to_peak_pct"])),
        "pct_within_1pct_of_peak": within_1,
        "pct_within_2pct_of_peak": within_2,
        "pct_within_3pct_of_peak": within_3,
        "exact_best_peak_pick_rate": exact_peak_pick,
        "exact_best_truth_pick_rate": exact_truth_pick,
        "median_gap_to_truth_util": q(out["gap_to_truth_utility"], 0.5),
        "mean_gap_to_truth_util": float(np.nanmean(out["gap_to_truth_utility"])),
    }

    return out, summary

def plot_peak_capture(out_df: pd.DataFrame, headless=False):
    if headless or out_df.empty:
        return
    try:
        # Histogram of model distance from peak (lower is better)
        plt.figure(figsize=(6,4))
        out_df["model_pct_from_peak"].dropna().hist(bins=60, edgecolor="black")
        plt.title("Model exit: % from trailing peak")
        plt.xlabel("pct_from_peak (lower = closer to peak)")
        plt.tight_layout(); plt.show()

        # Histogram of gap to the best-possible peak
        plt.figure(figsize=(6,4))
        out_df["gap_to_peak_pct"].dropna().hist(bins=60, edgecolor="black")
        plt.title("Gap to closest possible peak in group")
        plt.xlabel("model_pct - best_peak_pct")
        plt.tight_layout(); plt.show()

        # CDF (gap to peak)
        x = out_df["gap_to_peak_pct"].dropna().sort_values()
        y = np.linspace(0, 1, len(x), endpoint=True)
        plt.figure(figsize=(6,4))
        plt.plot(x, y)
        plt.title("CDF: gap to peak (fraction ≤ x)")
        plt.xlabel("gap_to_peak_pct")
        plt.tight_layout(); plt.show()
    except Exception as e:
        print(f"[plot peaks warn] {e}")
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deep", action="store_true", help="run deep diagnostics")
    parser.add_argument("--ablate", nargs="*", default=[], help="feature names to drop & quick-retrain")
    parser.add_argument("--headless", action="store_true", help="skip plots")
    parser.add_argument("--meta", default="meta_results.json", help="path to meta_results.json")
    parser.add_argument("--model", default="exit_meta_model.pkl", help="path to trained model")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    # 1) Load & clean
    df = load_meta_results(args.meta)

    # 2) Rebuild rel from utility if needed
    df = ensure_rel_from_utility(df)

    # 3) Predict using trained model
    df, model, model_features, gains = load_model_and_predict(df, args.model)

    # 4) Metrics on graded labels (preferred)
    truth_col = "rel" if "rel" in df.columns else "target"
    topk, df_ranked = compute_topk_overlap(df.copy(), truth_col=truth_col, ks=(1,3,5))
    ndcg = compute_ndcg(df_ranked, truth_col=truth_col, ks=(3,5))

    print("\n=== Accuracy (graded truth) ===")
    for k, v in topk.items():
        print(f"Top-{k} overlap: {v:.2%}")
    for k, v in ndcg.items():
        print(f"{k}: {v:.3f}")

    # 5) Summary & sanity
    summarize(df_ranked)

    if args.deep:
        check_missingness(df_ranked)
        label_balance_vs_spread(df_ranked)
        score_distributions(df_ranked)
        pr = precision_at_k(df_ranked, truth_col=truth_col, ks=(1,3,5))
        print("\n=== Precision@k (fraction of rel>0 among top-k picks) ===")
        for k,v in pr.items(): print(f"precision@{k}: {v:.2%}")

        feature_redundancy(df_ranked, model_features)
        forward_outcome_sanity(df_ranked)

        if args.ablate:
            _ = quick_ablation(df_ranked, drop_feats=args.ablate, truth_col=truth_col, headless=args.headless)

    # 6) Optional plots
    if not args.headless:
        try:
            spread = df_ranked.groupby("group_id")["score"].std()
            spread.hist(bins=60, edgecolor="black")
            plt.title("Score Std Dev per Group")
            plt.xlabel("Std( score | group )")
            plt.ylabel("#Groups")
            plt.tight_layout(); plt.show()

            if "utility" in df_ranked.columns:
                plt.figure()
                plt.scatter(df_ranked["utility"], df_ranked["score"], s=6, alpha=0.3)
                plt.title("Score vs Utility")
                plt.xlabel("Utility")
                plt.ylabel("Predicted Score")
                plt.tight_layout(); plt.show()
        except Exception as e:
            print(f"[plot warn] {e}")

    # 7) Feature importances
    try:
        if gains is not None and model_features is not None and len(gains)==len(model_features):
            s = pd.Series(gains, index=model_features).sort_values()
            print("\n=== Top 15 features by gain ===")
            print(s.tail(15))
            if not args.headless:
                try:
                    s.plot.barh(figsize=(8,6), title="EXIT Meta Ranker Feature Importances (gain)")
                    plt.tight_layout(); plt.show()
                except Exception:
                    pass
    except Exception as e:
        print(f"[feat warn] {e}")

        # --- Peak capture evaluation ---
    peak_df, peak_summary = evaluate_peak_capture(df_ranked, truth_col=truth_col)
    print("\n=== Peak capture summary ===")
    for k, v in peak_summary.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    plot_peak_capture(peak_df, headless=args.headless)


if __name__ == "__main__":
    main()
