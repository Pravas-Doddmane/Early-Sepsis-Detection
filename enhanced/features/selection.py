"""
Phase 4: Hybrid Feature Selection (GPU-accelerated)
=====================================================
Reads : enhanced/data/processed/train_temporal.parquet
        enhanced/data/processed/temporal_feature_cols.json
Writes: enhanced/experiments/selected_features.json
        enhanced/experiments/feature_importance.csv
        enhanced/experiments/feature_selection_plot.png

Strategy (fit on TRAIN only -- no leakage):
  Step 1 -- Mutual Information (sklearn, CPU)
            Fast filter: rank all features by MI with SepsisLabel
            Keep top MI_TOPK (default 150) candidates

  Step 2 -- Boruta with XGBoost estimator (GPU-accelerated!)
            Uses XGBoost GPU histogram for fast tree building
            Runs n_iter=50 shadow-feature trials, p<0.01
            Outputs: confirmed, tentative, rejected sets

  Step 3 -- Union(Boruta_confirmed, top_MI_100) = final_features
            Saves JSON + CSV + feature importance plot

GPU note:
  XGBoost inside Boruta uses tree_method='hist' + device='cuda'
  This gives ~5-10x speedup vs CPU for the RF iterations.
  LightGBM/CatBoost training (Phase 5) also uses GPU.

Important: This phase samples the training data for speed.
  Full 1.09M row training set is too large for Boruta (50 RF iterations).
  We use a stratified sample of SAMPLE_ROWS rows (default 100k).
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils import resample

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent.parent
PROCESSED    = ROOT / "enhanced" / "data" / "processed"
EXPERIMENTS  = ROOT / "enhanced" / "experiments"
EXPERIMENTS.mkdir(parents=True, exist_ok=True)

TEMPORAL_JSON = PROCESSED / "temporal_feature_cols.json"

# ─── Config ───────────────────────────────────────────────────────────────────
SAMPLE_ROWS  = 150_000   # stratified sample for Boruta (memory/speed)
MI_TOPK      = 150       # top-k features from MI step
BORUTA_ITERS = 50        # Boruta shadow iterations
BORUTA_ALPHA = 0.01      # Boruta p-value threshold
RANDOM_STATE = 42
USE_GPU      = True      # RTX 4060 -- set False if CUDA not available


# ─── GPU check ────────────────────────────────────────────────────────────────
def check_gpu():
    try:
        import xgboost as xgb
        # Quick smoke test with GPU
        dtrain = xgb.DMatrix(np.random.rand(100, 10), label=np.random.randint(0, 2, 100))
        params = {"tree_method": "hist", "device": "cuda", "verbosity": 0}
        xgb.train(params, dtrain, num_boost_round=2)
        print("  GPU (CUDA) available -- Boruta will use XGBoost on GPU")
        return True
    except Exception as e:
        print(f"  GPU not available ({e}) -- falling back to CPU")
        return False


# ─── Step 1: Mutual Information ────────────────────────────────────────────────
def run_mutual_info(X: pd.DataFrame, y: pd.Series, feature_cols: list) -> pd.Series:
    print("\n[Step 1] Mutual Information ranking...")
    t0 = time.time()
    # Fill NaN with 0 for MI (MI requires no NaN)
    X_mi = X[feature_cols].fillna(0).values
    mi_scores = mutual_info_classif(X_mi, y.values, random_state=RANDOM_STATE, n_jobs=-1)
    mi_series = pd.Series(mi_scores, index=feature_cols).sort_values(ascending=False)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  Top 10 by MI: {mi_series.head(10).index.tolist()}")
    return mi_series


# ─── Step 2: Boruta with XGBoost (GPU) ────────────────────────────────────────
def run_boruta(X: pd.DataFrame, y: pd.Series, feature_cols: list, use_gpu: bool) -> dict:
    print(f"\n[Step 2] Boruta feature selection (n_iter={BORUTA_ITERS})...")
    try:
        from boruta import BorutaPy
    except ImportError:
        print("  boruta not installed -- skipping. Run: pip install boruta")
        return {"confirmed": [], "tentative": feature_cols[:50], "rejected": []}

    import xgboost as xgb
    from sklearn.pipeline import Pipeline

    # XGBoost estimator configured for GPU
    if use_gpu:
        estimator = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            tree_method="hist",
            device="cuda",
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=RANDOM_STATE,
            n_jobs=1,   # XGBoost GPU handles parallelism internally
            verbosity=0,
        )
    else:
        estimator = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            tree_method="hist",
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )

    boruta_selector = BorutaPy(
        estimator=estimator,
        n_estimators="auto",
        verbose=1,
        alpha=BORUTA_ALPHA,
        max_iter=BORUTA_ITERS,
        random_state=RANDOM_STATE,
    )

    X_b = X[feature_cols].fillna(0).values.astype(np.float32)
    y_b = y.values.astype(int)

    t0 = time.time()
    boruta_selector.fit(X_b, y_b)
    elapsed = time.time() - t0

    feature_arr = np.array(feature_cols)
    confirmed  = feature_arr[boruta_selector.support_].tolist()
    tentative  = feature_arr[boruta_selector.support_weak_].tolist()
    rejected   = feature_arr[~boruta_selector.support_ & ~boruta_selector.support_weak_].tolist()

    print(f"  Boruta done in {elapsed:.1f}s")
    print(f"  Confirmed: {len(confirmed)} | Tentative: {len(tentative)} | Rejected: {len(rejected)}")
    print(f"  Top confirmed: {confirmed[:10]}")

    # Feature ranking from Boruta
    ranking = pd.Series(boruta_selector.ranking_, index=feature_cols)
    return {
        "confirmed" : confirmed,
        "tentative" : tentative,
        "rejected"  : rejected,
        "ranking"   : ranking,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("Phase 4: Hybrid Feature Selection")
    print("=" * 65)

    # Load feature list
    with open(TEMPORAL_JSON) as f:
        info = json.load(f)

    all_features = info["all_feature_columns"]
    target       = info["target"]          # SepsisLabel
    print(f"\nTotal features available: {len(all_features)}")

    # Load train temporal data
    print("\nLoading train_temporal.parquet (this may take ~30s)...")
    t0 = time.time()
    train = pd.read_parquet(PROCESSED / "train_temporal.parquet")
    print(f"  Loaded: {train.shape[0]:,} rows x {train.shape[1]} cols in {time.time()-t0:.1f}s")

    # Verify all features present
    available = [f for f in all_features if f in train.columns]
    missing_cols = [f for f in all_features if f not in train.columns]
    if missing_cols:
        print(f"  Warning: {len(missing_cols)} features missing from parquet: {missing_cols[:5]}")
    print(f"  Using {len(available)} features")

    y_full = train[target].astype(int)
    X_full = train[available]

    # Class imbalance info
    n_pos = y_full.sum()
    n_neg = len(y_full) - n_pos
    pos_rate = n_pos / len(y_full) * 100
    print(f"\n  Class balance: {n_pos:,} sepsis ({pos_rate:.1f}%), {n_neg:,} non-sepsis")

    # ── Stratified sample for Boruta ──────────────────────────────────────────
    print(f"\nStratified sampling {SAMPLE_ROWS:,} rows for Boruta...")
    pos_idx = y_full[y_full == 1].index
    neg_idx = y_full[y_full == 0].index

    # Preserve class ratio in sample
    n_pos_sample = min(len(pos_idx), int(SAMPLE_ROWS * pos_rate / 100))
    n_neg_sample = SAMPLE_ROWS - n_pos_sample

    sampled_pos = resample(pos_idx, n_samples=n_pos_sample, random_state=RANDOM_STATE, replace=False)
    sampled_neg = resample(neg_idx, n_samples=n_neg_sample, random_state=RANDOM_STATE, replace=False)
    sample_idx  = np.concatenate([sampled_pos, sampled_neg])
    np.random.shuffle(sample_idx)

    X_sample = X_full.loc[sample_idx]
    y_sample = y_full.loc[sample_idx]
    print(f"  Sample: {len(y_sample):,} rows | {y_sample.sum():,} sepsis ({y_sample.mean()*100:.1f}%)")

    # ── Check GPU ─────────────────────────────────────────────────────────────
    use_gpu = USE_GPU and check_gpu()

    # ── Step 1: Mutual Information (full train, fast) ─────────────────────────
    # MI runs on sample too for speed (1M rows can take 10+ minutes)
    mi_scores = run_mutual_info(X_sample, y_sample, available)
    top_mi_features = mi_scores.head(MI_TOPK).index.tolist()
    print(f"\n  MI top-{MI_TOPK} candidates selected")

    # ── Step 2: Boruta on MI top-k candidates (GPU) ───────────────────────────
    # Run Boruta only on top MI features (reduces feature space for speed)
    print(f"\n  Running Boruta on {len(top_mi_features)} MI-filtered features...")
    boruta_result = run_boruta(X_sample, y_sample, top_mi_features, use_gpu)

    confirmed  = boruta_result["confirmed"]
    tentative  = boruta_result["tentative"]

    # ── Step 3: Union of Boruta confirmed + top MI-100 ────────────────────────
    top_mi_100 = mi_scores.head(100).index.tolist()
    final_features = list(dict.fromkeys(confirmed + tentative + top_mi_100))
    # dict.fromkeys preserves order and deduplicates
    print(f"\n[Step 3] Final feature set (union):")
    print(f"  Boruta confirmed : {len(confirmed)}")
    print(f"  Boruta tentative : {len(tentative)}")
    print(f"  MI top-100       : {len(top_mi_100)}")
    print(f"  Union (final)    : {len(final_features)}")

    # ── Build importance CSV ──────────────────────────────────────────────────
    importance_df = pd.DataFrame({
        "feature"       : mi_scores.index,
        "mi_score"      : mi_scores.values,
        "mi_rank"       : range(1, len(mi_scores) + 1),
    })

    if "ranking" in boruta_result:
        boruta_rank = boruta_result["ranking"].reindex(importance_df["feature"]).values
        importance_df["boruta_rank"]      = boruta_rank
        importance_df["boruta_confirmed"] = importance_df["feature"].isin(confirmed)
        importance_df["boruta_tentative"] = importance_df["feature"].isin(tentative)
    else:
        importance_df["boruta_rank"]      = np.nan
        importance_df["boruta_confirmed"] = False
        importance_df["boruta_tentative"] = False

    importance_df["selected"] = importance_df["feature"].isin(final_features)
    importance_df = importance_df.sort_values("mi_rank")

    csv_out = EXPERIMENTS / "feature_importance.csv"
    importance_df.to_csv(csv_out, index=False)
    print(f"\n  Saved: {csv_out.name}")

    # ── Save selected features JSON ───────────────────────────────────────────
    selected_info = {
        "final_features"         : final_features,
        "n_final"                : len(final_features),
        "boruta_confirmed"       : confirmed,
        "boruta_tentative"       : tentative,
        "boruta_rejected"        : boruta_result.get("rejected", []),
        "top_mi_100"             : top_mi_100,
        "mi_top_k"               : MI_TOPK,
        "boruta_iterations"      : BORUTA_ITERS,
        "boruta_alpha"           : BORUTA_ALPHA,
        "sample_rows"            : SAMPLE_ROWS,
        "gpu_used"               : use_gpu,
        "target"                 : target,
    }

    json_out = EXPERIMENTS / "selected_features.json"
    with open(json_out, "w") as f:
        json.dump(selected_info, f, indent=2)
    print(f"  Saved: {json_out.name}")

    # ── Feature importance plot ───────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        top50 = importance_df.head(50)
        colors = []
        for _, row in top50.iterrows():
            if row["boruta_confirmed"]:
                colors.append("#2ecc71")   # green = confirmed
            elif row["boruta_tentative"]:
                colors.append("#f39c12")   # orange = tentative
            elif row["selected"]:
                colors.append("#3498db")   # blue = MI only
            else:
                colors.append("#bdc3c7")   # grey = not selected

        fig, ax = plt.subplots(figsize=(12, 16))
        bars = ax.barh(range(len(top50)), top50["mi_score"].values, color=colors[::-1])
        ax.set_yticks(range(len(top50)))
        ax.set_yticklabels(top50["feature"].values[::-1], fontsize=9)
        ax.set_xlabel("Mutual Information Score")
        ax.set_title(f"Top-50 Features by MI Score\n(Green=Boruta Confirmed, Orange=Tentative, Blue=MI-only)")
        ax.invert_yaxis()

        # Legend
        from matplotlib.patches import Patch
        legend = [
            Patch(color="#2ecc71", label="Boruta Confirmed"),
            Patch(color="#f39c12", label="Boruta Tentative"),
            Patch(color="#3498db", label="MI top-100 only"),
        ]
        ax.legend(handles=legend, loc="lower right")

        plt.tight_layout()
        plot_out = EXPERIMENTS / "feature_selection_plot.png"
        plt.savefig(plot_out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {plot_out.name}")
    except Exception as e:
        print(f"  Plot skipped: {e}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"Phase 4 COMPLETE")
    print(f"  Input features : {len(available)}")
    print(f"  Final selected : {len(final_features)}")
    print(f"  Reduction      : {100*(1 - len(final_features)/len(available)):.1f}%")
    print(f"  Output JSON    : enhanced/experiments/selected_features.json")
    print(f"  Output CSV     : enhanced/experiments/feature_importance.csv")
    print(f"\nNext: Phase 5 -- train 4 base models (with GPU)")
    print(f"  python enhanced/models/train_xgb.py")
    print(f"  python enhanced/models/train_lgbm.py")
    print(f"  python enhanced/models/train_catboost.py")
    print(f"  python enhanced/models/train_rf.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
