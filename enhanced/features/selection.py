#!/usr/bin/env python
"""
Phase 4: Hybrid Feature Selection
Pipeline:
  All Features (309)
      │
      ├─ Step 1: Mutual Information (sklearn, CPU, fast) → top 150
      │
      ├─ Step 2: Boruta with XGBoost GPU (RTX 4060)
      │          n_iter=50, p<0.01 — runs on 150k stratified sample
      │
      └─ Union → final_features (~80-100)
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

INPUT_DIR = Path("enhanced/data/processed")
OUTPUT_DIR = Path("enhanced/experiments")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = 'SepsisLabel'
ID_VAR = 'patient_id'
TIME_VAR = 'ICULOS'

EXCLUDE_COLS = {TARGET, ID_VAR, TIME_VAR}


def load_temporal_data():
    """Load temporal features."""
    print("Loading temporal features...")
    train = pd.read_parquet(INPUT_DIR / "train_temporal.parquet")
    val = pd.read_parquet(INPUT_DIR / "val_temporal.parquet")
    test = pd.read_parquet(INPUT_DIR / "test_temporal.parquet")

    # Feature columns
    feature_cols = [c for c in train.columns if c not in EXCLUDE_COLS]
    print(f"Total features: {len(feature_cols)}")
    print(f"Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")

    return train, val, test, feature_cols


def step1_mutual_info(train, feature_cols, top_k=150):
    """Step 1: Mutual Information filter (fast, CPU)."""
    print(f"\n[Step 1] Mutual Information → top {top_k} features...")

    X = train[feature_cols].fillna(0)
    y = train[TARGET]

    # Sample for speed (stratified)
    sample_size = min(200000, len(X))
    if len(X) > sample_size:
        _, X_sample, _, y_sample = train_test_split(
            X, y, train_size=sample_size, stratify=y, random_state=42
        )
    else:
        X_sample, y_sample = X, y

    print(f"  Computing MI on {len(X_sample):,} samples...")
    mi_scores = mutual_info_classif(X_sample, y_sample, random_state=42, n_neighbors=3)

    mi_df = pd.DataFrame({'feature': feature_cols, 'mi_score': mi_scores})
    mi_df = mi_df.sort_values('mi_score', ascending=False)

    selected_mi = mi_df.head(top_k)['feature'].tolist()
    print(f"  Selected top {len(selected_mi)} by MI")

    # Save MI scores
    mi_df.to_csv(OUTPUT_DIR / "mi_scores.csv", index=False)

    return selected_mi, mi_df


def step2_boruta(train, candidate_features, n_iter=50, sample_size=150000):
    """Step 2: Boruta with XGBoost GPU on stratified sample."""
    print(f"\n[Step 2] Boruta (XGBoost GPU, n_iter={n_iter})...")

    X = train[candidate_features].fillna(0)
    y = train[TARGET]

    # Stratified sample
    if len(X) > sample_size:
        _, X_sample, _, y_sample = train_test_split(
            X, y, train_size=sample_size, stratify=y, random_state=42
        )
    else:
        X_sample, y_sample = X, y

    print(f"  Running Boruta on {len(X_sample):,} samples × {len(candidate_features)} features...")

    # XGBoost GPU estimator for Boruta
    xgb_clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',
        device='cuda',
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )

    # Simple Boruta implementation
    # Create shadow features (shuffled)
    X_shadow = X_sample.copy()
    for col in X_shadow.columns:
        X_shadow[col] = np.random.permutation(X_shadow[col].values)
    X_shadow.columns = [f'shadow_{c}' for c in X_shadow.columns]

    # Combine real + shadow
    X_combined = pd.concat([X_sample, X_shadow], axis=1)

    # Run multiple iterations
    hit_counts = pd.Series(0, index=candidate_features)

    for iteration in range(n_iter):
        if iteration % 10 == 0:
            print(f"  Iteration {iteration+1}/{n_iter}")

        # Train on combined
        xgb_clf.fit(X_combined, y_sample)

        # Get feature importances
        importances = pd.Series(xgb_clf.feature_importances_, index=X_combined.columns)

        # Max shadow importance
        max_shadow = importances.filter(like='shadow_').max()

        # Real features beating max shadow
        real_imp = importances[candidate_features]
        hits = (real_imp > max_shadow).astype(int)
        hit_counts += hits

        # Re-shuffle shadow features
        for col in X_shadow.columns:
            orig_col = col.replace('shadow_', '')
            X_shadow[col] = np.random.permutation(X_sample[orig_col].values)

    # Selection: features with hits significantly > random
    # Threshold: binomial test p < 0.01 (or heuristic: hits > n_iter * 0.5)
    # Use Bonferroni-corrected binomial threshold
    from scipy import stats
    p_threshold = 0.01 / len(candidate_features)
    selected_boruta = []

    for feat in candidate_features:
        hits = hit_counts[feat]
        p_val = 1 - stats.binom.cdf(hits - 1, n_iter, 0.5)
        if p_val < p_threshold:
            selected_boruta.append(feat)

    print(f"  Boruta selected: {len(selected_boruta)} features (p<{p_threshold:.2e})")

    # Save hit counts
    hit_df = pd.DataFrame({'feature': hit_counts.index, 'hits': hit_counts.values})
    hit_df.to_csv(OUTPUT_DIR / "boruta_hits.csv", index=False)

    return selected_boruta, hit_counts


def final_union(selected_mi, selected_boruta):
    """Union of MI + Boruta."""
    final = list(set(selected_mi) | set(selected_boruta))
    print(f"\n[Final] Union: {len(selected_mi)} MI + {len(selected_boruta)} Boruta = {len(final)} features")

    # Save final list
    with open(OUTPUT_DIR / "selected_features.json", 'w') as f:
        json.dump(final, f, indent=2)

    # Save comparison
    comparison = pd.DataFrame({
        'feature': final,
        'in_mi': [f in selected_mi for f in final],
        'in_boruta': [f in selected_boruta for f in final]
    })
    comparison.to_csv(OUTPUT_DIR / "feature_selection_comparison.csv", index=False)

    return final


def main():
    print("=" * 60)
    print("Phase 4: Hybrid Feature Selection")
    print("=" * 60)

    # Load data
    train, val, test, feature_cols = load_temporal_data()

    # Step 1: Mutual Information
    selected_mi, mi_df = step1_mutual_info(train, feature_cols, top_k=150)

    # Step 2: Boruta on MI candidates
    selected_boruta, hit_counts = step2_boruta(train, selected_mi, n_iter=50, sample_size=150000)

    # Final union
    final_features = final_union(selected_mi, selected_boruta)

    print("\n" + "=" * 60)
    print("Phase 4 Complete!")
    print("=" * 60)
    print(f"Final features: {len(final_features)}")
    print(f"Saved: {OUTPUT_DIR}/selected_features.json")
    print(f"Saved: {OUTPUT_DIR}/mi_scores.csv")
    print(f"Saved: {OUTPUT_DIR}/boruta_hits.csv")
    print(f"Saved: {OUTPUT_DIR}/feature_selection_comparison.csv")
    print("\nNext: Retrain Phase 5 models on selected features")
    print("  python enhanced/models/train_rf.py")
    print("  python enhanced/models/train_xgb.py")
    print("  python enhanced/models/train_lgbm.py")
    print("  python enhanced/models/train_catboost.py")


if __name__ == "__main__":
    main()