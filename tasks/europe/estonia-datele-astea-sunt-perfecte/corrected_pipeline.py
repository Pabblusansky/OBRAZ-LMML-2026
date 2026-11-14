"""
e-Residency application fraud screen -- audit and corrected evaluation pipeline.

Demonstrates:
1. Original pipeline reproduction (93.42%).
2. Isolation of false leads (red herrings) that leave accuracy ~93-96%:
   - False Lead A: StandardScaler fitted on the full dataset before train_test_split.
   - False Lead B: Outliers / physically impossible feature values (41 rows).
   - False Lead C: Unstratified / single random_state=1137 split vs cross-validation.
3. True bug diagnosis and fix:
   - Entity/applicant leakage across train and test: 1971 rows come from only
     900 unique (date_of_birth, citizenship) applicants with near-identical
     submissions and identical labels.
   - Group-aware evaluation with make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1)):
     * LeaveOneGroupOut() on all 1971 rows grouped by (date_of_birth, citizenship)
     * LeaveOneOut() on the 900 deduplicated unique applicants
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    LeaveOneGroupOut,
    LeaveOneOut,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "form_fill_seconds",
    "doc_scan_quality",
    "photo_match_score",
    "email_domain_age_days",
    "ip_geo_distance_km",
    "address_history_years",
]


def load_dataset() -> pd.DataFrame:
    """Load applications.csv from cwd or from the task's files/ directory."""
    candidates = [
        Path("applications.csv"),
        Path(__file__).resolve().parent / "files" / "applications.csv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError("Could not locate applications.csv")


def make_model_pipeline():
    """Return the prescribed StandardScaler + 1-NN pipeline."""
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1))


def find_impossible_mask(df: pd.DataFrame) -> pd.Series:
    """Identify rows containing physically impossible numeric values."""
    return (
        (df["form_fill_seconds"] < 0)
        | (df["doc_scan_quality"] < 0)
        | (df["doc_scan_quality"] > 1)
        | (df["photo_match_score"] < 0)
        | (df["photo_match_score"] > 1)
        | (df["email_domain_age_days"] < 0)
        | (df["ip_geo_distance_km"] < 0)
        | (df["address_history_years"] < 0)
    )


def clip_impossible_values(X: pd.DataFrame) -> pd.DataFrame:
    """Clip numeric features to their physically valid domains."""
    X_clip = X.copy()
    for col in [
        "form_fill_seconds",
        "email_domain_age_days",
        "ip_geo_distance_km",
        "address_history_years",
    ]:
        X_clip[col] = X_clip[col].clip(lower=0.0)
    for col in ["doc_scan_quality", "photo_match_score"]:
        X_clip[col] = X_clip[col].clip(0.0, 1.0)
    return X_clip


def run_original_pipeline(X: pd.DataFrame, y: pd.Series) -> tuple[float, int, int]:
    """Reproduce project_pipeline.py exactly."""
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=1137
    )
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    correct = int((preds == y_test).sum())
    return accuracy_score(y_test, preds), correct, len(y_test)


def evaluate_false_lead_a(X: pd.DataFrame, y: pd.Series) -> None:
    """False Lead A: StandardScaler fitted before train_test_split."""
    print("\n--- False Lead A: Preprocessing Leakage (StandardScaler before split) ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1137
    )
    pipe = make_model_pipeline()
    pipe.fit(X_train, y_train)
    acc_1137 = accuracy_score(y_test, pipe.predict(X_test))

    leaked_accs, proper_accs = [], []
    X_prescaled = StandardScaler().fit_transform(X)
    for seed in range(100):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
        X_tr_l, X_te_l, _, _ = train_test_split(
            X_prescaled, y, test_size=0.2, random_state=seed
        )
        leaked_accs.append(
            accuracy_score(
                y_te, KNeighborsClassifier(n_neighbors=1).fit(X_tr_l, y_tr).predict(X_te_l)
            )
        )
        proper_accs.append(
            accuracy_score(y_te, make_model_pipeline().fit(X_tr, y_tr).predict(X_te))
        )

    loo_acc = accuracy_score(
        y, cross_val_predict(make_model_pipeline(), X, y, cv=LeaveOneOut())
    )
    print(f"Fix A (scaler inside train, rs=1137): {acc_1137:.4f} ({acc_1137:.2%})")
    print(
        f"100-seed mean±std: leaked={np.mean(leaked_accs):.4f}±{np.std(leaked_accs):.4f} "
        f"vs proper={np.mean(proper_accs):.4f}±{np.std(proper_accs):.4f}"
    )
    print(f"LeaveOneOut on all rows (scaler inside fold): {loo_acc:.4f} ({loo_acc:.2%})")


def evaluate_false_lead_b(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> None:
    """False Lead B: Impossible / uncleaned feature values (41 rows)."""
    print("\n--- False Lead B: Impossible Feature Values (negatives / >1 scores) ---")
    bad_mask = find_impossible_mask(df)
    print(f"Rows with out-of-range values: {int(bad_mask.sum())} / {len(df)}")

    df_drop = df[~bad_mask].reset_index(drop=True)
    X_drop, y_drop = df_drop[FEATURE_COLS], df_drop["label"]
    X_drop_s = StandardScaler().fit_transform(X_drop)
    X_tr_d, X_te_d, y_tr_d, y_te_d = train_test_split(
        X_drop_s, y_drop, test_size=0.2, random_state=1137
    )
    acc_drop = accuracy_score(
        y_te_d, KNeighborsClassifier(n_neighbors=1).fit(X_tr_d, y_tr_d).predict(X_te_d)
    )
    loo_drop = accuracy_score(
        y_drop,
        cross_val_predict(make_model_pipeline(), X_drop, y_drop, cv=LeaveOneOut()),
    )

    X_clip = clip_impossible_values(X)
    X_clip_s = StandardScaler().fit_transform(X_clip)
    X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
        X_clip_s, y, test_size=0.2, random_state=1137
    )
    acc_clip = accuracy_score(
        y_te_c, KNeighborsClassifier(n_neighbors=1).fit(X_tr_c, y_tr_c).predict(X_te_c)
    )
    loo_clip = accuracy_score(
        y, cross_val_predict(make_model_pipeline(), X_clip, y, cv=LeaveOneOut())
    )

    print(f"Fix B1 (drop {int(bad_mask.sum())} rows, rs=1137): {acc_drop:.4f} ({acc_drop:.2%}) | LOO={loo_drop:.4f}")
    print(f"Fix B2 (clip to valid bounds, rs=1137): {acc_clip:.4f} ({acc_clip:.2%}) | LOO={loo_clip:.4f}")


def evaluate_false_lead_c(X: pd.DataFrame, y: pd.Series) -> None:
    """False Lead C: Unstratified single split / random_state=1137."""
    print("\n--- False Lead C: Split Configuration (stratify / seeds / naive LOO) ---")
    X_scaled = StandardScaler().fit_transform(X)
    X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(
        X_scaled, y, test_size=0.2, random_state=1137, stratify=y
    )
    acc_strat = accuracy_score(
        y_te_s, KNeighborsClassifier(n_neighbors=1).fit(X_tr_s, y_tr_s).predict(X_te_s)
    )
    skf5_acc = accuracy_score(
        y,
        cross_val_predict(
            make_model_pipeline(),
            X,
            y,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=1137),
        ),
    )
    loo_naive = accuracy_score(
        y, cross_val_predict(make_model_pipeline(), X, y, cv=LeaveOneOut())
    )
    print(f"Fix C1 (stratify=y, rs=1137): {acc_strat:.4f} ({acc_strat:.2%})")
    print(f"Fix C2 (5-fold StratifiedKFold): {skf5_acc:.4f} ({skf5_acc:.2%})")
    print(f"Fix C3 (naive LeaveOneOut on all 1971 rows): {loo_naive:.4f} ({loo_naive:.2%})")


def evaluate_true_bug_and_fix(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> None:
    """Diagnose applicant leakage and compute true performance."""
    print("\n--- True Bug: Applicant/Entity Leakage Across Train and Test ---")
    groups = df.groupby(["date_of_birth", "citizenship"]).ngroup()
    n_groups = groups.nunique()
    label_nunique = df.groupby(["date_of_birth", "citizenship"])["label"].nunique()

    X_scaled = StandardScaler().fit_transform(X)
    group_arr = groups.to_numpy()
    dists = cdist(X_scaled, X_scaled)
    same_group = group_arr[:, None] == group_arr[None, :]
    max_within = dists[same_group].max()
    min_between = dists[~same_group].min()

    print(f"Total rows: {len(df)} | Unique (date_of_birth, citizenship) applicants: {n_groups}")
    print(f"Applicants with conflicting labels: {int((label_nunique > 1).sum())}")
    print(
        f"Scaled Euclidean distance: max within-applicant={max_within:.4f} "
        f"< min between-applicant={min_between:.4f}"
    )

    # Breakdown on the original random_state=1137 test set
    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
        X_scaled, y, groups, test_size=0.2, random_state=1137
    )
    preds = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train).predict(X_test)
    seen = g_test.isin(set(g_train))
    print(
        f"Original test split (395 rows): {int(seen.sum())} ({seen.mean():.2%}) "
        f"have same applicant in train -> acc={accuracy_score(y_test[seen], preds[seen]):.4f}; "
        f"{int((~seen).sum())} ({(~seen).mean():.2%}) unseen -> acc={accuracy_score(y_test[~seen], preds[~seen]):.4f}"
    )

    # Corrected evaluation 1: LeaveOneGroupOut over all 1971 rows
    pipe = make_model_pipeline()
    preds_logo = cross_val_predict(
        pipe, X, y, cv=LeaveOneGroupOut(), groups=groups
    )
    acc_logo = accuracy_score(y, preds_logo)
    correct_logo = int((preds_logo == y).sum())

    # Corrected evaluation 2: Deduplicate to 1 submission per applicant + LeaveOneOut
    df_dedup = df.drop_duplicates(subset=["date_of_birth", "citizenship"], keep="first")
    X_dedup, y_dedup = df_dedup[FEATURE_COLS], df_dedup["label"]
    preds_dedup = cross_val_predict(pipe, X_dedup, y_dedup, cv=LeaveOneOut())
    acc_dedup = accuracy_score(y_dedup, preds_dedup)
    correct_dedup = int((preds_dedup == y_dedup).sum())

    print("\n--- Corrected Pipeline Results ---")
    print(
        f"LeaveOneGroupOut (all {len(df)} rows, grouped by applicant): "
        f"{acc_logo:.4f} ({acc_logo:.2%}, {correct_logo}/{len(df)})"
    )
    print(
        f"Deduplicated LeaveOneOut ({len(df_dedup)} unique applicants, keep='first'): "
        f"{acc_dedup:.4f} ({acc_dedup:.2%}, {correct_dedup}/{len(df_dedup)})"
    )
    print(
        f"Flag (truncation int(acc*100), matching 0.8649 -> SIGMOID_86): "
        f"SIGMOID_{int(acc_logo * 100)}"
    )
    print(
        f"Flag (nearest integer rounding): SIGMOID_{round(acc_logo * 100)}"
    )


def main() -> None:
    df = load_dataset()
    X = df[FEATURE_COLS]
    y = df["label"]

    orig_acc, orig_ok, orig_total = run_original_pipeline(X, y)
    print(f"Original pipeline test accuracy: {orig_acc:.4f} ({orig_ok}/{orig_total})")

    evaluate_false_lead_a(X, y)
    evaluate_false_lead_b(df, X, y)
    evaluate_false_lead_c(X, y)
    evaluate_true_bug_and_fix(df, X, y)


if __name__ == "__main__":
    main()
