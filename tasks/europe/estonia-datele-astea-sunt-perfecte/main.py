"""Audit complet si pipeline corectat pentru detectarea fraudelor e-Residency.

Ruleaza verificarea celor trei piste din pipeline-ul original (proiectul
Estonia - "Datele astea sunt perfecte!"), evalueaza pipeline-ul corectat cu
validare incrucisata grupata pe aplicant si genereaza automat `solution.txt`
langa acest script.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, pairwise_distances
from sklearn.model_selection import (
    LeaveOneGroupOut,
    LeaveOneOut,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_CANDIDATES = (
    Path("applications.csv"),
    BASE_DIR / "files" / "applications.csv",
)
SOLUTION_PATH = BASE_DIR / "solution.txt"

FEATURE_COLS = [
    "form_fill_seconds",
    "doc_scan_quality",
    "photo_match_score",
    "email_domain_age_days",
    "ip_geo_distance_km",
    "address_history_years",
]
RANDOM_STATE = 1137
TEST_SIZE = 0.2


def resolve_dataset_path() -> Path:
    """Returneaza calea catre applications.csv."""
    for candidate in DATA_CANDIDATES:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(p) for p in DATA_CANDIDATES)
    raise FileNotFoundError(f"Nu am gasit applications.csv in: {searched}")


def load_dataset() -> pd.DataFrame:
    """Incarca setul de date applications.csv."""
    return pd.read_csv(resolve_dataset_path())


def make_model_pipeline() -> Pipeline:
    """Construieste modelul impus: StandardScaler + 1-NN."""
    return make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=1),
    )


def evaluate_cv(
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    groups: pd.Series | None = None,
) -> tuple[float, int, int]:
    """Evalueaza pipeline-ul cu cross_val_predict si returneaza (acc, corecte, total)."""
    preds = cross_val_predict(
        make_model_pipeline(),
        X,
        y,
        cv=cv,
        groups=groups,
    )
    correct = int((preds == y.to_numpy()).sum())
    total = len(y)
    return float(accuracy_score(y, preds)), correct, total


def evaluate_original_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = RANDOM_STATE,
) -> tuple[float, int, int]:
    """Reproduce exact pipeline-ul initial cu scurgere de scalare si split ne-grupat."""
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=TEST_SIZE,
        random_state=random_state,
    )
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    correct = int((preds == y_test.to_numpy()).sum())
    return float(accuracy_score(y_test, preds)), correct, len(y_test)


def evaluate_proper_split(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = RANDOM_STATE,
    stratify: bool = False,
) -> tuple[float, int, int]:
    """Split train/test cu StandardScaler aplicat strict in interiorul train-ului."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=random_state,
        stratify=y if stratify else None,
    )
    model = make_model_pipeline()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    correct = int((preds == y_test.to_numpy()).sum())
    return float(accuracy_score(y_test, preds)), correct, len(y_test)


def audit_false_lead_preprocessing(X: pd.DataFrame, y: pd.Series) -> dict[str, float | int]:
    """Testeaza pista falsa A: StandardScaler fit-uit inainte de train_test_split."""
    fix_acc, fix_correct, fix_total = evaluate_proper_split(X, y)

    leaked_scores = []
    proper_scores = []
    for seed in range(100):
        acc_leaked, _, _ = evaluate_original_pipeline(X, y, random_state=seed)
        acc_proper, _, _ = evaluate_proper_split(X, y, random_state=seed)
        leaked_scores.append(acc_leaked)
        proper_scores.append(acc_proper)

    loo_acc, loo_correct, loo_total = evaluate_cv(X, y, LeaveOneOut())
    return {
        "fix_acc": fix_acc,
        "fix_correct": fix_correct,
        "fix_total": fix_total,
        "leaked_mean": float(np.mean(leaked_scores)),
        "leaked_std": float(np.std(leaked_scores)),
        "proper_mean": float(np.mean(proper_scores)),
        "proper_std": float(np.std(proper_scores)),
        "loo_acc": loo_acc,
        "loo_correct": loo_correct,
        "loo_total": loo_total,
    }


def find_impossible_mask(df: pd.DataFrame) -> pd.Series:
    """Identifica randurile cu valori numeric imposibile conform domeniului."""
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
    """Limiteaza valorile imposibile la intervalele fizice valide."""
    clipped = X.copy()
    non_negative_cols = [
        "form_fill_seconds",
        "email_domain_age_days",
        "ip_geo_distance_km",
        "address_history_years",
    ]
    unit_interval_cols = ["doc_scan_quality", "photo_match_score"]
    for col in non_negative_cols:
        clipped[col] = clipped[col].clip(lower=0.0)
    for col in unit_interval_cols:
        clipped[col] = clipped[col].clip(lower=0.0, upper=1.0)
    return clipped


def audit_false_lead_impossible_values(df: pd.DataFrame) -> dict[str, object]:
    """Testeaza pista falsa B: randurile cu valori imposibile (negative sau >1)."""
    X = df[FEATURE_COLS]
    y = df["label"]
    bad_mask = find_impossible_mask(df)

    breakdown = {
        "form_fill_seconds < 0": int((df["form_fill_seconds"] < 0).sum()),
        "doc_scan_quality < 0": int((df["doc_scan_quality"] < 0).sum()),
        "doc_scan_quality > 1": int((df["doc_scan_quality"] > 1).sum()),
        "photo_match_score < 0": int((df["photo_match_score"] < 0).sum()),
        "photo_match_score > 1": int((df["photo_match_score"] > 1).sum()),
        "email_domain_age_days < 0": int((df["email_domain_age_days"] < 0).sum()),
        "ip_geo_distance_km < 0": int((df["ip_geo_distance_km"] < 0).sum()),
        "address_history_years < 0": int((df["address_history_years"] < 0).sum()),
    }

    X_clean = X.loc[~bad_mask].reset_index(drop=True)
    y_clean = y.loc[~bad_mask].reset_index(drop=True)
    drop_split_acc, drop_split_correct, drop_split_total = evaluate_proper_split(X_clean, y_clean)
    drop_loo_acc, drop_loo_correct, drop_loo_total = evaluate_cv(X_clean, y_clean, LeaveOneOut())

    X_clipped = clip_impossible_values(X)
    clip_split_acc, clip_split_correct, clip_split_total = evaluate_proper_split(X_clipped, y)
    clip_loo_acc, clip_loo_correct, clip_loo_total = evaluate_cv(X_clipped, y, LeaveOneOut())

    return {
        "n_bad_rows": int(bad_mask.sum()),
        "n_total": len(df),
        "breakdown": breakdown,
        "drop_split_acc": drop_split_acc,
        "drop_split_correct": drop_split_correct,
        "drop_split_total": drop_split_total,
        "drop_loo_acc": drop_loo_acc,
        "drop_loo_correct": drop_loo_correct,
        "drop_loo_total": drop_loo_total,
        "clip_split_acc": clip_split_acc,
        "clip_split_correct": clip_split_correct,
        "clip_split_total": clip_split_total,
        "clip_loo_acc": clip_loo_acc,
        "clip_loo_correct": clip_loo_correct,
        "clip_loo_total": clip_loo_total,
    }


def audit_false_lead_split_config(
    X: pd.DataFrame,
    y: pd.Series,
    naive_loo: tuple[float, int, int],
) -> dict[str, float | int]:
    """Testeaza pista falsa C: lipsa stratify=y si split simplu vs CV ne-grupat."""
    strat_acc, strat_correct, strat_total = evaluate_proper_split(X, y, stratify=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    skf_acc, skf_correct, skf_total = evaluate_cv(X, y, skf)
    naive_loo_acc, naive_loo_correct, naive_loo_total = naive_loo
    return {
        "strat_acc": strat_acc,
        "strat_correct": strat_correct,
        "strat_total": strat_total,
        "skf_acc": skf_acc,
        "skf_correct": skf_correct,
        "skf_total": skf_total,
        "naive_loo_acc": naive_loo_acc,
        "naive_loo_correct": naive_loo_correct,
        "naive_loo_total": naive_loo_total,
    }


def audit_applicant_leakage(df: pd.DataFrame) -> dict[str, object]:
    """Demonstreaza bugul real: cereri multiple ale aceluiasi aplicant in train si test."""
    groups = df["date_of_birth"].astype(str) + "|" + df["citizenship"].astype(str)
    group_sizes = groups.value_counts()
    size_dist = {int(k): int(v) for k, v in group_sizes.value_counts().sort_index().items()}

    labels_per_group = df.groupby(groups)["label"].nunique()
    n_conflicting = int((labels_per_group > 1).sum())

    # Selecteaza un grup reprezentativ cu 4 cereri pentru exemplificare.
    sample_group = str(group_sizes[group_sizes == group_sizes.max()].index[0])
    sample_rows = df.loc[groups == sample_group]

    # Distante euclidiene in spatiul scalat: intra-aplicant vs inter-aplicant.
    X_scaled = StandardScaler().fit_transform(df[FEATURE_COLS])
    dist_matrix = pairwise_distances(X_scaled, metric="euclidean")
    tri_i, tri_j = np.triu_indices(len(df), k=1)
    same_group = groups.to_numpy()[tri_i] == groups.to_numpy()[tri_j]
    pair_dists = dist_matrix[tri_i, tri_j]
    max_within_dist = float(pair_dists[same_group].max())
    min_between_dist = float(pair_dists[~same_group].min())

    # Masoara contaminarea exacta din split-ul original random_state=1137.
    train_idx, test_idx = train_test_split(
        df.index.to_numpy(),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    model = make_model_pipeline()
    model.fit(df.loc[train_idx, FEATURE_COLS], df.loc[train_idx, "label"])
    test_preds = model.predict(df.loc[test_idx, FEATURE_COLS])
    y_test = df.loc[test_idx, "label"].to_numpy()

    train_groups = set(groups.iloc[train_idx])
    leaked_mask = groups.iloc[test_idx].isin(train_groups).to_numpy()

    n_test_total = len(test_idx)
    n_test_leaked = int(leaked_mask.sum())
    n_test_unseen = int((~leaked_mask).sum())
    leaked_acc = float(accuracy_score(y_test[leaked_mask], test_preds[leaked_mask]))
    unseen_acc = float(accuracy_score(y_test[~leaked_mask], test_preds[~leaked_mask]))

    return {
        "groups": groups,
        "n_rows": len(df),
        "n_unique_applicants": int(groups.nunique()),
        "size_dist": size_dist,
        "n_conflicting_labels": n_conflicting,
        "sample_group": sample_group.split("|"),
        "sample_indices": sample_rows.index.tolist(),
        "sample_subs": sample_rows["submission_id"].tolist(),
        "sample_labels": sample_rows["label"].tolist(),
        "max_within_dist": max_within_dist,
        "min_between_dist": min_between_dist,
        "n_test_total": n_test_total,
        "n_test_leaked": n_test_leaked,
        "pct_test_leaked": 100.0 * n_test_leaked / n_test_total,
        "leaked_acc": leaked_acc,
        "n_test_unseen": n_test_unseen,
        "pct_test_unseen": 100.0 * n_test_unseen / n_test_total,
        "unseen_acc": unseen_acc,
    }


def evaluate_corrected_pipeline(df: pd.DataFrame, groups: pd.Series) -> dict[str, object]:
    """Ruleaza pipeline-ul corectat: LeaveOneGroupOut si LeaveOneOut deduplicat."""
    X = df[FEATURE_COLS]
    y = df["label"]

    logo_acc, logo_correct, logo_total = evaluate_cv(
        X,
        y,
        LeaveOneGroupOut(),
        groups=groups,
    )

    df_unique = df.drop_duplicates(
        subset=["date_of_birth", "citizenship"],
        keep="first",
    ).reset_index(drop=True)
    dedup_acc, dedup_correct, dedup_total = evaluate_cv(
        df_unique[FEATURE_COLS],
        df_unique["label"],
        LeaveOneOut(),
    )

    flag_trunc = f"SIGMOID_{int(logo_acc * 100)}"
    flag_round = f"SIGMOID_{round(logo_acc * 100)}"

    return {
        "logo_acc": logo_acc,
        "logo_correct": logo_correct,
        "logo_total": logo_total,
        "dedup_acc": dedup_acc,
        "dedup_correct": dedup_correct,
        "dedup_total": dedup_total,
        "flag_trunc": flag_trunc,
        "flag_round": flag_round,
    }


def format_solution_text(results: dict[str, object]) -> str:
    """Construiește continutul complet pentru solution.txt din metricile calculate."""
    orig = results["orig"]
    fa = results["false_a"]
    fb = results["false_b"]
    fc = results["false_c"]
    leak = results["leak"]
    corr = results["corrected"]

    size_lines = ", ".join(
        f"{cnt} aplicanti cu {size} {'cerere' if size == 1 else 'cereri'}"
        for size, cnt in leak["size_dist"].items()
    )
    bad_nonzero = ", ".join(
        f"{name}: {cnt}"
        for name, cnt in fb["breakdown"].items()
        if cnt > 0
    )
    dob, citizenship = leak["sample_group"]

    lines = [
        "ESTONIA - DATELE ASTEA SUNT PERFECTE!",
        "Auditul pipeline-ului de detectare a fraudelor e-Residency",
        "================================================================",
        "",
        f"FLAG: {corr['flag_trunc']}",
        f"(Acuratete reala LeaveOneGroupOut = {corr['logo_acc']:.4f} -> trunchiere int(acc*100) ca in exemplul din enunt 0.8649 -> SIGMOID_86: {corr['flag_trunc']}; rotunjire la cel mai apropiat intreg: {corr['flag_round']}.)",
        "",
        "----------------------------------------------------------------",
        "1. REZUMAT EXECUTIV",
        "----------------------------------------------------------------",
        f"Pipeline-ul livrat (project_pipeline.py) raporteaza o acuratete pe test de {orig['acc']:.4f} ({orig['acc']*100:.2f}%, {orig['correct']}/{orig['total']} predictii corecte), valoare artificial umflata.",
        "Auditul celor trei alegeri discutabile arata clar ca doar UNA este responsabilă pentru scorul fals:",
        "- BUGUL ADEVARAT: Scurgere de date la nivel de entitate/aplicant (entity/applicant leakage) — campurile `date_of_birth` si `citizenship` sunt aruncate inainte de impartirea in train/test, desi acelasi aplicant apare cu multiple cereri aproape identice in setul de date.",
        "- PISTA FALSA A: Scurgere de preprocesare — `StandardScaler().fit_transform(X)` rulat pe tot setul de date inainte de `train_test_split`.",
        "- PISTA FALSA B: Valori numerice imposibile in `applications.csv` (timpi/varste/distante negative sau scoruri > 1) si/sau configuratia de split (lipsa `stratify=y` / split unic).",
        "",
        "----------------------------------------------------------------",
        "2. DOVEZI PENTRU PISTELE FALSE (SCORUL RAMANE ~93-96%)",
        "----------------------------------------------------------------",
        f"Baseline original (project_pipeline.py, rs={RANDOM_STATE}): {orig['acc']:.4f} ({orig['acc']*100:.2f}%, {orig['correct']}/{orig['total']})",
        "",
        "Pista falsa A — Scurgerea de preprocesare (StandardScaler inainte de split):",
        f"  - Reparare pe split-ul original (scaler fit-uit strict in train, rs={RANDOM_STATE}): {fa['fix_acc']:.4f} ({fa['fix_acc']*100:.2f}%, {fa['fix_correct']}/{fa['fix_total']})",
        f"  - Medie pe 100 de seed-uri: cu scurgere = {fa['leaked_mean']:.4f} +/- {fa['leaked_std']:.4f} vs. corect = {fa['proper_mean']:.4f} +/- {fa['proper_std']:.4f}",
        f"  - LeaveOneOut pe toate cele {leak['n_rows']} randuri (scaler in fiecare fold): {fa['loo_acc']:.4f} ({fa['loo_acc']*100:.2f}%, {fa['loo_correct']}/{fa['loo_total']})",
        "  Concluzie A: Scurgerea din StandardScaler nu umfla scorul; acuratetea ramane ~94-96%.",
        "",
        "Pista falsa B — Valori numerice imposibile in date:",
        f"  - Randuri cu cel putin o valoare in afara domeniului valid: {fb['n_bad_rows']} / {fb['n_total']} ({bad_nonzero})",
        f"  - Reparare B1 (eliminarea celor {fb['n_bad_rows']} randuri): split rs={RANDOM_STATE} -> {fb['drop_split_acc']:.4f} ({fb['drop_split_acc']*100:.2f}%, {fb['drop_split_correct']}/{fb['drop_split_total']}) | LeaveOneOut -> {fb['drop_loo_acc']:.4f} ({fb['drop_loo_acc']*100:.2f}%)",
        f"  - Reparare B2 (limitare / clipping la limite valide): split rs={RANDOM_STATE} -> {fb['clip_split_acc']:.4f} ({fb['clip_split_acc']*100:.2f}%, {fb['clip_split_correct']}/{fb['clip_split_total']}) | LeaveOneOut -> {fb['clip_loo_acc']:.4f} ({fb['clip_loo_acc']*100:.2f}%)",
        "  Concluzie B: Eliminarea sau corectarea anomaliilor numerice nu reduce scorul umflat (~93-96%).",
        "",
        "Pista falsa C — Configuratia de split (stratify / K-Fold / LOO fara grup):",
        f"  - Split rs={RANDOM_STATE} cu `stratify=y` si scaler in pipeline: {fc['strat_acc']:.4f} ({fc['strat_acc']*100:.2f}%, {fc['strat_correct']}/{fc['strat_total']})",
        f"  - StratifiedKFold(n_splits=5, shuffle=True, random_state={RANDOM_STATE}): {fc['skf_acc']:.4f} ({fc['skf_acc']*100:.2f}%, {fc['skf_correct']}/{fc['skf_total']})",
        f"  - LeaveOneOut naiv pe toate randurile (fara grupare): {fc['naive_loo_acc']:.4f} ({fc['naive_loo_acc']*100:.2f}%, {fc['naive_loo_correct']}/{fc['naive_loo_total']})",
        "  Concluzie C: Nici stratificarea, nici validarea incrucisata ne-grupata nu elimina umflarea scorului.",
        "",
        "----------------------------------------------------------------",
        "3. BUGUL ADEVARAT: SCURGEREA DE APLICANT IN TRAIN/TEST",
        "----------------------------------------------------------------",
        f"Desi `applications.csv` are {leak['n_rows']} randuri, perechea `(date_of_birth, citizenship)` identifica doar {leak['n_unique_applicants']} aplicanti unici:",
        f"  - Distributia cererilor per aplicant: {size_lines}.",
        f"  - Aplicanti cu etichete contradictorii intre cererile proprii: {leak['n_conflicting_labels']} (toti au eticheta consistenta).",
        f"  - Exemplu de grup: ({dob}, {citizenship}) -> {len(leak['sample_subs'])} cereri ({', '.join(leak['sample_subs'])}), toate cu label={leak['sample_labels'][0]}.",
        f"  - Distanta euclidiana in spatiul scalat: max intra-aplicant = {leak['max_within_dist']:.4f} < min inter-aplicant = {leak['min_between_dist']:.4f}.",
        "  - Consecinta in split-ul original (`random_state=1137`, 395 randuri in test):",
        f"    * {leak['n_test_leaked']} / {leak['n_test_total']} ({leak['pct_test_leaked']:.2f}%) au acelasi aplicant in train -> acuratete = {leak['leaked_acc']:.4f} ({leak['leaked_acc']*100:.2f}%)",
        f"    * {leak['n_test_unseen']} / {leak['n_test_total']} ({leak['pct_test_unseen']:.2f}%) sunt aplicanti nevazuti in train -> acuratete = {leak['unseen_acc']:.4f} ({leak['unseen_acc']*100:.2f}%)",
        "  Modelul 1-NN doar memoreaza clonele aproape identice ale aceluiasi aplicant intre train si test.",
        "",
        "----------------------------------------------------------------",
        "4. PIPELINE-UL CORECTAT",
        "----------------------------------------------------------------",
        "Solutia pastreaza exact modelul impus (`StandardScaler` + `KNeighborsClassifier(n_neighbors=1)` pe cele 6 caracteristici numerice), incapsulat intr-un `Pipeline`, si izoleaza aplicantul `(date_of_birth, citizenship)` intre train si validare cu `LeaveOneGroupOut()` (echivalentul `LeaveOneOut()` la nivel de aplicant) + `cross_val_predict`:",
        "",
        "```python",
        "import pandas as pd",
        "from sklearn.metrics import accuracy_score",
        "from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict",
        "from sklearn.neighbors import KNeighborsClassifier",
        "from sklearn.pipeline import make_pipeline",
        "from sklearn.preprocessing import StandardScaler",
        "",
        'FEATURE_COLS = [',
        '    "form_fill_seconds",',
        '    "doc_scan_quality",',
        '    "photo_match_score",',
        '    "email_domain_age_days",',
        '    "ip_geo_distance_km",',
        '    "address_history_years",',
        ']',
        "",
        'df = pd.read_csv("applications.csv")',
        'X = df[FEATURE_COLS]',
        'y = df["label"]',
        'groups = df["date_of_birth"].astype(str) + "|" + df["citizenship"].astype(str)',
        "",
        "pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1))",
        "oof_preds = cross_val_predict(pipeline, X, y, cv=LeaveOneGroupOut(), groups=groups)",
        "real_accuracy = accuracy_score(y, oof_preds)",
        'print(f"Real LeaveOneGroupOut accuracy: {real_accuracy:.4f}")',
        'print(f"Flag: SIGMOID_{int(real_accuracy * 100)}")',
        "```",
        "",
        "----------------------------------------------------------------",
        "5. VALOAREA REALA A PERFORMANTEI",
        "----------------------------------------------------------------",
        f"- LeaveOneGroupOut() pe toate cele {corr['logo_total']} cereri (grupat pe aplicant): {corr['logo_acc']:.4f} ({corr['logo_acc']*100:.2f}%, {corr['logo_correct']}/{corr['logo_total']} corecte)",
        f"- LeaveOneOut() pe cei {corr['dedup_total']} aplicanti deduplicati (keep='first'): {corr['dedup_acc']:.4f} ({corr['dedup_acc']*100:.2f}%, {corr['dedup_correct']}/{corr['dedup_total']} corecte)",
        f"- Concluzie finala: Acuratetea reala a sistemului pe aplicanti noi este ~{corr['logo_acc']*100:.2f}% (cu ~17 puncte procentuale sub cei {orig['acc']*100:.2f}% raportati initial), iar flagul este {corr['flag_trunc']} (sau {corr['flag_round']} la rotunjire la cel mai apropiat intreg).",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """Ruleaza auditul complet, afiseaza rezultatele si scrie solution.txt."""
    df = load_dataset()
    X = df[FEATURE_COLS]
    y = df["label"]

    orig_acc, orig_correct, orig_total = evaluate_original_pipeline(X, y)
    false_a = audit_false_lead_preprocessing(X, y)
    false_b = audit_false_lead_impossible_values(df)
    false_c = audit_false_lead_split_config(
        X,
        y,
        (
            float(false_a["loo_acc"]),
            int(false_a["loo_correct"]),
            int(false_a["loo_total"]),
        ),
    )
    leak = audit_applicant_leakage(df)
    corrected = evaluate_corrected_pipeline(df, leak["groups"])

    results = {
        "orig": {
            "acc": orig_acc,
            "correct": orig_correct,
            "total": orig_total,
        },
        "false_a": false_a,
        "false_b": false_b,
        "false_c": false_c,
        "leak": leak,
        "corrected": corrected,
    }

    report = format_solution_text(results)
    SOLUTION_PATH.write_text(report, encoding="utf-8")

    print(f"Original pipeline accuracy: {orig_acc:.4f} ({orig_correct}/{orig_total})")
    print(f"False Lead A (scaler in pipeline, rs={RANDOM_STATE}): {false_a['fix_acc']:.4f} | LOO: {false_a['loo_acc']:.4f}")
    print(f"False Lead B (drop {false_b['n_bad_rows']} bad rows): {false_b['drop_split_acc']:.4f} | clip: {false_b['clip_split_acc']:.4f}")
    print(f"False Lead C (stratify={True}): {false_c['strat_acc']:.4f} | 5-fold SKF: {false_c['skf_acc']:.4f}")
    print(
        f"True Bug (test leaked {leak['n_test_leaked']}/{leak['n_test_total']}): "
        f"leaked acc={leak['leaked_acc']:.4f} vs unseen acc={leak['unseen_acc']:.4f}"
    )
    print(
        f"Corrected LeaveOneGroupOut: {corrected['logo_acc']:.4f} "
        f"({corrected['logo_correct']}/{corrected['logo_total']}) -> {corrected['flag_trunc']} "
        f"(rounded: {corrected['flag_round']})"
    )
    print(
        f"Corrected Deduplicated LeaveOneOut: {corrected['dedup_acc']:.4f} "
        f"({corrected['dedup_correct']}/{corrected['dedup_total']})"
    )
    print(f"[OK] Fisierul solution.txt a fost generat: {SOLUTION_PATH} ({SOLUTION_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
