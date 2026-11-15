"""Evaluarea și compararea celor trei modele-oracol (Ares, Athena, Apollo) pentru paza Sanctuarului de la Delfi."""

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# Import opțional matplotlib pentru generarea graficelor (degradare elegantă dacă lipsește din mediul curent)
try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Încercăm importul PyTorch și al definiției modelului din model.py.
# Dacă PyTorch nu este instalat în mediul curent (ex. Python 3.14), folosim citirea
# directă a tensorilor din arhiva ZIP a fișierului .pth și inferența echivalentă în NumPy.
try:
    import torch
    from model import FEATURE_COLUMNS, MythicThreatClassifier

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    FEATURE_COLUMNS = [
        "creature_mass_talents",
        "oracle_omen",
        "aggression_index",
        "divine_aura",
        "aegis_resonance",
        "armor_resistance",
        "witness_count",
        "styx_disturbance",
        "omen_intensity",
    ]

# Pragul de decizie impus în enunț pentru clasificarea binară pe probabilități
DECISION_THRESHOLD = 0.50

# Dicționar cu numele modelelor și căile relative către fișierele de ponderi
ORACLE_MODELS = {
    "ARES": Path("models/model_ares.pth"),
    "ATHENA": Path("models/model_athena.pth"),
    "APOLLO": Path("models/model_apollo.pth"),
}


def load_test_data(dataset_path: Path, labels_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Încarcă setul de test și etichetele adevărate, aliniate strict după encounter_id."""
    df_features = pd.read_csv(dataset_path)
    df_labels = pd.read_csv(labels_path)

    # Îmbinăm cele două tabele după identificatorul unic al întâlnirii pentru a garanta corespondența exactă
    df = df_features.merge(df_labels, on="encounter_id", how="inner", validate="one_to_one")
    if len(df) != len(df_features) or len(df) != len(df_labels):
        raise ValueError("Nepotrivire între rândurile din test_dataset.csv și test_labels.csv.")

    # Extragem matricea de caracteristici în ordinea exactă din FEATURE_COLUMNS și vectorul țintă binar
    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = df["is_catastrophic_threat"].to_numpy(dtype=np.int64)
    return X, y


def _extract_linear_params_from_zip(checkpoint_path: Path) -> tuple[np.ndarray, float]:
    """Extrage vectorul de ponderi și bias-ul float32 direct din arhiva ZIP a unui fișier .pth PyTorch."""
    with zipfile.ZipFile(checkpoint_path, "r") as archive:
        # În formatul standard torch.save pe un state_dict cu 2 tensori (linear.weight, linear.bias),
        # fișierele binare din directorul */data/ stochează tensorii în ordinea definiției din modul.
        data_entries = sorted(name for name in archive.namelist() if "/data/" in name and not name.endswith("/"))
        tensors = [np.frombuffer(archive.read(entry), dtype=np.float32) for entry in data_entries]

    # Identificăm vectorul de ponderi (dimensiune = numărul de caracteristici) și scalarul bias (dimensiune = 1)
    weights = next((t.copy() for t in tensors if t.size == len(FEATURE_COLUMNS)), None)
    bias_arr = next((t.copy() for t in tensors if t.size == 1), None)
    if weights is None or bias_arr is None:
        raise ValueError(f"Nu s-au putut extrage parametrii liniari din {checkpoint_path}.")
    return weights, float(bias_arr[0])


def predict_with_oracle(checkpoint_path: Path, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Rulează inferența unui model-oracol pe matricea X și întoarce (probabilități, logiți, ponderi, bias)."""
    if HAS_TORCH:
        # Încărcăm state_dict-ul PyTorch pe CPU și îl injectăm în arhitectura comună
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = MythicThreatClassifier()
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.eval()

        # Dezactivăm calculul gradienților pe durata evaluării
        with torch.no_grad():
            x_tensor = torch.from_numpy(X)
            logits_tensor = model(x_tensor)
            probs_tensor = torch.sigmoid(logits_tensor)

        weights = model.linear.weight.detach().cpu().numpy().reshape(-1)
        bias = float(model.linear.bias.detach().cpu().item())
        return probs_tensor.numpy(), logits_tensor.numpy(), weights, bias

    # Calea fallback în NumPy pur, matematic identică cu stratul nn.Linear + torch.sigmoid
    weights, bias = _extract_linear_params_from_zip(checkpoint_path)
    logits = X @ weights + bias
    # Calculăm funcția sigmoid numeric stabilă pentru a transforma logiții în probabilități [0, 1]
    probs = np.where(logits >= 0, 1.0 / (1.0 + np.exp(-logits)), np.exp(logits) / (1.0 + np.exp(logits)))
    return probs, logits, weights, bias


def compute_metrics(name: str, y_true: np.ndarray, probs: np.ndarray, weights: np.ndarray, bias: float) -> dict:
    """Calculează metricile de clasificare la pragul de 0.50 și scorurile independente de prag."""
    # Aplicăm pragul de decizie 0.50 asupra probabilităților obținute din sigmoid
    y_pred = (probs >= DECISION_THRESHOLD).astype(np.int64)

    # Extragem cele 4 cadrane ale matricei de confuzie binare
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Metrici de bază și metrici sensibile la costul asimetric al erorilor (FN >> FP)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    miss_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    # F2-score acordă o pondere dublă Recall-ului față de Precizie (beta = 2)
    f2 = fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)

    # Metrici de clasare globale pe probabilități (independente de prag)
    roc_auc = roc_auc_score(y_true, probs)
    pr_auc = average_precision_score(y_true, probs)

    # Identificăm caracteristica dominantă folosită de fiecare model liniar
    active_idx = int(np.argmax(np.abs(weights)))
    active_feature = FEATURE_COLUMNS[active_idx]

    return {
        "model": name,
        "active_feature": active_feature,
        "bias": round(bias, 4),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "accuracy": round(float(accuracy), 4),
        "balanced_accuracy": round(float(balanced_acc), 4),
        "recall_tpr": round(float(recall), 4),
        "miss_rate_fnr": round(float(miss_rate), 4),
        "precision": round(float(precision), 4),
        "specificity_tnr": round(float(specificity), 4),
        "f1_score": round(float(f1), 4),
        "f2_score": round(float(f2), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
    }


def save_comparison_plot(y_true: np.ndarray, results: list[dict], probs_by_model: dict[str, np.ndarray], out_path: Path) -> None:
    """Generează și salvează graficele comparative: matrici de confuzie, metrici și curbe ROC/PR."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        "Evaluarea oracolelor din Delfi (prag decizie = 0.50 | 100 amenințări reale vs. 900 întâlniri sigure)",
        fontsize=13,
        fontweight="bold",
    )

    # Rândul 1: Matricile de confuzie pentru fiecare dintre cele trei modele
    for col_idx, res in enumerate(results):
        ax = axes[0, col_idx]
        cm = np.array([[res["TN"], res["FP"]], [res["FN"], res["TP"]]])
        ax.imshow(cm, cmap="Blues", aspect="auto")
        ax.set_title(f"{res['model']} (folosește: {res['active_feature']})\nFN={res['FN']} | Recall={res['recall_tpr']:.0%}", fontsize=10)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predis: 0 (Sigur)", "Predis: 1 (Amenințare)"])
        ax.set_yticklabels(["Real: 0", "Real: 1"])

        labels = [["TN", "FP"], ["FN\n(RATAT!)", "TP"]]
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                color = "white" if val > cm.max() / 2 else "black"
                if i == 1 and j == 0:
                    color = "crimson" if val > cm.max() / 2 else "darkred"
                ax.text(j, i, f"{labels[i][j]}\n{val}", ha="center", va="center", color=color, fontweight="bold")

    # Rândul 2, Coloana 1: Comparație metrici cheie sub formă de bare grupate
    ax_bar = axes[1, 0]
    metric_keys = ["accuracy", "recall_tpr", "precision", "f1_score", "f2_score", "balanced_accuracy"]
    metric_labels = ["Accuracy", "Recall", "Precision", "F1", "F2", "Bal. Acc"]
    x = np.arange(len(metric_keys))
    width = 0.25
    colors = {"ARES": "#d62728", "ATHENA": "#1f77b4", "APOLLO": "#2ca02c"}

    for idx, res in enumerate(results):
        vals = [res[k] for k in metric_keys]
        ax_bar.bar(x + (idx - 1) * width, vals, width, label=res["model"], color=colors[res["model"]], alpha=0.88)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metric_labels, rotation=20, ha="right")
    ax_bar.set_ylim(0, 1.08)
    ax_bar.set_ylabel("Scor")
    ax_bar.set_title("Metricile ascunse în spatele acurateței de 95%")
    ax_bar.grid(axis="y", linestyle="--", alpha=0.4)
    ax_bar.legend()

    # Rândul 2, Coloana 2: Curbele ROC
    ax_roc = axes[1, 1]
    for res in results:
        name = res["model"]
        fpr, tpr, _ = roc_curve(y_true, probs_by_model[name])
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {res['roc_auc']:.4f})", color=colors[name], linewidth=2)
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax_roc.set_xlabel("False Positive Rate (1 - Specificitate)")
    ax_roc.set_ylabel("True Positive Rate (Recall)")
    ax_roc.set_title("Curba ROC")
    ax_roc.grid(alpha=0.3)
    ax_roc.legend(loc="lower right")

    # Rândul 2, Coloana 3: Curbele Precision-Recall
    ax_pr = axes[1, 2]
    for res in results:
        name = res["model"]
        prec_curve, rec_curve, _ = precision_recall_curve(y_true, probs_by_model[name])
        ax_pr.plot(rec_curve, prec_curve, label=f"{name} (AP = {res['pr_auc']:.4f})", color=colors[name], linewidth=2)
    ax_pr.set_xlabel("Recall (Sensibilitate)")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Curba Precision-Recall")
    ax_pr.grid(alpha=0.3)
    ax_pr.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Punctul principal de execuție: încarcă datele, evaluează oracolele și alege modelul optim."""
    base_dir = Path(__file__).resolve().parent
    X, y_true = load_test_data(base_dir / "test_dataset.csv", base_dir / "test_labels.csv")

    pos_count = int(y_true.sum())
    neg_count = int(len(y_true) - pos_count)
    print(f"Set de test încărcat: {len(y_true)} întâlniri ({pos_count} amenințări catastrofale, {neg_count} întâlniri normale).")
    print(f"Backend inferență: {'PyTorch' if HAS_TORCH else 'NumPy (fallback echivalent din arhiva .pth)'}\n")

    results = []
    probs_by_model = {}

    for model_name, rel_path in ORACLE_MODELS.items():
        probs, _, weights, bias = predict_with_oracle(base_dir / rel_path, X)
        probs_by_model[model_name] = probs
        metrics = compute_metrics(model_name, y_true, probs, weights, bias)
        results.append(metrics)

    # Construim DataFrame-ul cu toate metricile și îl salvăm ca dovadă CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv(base_dir / "metrics_comparison.csv", index=False)

    # Afișăm tabelul comparativ complet în consolă
    print("=== Comparația metricilor la pragul 0.50 ===")
    print(df_results.to_string(index=False))

    # Alegem modelul care minimizează rata de ratare a amenințărilor catastrofale (FN minim / Recall maxim),
    # departajând eventualele egalități prin F2-Score și ROC-AUC.
    best_row = max(results, key=lambda r: (r["recall_tpr"], r["f2_score"], r["roc_auc"]))
    flag = f"SIGMOID_{best_row['model']}"

    # Salvăm graficul comparativ (dacă matplotlib este disponibil) și fișierul cu flag-ul rezultat
    if HAS_MATPLOTLIB:
        save_comparison_plot(y_true, results, probs_by_model, base_dir / "evaluation_comparison.png")
    else:
        print("\n[Info] matplotlib nu este disponibil în acest interpreter; graficul PNG a fost omis.")
    (base_dir / "flag.txt").write_text(f"{flag}\n", encoding="utf-8")

    print("\n=== Concluzie ===")
    print(
        f"Deși toate cele 3 modele au exact {best_row['accuracy']:.2%} acuratețe (50 de erori din 1000), "
        f"{best_row['model']} are cel mai mare Recall ({best_row['recall_tpr']:.2%}), cele mai puține "
        f"amenințări ratate (FN = {best_row['FN']}), cel mai mare F2-Score ({best_row['f2_score']:.4f}), "
        f"cel mai mare F1-Score ({best_row['f1_score']:.4f}) și cel mai mare ROC-AUC ({best_row['roc_auc']:.4f})."
    )
    print(f"FLAG: {flag}")


if __name__ == "__main__":
    main()
