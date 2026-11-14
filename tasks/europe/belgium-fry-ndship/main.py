"""Modelează lungimea marginii parcelelor din dimensiunile geometrice și dezvăluie mesajele ascunse în reziduuri."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib
# Folosim backend-ul non-interactiv Agg pentru a genera imagini PNG fără fereastră grafică
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error


# Directorul curent al scriptului și calea către setul de date de intrare
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "potato_fields.csv"


@dataclass(frozen=True)
class ModelSpec:
    """Configurație imuabilă pentru un model de regresie liniară și graficul său de reziduuri."""

    key: str
    title: str
    features: tuple[str, ...]
    inner_band: tuple[float, float] | None
    flag: str | None
    output_filename: str | None


@dataclass
class ModelResult:

    spec: ModelSpec
    model: LinearRegression
    y_pred: np.ndarray
    residuals: np.ndarray
    r2: float
    rmse: float


def load_and_engineer_features(csv_path: Path) -> pd.DataFrame:
    # Încărcăm setul de date CSV cu parcelele de cartofi măsurate pe teren
    df = pd.read_csv(csv_path)
    # Convertim aria din hectare în m² (1 ha = 10.000 m²) și împărțim la latura lungă (L)
    # pentru a obține latura scurtă a parcelei dreptunghiulare: W = (area_ha * 10000) / L
    df["short_side_m"] = (df["area_ha"] * 10000.0) / df["long_side_m"]
    # Calculăm perimetrul teoretic complet al dreptunghiului: P = 2 * (L + W)
    df["perim_rect_m"] = 2.0 * (df["long_side_m"] + df["short_side_m"])
    return df


def fit_model(df: pd.DataFrame, spec: ModelSpec, target: str = "edge_m") -> ModelResult:
    # Selectăm matricea de trăsături (X) și vectorul țintă real (edge_m)
    x = df[list(spec.features)]
    y = df[target].to_numpy()
    # Ajustăm regresia liniară prin metoda celor mai mici pătrate (OLS)
    reg = LinearRegression().fit(x, y)
    # Calculăm predicțiile modelului și reziduurile (diferența real - prezis)
    y_pred = reg.predict(x)
    residuals = y - y_pred
    # Evaluăm calitatea ajustării prin coeficientul de determinare R² și RMSE
    r2 = float(reg.score(x, y))
    rmse = float(root_mean_squared_error(y, y_pred))
    return ModelResult(
        spec=spec,
        model=reg,
        y_pred=y_pred,
        residuals=residuals,
        r2=r2,
        rmse=rmse,
    )


def format_equation(result: ModelResult) -> str:
    """Formatează ecuația regresiei liniare ajustate sub formă de text lizibil."""
    # Construim perechile coeficient * nume_trăsătură cu 4 zecimale
    terms = [
        f"{coef:+.4f}*{name}"
        for name, coef in zip(result.spec.features, result.model.coef_)
    ]
    # Concatenăm termenul liber (intercept) și termenii liniari ai modelului
    return f"edge_m ≈ {result.model.intercept_:+.4f} " + " ".join(terms)


def plot_residual_axis(ax: plt.Axes, result: ModelResult) -> None:
    """Desenează graficul reziduurilor în funcție de valorile ajustate pe o axă matplotlib."""
    # Reprezentăm pe axa X valoarea prezisă y_hat (nu perimetrul brut, deoarece straturile 2 și 3
    # includ decalaje din ditch_m/hedge_m și sunt aliniate precis față de propriul y_hat)
    x = result.y_pred
    e = result.residuals
    band = result.spec.inner_band

    if band is not None:
        lo, hi = band
        # Separăm banda centrală (care formează literele) de benzile exterioare de compensare
        # care mențin media și covarianța reziduurilor egale cu zero
        in_band = (e > lo) & (e < hi)
        # Punctele din benzile exterioare sunt desenate discret în gri deschis
        ax.scatter(
            x[~in_band],
            e[~in_band],
            s=6,
            color="#9aa5b1",
            alpha=0.45,
            edgecolors="none",
            label="Outer compensating bars",
        )
        # Punctele din banda interioară sunt desenate cu contrast puternic pentru a citi ușor cuvântul
        ax.scatter(
            x[in_band],
            e[in_band],
            s=8,
            color="#0f172a",
            alpha=0.9,
            edgecolors="none",
            label="Inner message band",
        )
    else:
        # Pentru modelul naiv nu există bandă separată, deci afișăm toate punctele uniform
        ax.scatter(
            x,
            e,
            s=7,
            color="#334155",
            alpha=0.65,
            edgecolors="none",
        )

    # Trasăm linia orizontală de referință pentru reziduu zero (e = 0)
    ax.axhline(0.0, color="#dc2626", linestyle="--", linewidth=0.9, alpha=0.6)
    # Etichetăm axele cu valorile ajustate și reziduurile exprimate în metri
    ax.set_xlabel("Fitted $\\hat{y}$ (m)", fontsize=10)
    ax.set_ylabel("Residual $e = y - \\hat{y}$ (m)", fontsize=10)

    # Adăugăm metrica R², RMSE și flagul descoperit (dacă există) în titlul panoului
    subtitle = f"$R^2 = {result.r2:.8f}$ | RMSE = {result.rmse:.4f} m"
    if result.spec.flag:
        subtitle += f" | Flag: {result.spec.flag}"
    ax.set_title(f"{result.spec.title}\n{subtitle}", fontsize=11, fontweight="bold", pad=6)
    # Activăm o grilă discretă pentru orientare vizuală
    ax.grid(True, linestyle=":", alpha=0.4)


def save_individual_plot(result: ModelResult, output_path: Path) -> None:
    """Salvează un grafic individual cu raport de aspect lat pentru un singur strat al modelului."""
    # Raportul lat (15 x 3.8) păstrează literele necomprimate pe orizontală
    fig, ax = plt.subplots(figsize=(15, 3.8), constrained_layout=True)
    plot_residual_axis(ax, result)
    # Exportăm imaginea la rezoluție înaltă (200 DPI) și eliberăm memoria figurii
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_combined_figure(results: Sequence[ModelResult], output_path: Path) -> None:
    """Salvează figura comparativă cu toate modelele suprapuse vertical."""
    # Creăm o figură cu câte un rând pentru fiecare model analizat
    fig, axes = plt.subplots(len(results), 1, figsize=(16, 13), constrained_layout=True)
    for ax, res in zip(axes, results):
        plot_residual_axis(ax, res)
    # Setăm titlul general al figurii comparative și exportăm imaginea finală
    fig.suptitle(
        "Belgian Potato Fields (Fry-ndship): Nested Residual Structure of Parcel Edge Models",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    """Ajustează toate modelele, afișează metricile în consolă și generează figurile de diagnostic."""
    # Încărcăm datele și derivăm dimensiunile geometrice necesare modelării
    df = load_and_engineer_features(DATA_PATH)
    print(f"Loaded {len(df)} parcels from {DATA_PATH.name}\n")

    # Definim cele 4 modele: modelul liniar naiv și cele 3 straturi geometrice imbricate
    specs = (
        ModelSpec(
            # Model naiv pe coloane brute: explică ~99% din variație, dar curbura A/L ascunde mesajul
            key="naive",
            title="Baseline Naive Linear Model (edge_m ~ area_ha + long_side_m)",
            features=("area_ha", "long_side_m"),
            inner_band=None,
            flag=None,
            output_filename=None,
        ),
        ModelSpec(
            # Stratul 1: doar dimensiunile dreptunghiului (L și W) -> reziduul scrie REGRESSION
            key="layer1",
            title="Layer 1 — Geometric Dimensions (edge_m ~ long_side_m + short_side_m)",
            features=("long_side_m", "short_side_m"),
            inner_band=(-12.5, 12.5),
            flag="SIGMOID_REGRESSION",
            output_filename="layer1_regression.png",
        ),
        ModelSpec(
            # Stratul 2: adăugăm șanțul declarat de fermier (ditch_m) -> reziduul scrie SUPERVISION
            key="layer2",
            title="Layer 2 — Farmer Declarations (edge_m ~ long_side_m + short_side_m + ditch_m)",
            features=("long_side_m", "short_side_m", "ditch_m"),
            inner_band=(-1.10, 1.10),
            flag="SIGMOID_SUPERVISION",
            output_filename="layer2_supervision.png",
        ),
        ModelSpec(
            # Stratul 3: includem și gardul viu măsurat de Mariette (hedge_m) -> reziduul scrie PREDICTION
            key="layer3",
            title="Layer 3 — Full Model w/ Hedge (edge_m ~ long_side_m + short_side_m + ditch_m + hedge_m)",
            features=("long_side_m", "short_side_m", "ditch_m", "hedge_m"),
            inner_band=(-0.072, 0.058),
            flag="SIGMOID_PREDICTION",
            output_filename="layer3_prediction.png",
        ),
    )

    results: list[ModelResult] = []
    # Antrenăm pe rând fiecare model, afișăm ecuația și metricile, apoi salvăm graficele individuale
    for spec in specs:
        res = fit_model(df, spec)
        results.append(res)
        print(f"[{spec.key}] {spec.title}")
        print(f"  Equation : {format_equation(res)}")
        print(f"  R^2      : {res.r2:.8f}")
        print(f"  RMSE     : {res.rmse:.6f} m")
        print(f"  Res range: [{res.residuals.min():+.4f}, {res.residuals.max():+.4f}] m")
        if spec.flag:
            print(f"  Flag     : {spec.flag}")
        print()

        # Salvăm figura individuală doar pentru straturile care au nume de fișier configurat
        if spec.output_filename:
            save_individual_plot(res, BASE_DIR / spec.output_filename)

    # Generăm figura finală combinată cu toate cele 4 panouri de reziduuri
    save_combined_figure(results, BASE_DIR / "residuals_fry_ndship.png")
    print("Generated residuals_fry_ndship.png and layer1/layer2/layer3 PNGs.")

if __name__ == "__main__":
    main()
