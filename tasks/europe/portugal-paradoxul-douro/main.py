# Portugalia - "Paradoxul Douro" (Paradoxul lui Simpson)
#
# Descriere algoritm:
# 1. Incarcam setul de date experimental (douro_vineyard_data.csv, 3.543 parcele)
#    si calculam ratele globale de succes (harvest_success) pentru treatment vs. control.
# 2. Verificam toate coloanele categorice grupand datele dupa valorile lor distincte
#    si numaram in cate grupuri apare inversarea (control > treatment).
# 3. Pentru toate coloanele numerice aplicam un algoritm de programare dinamica (DP)
#    care cauta daca exista macar o partitionare in >= 2 intervale contigue in care
#    rata de succes a metodei control este strict mai mare decat a metodei treatment
#    in absolut fiecare interval (garantand minim min_parcele_grup parcele per brat).
# 4. Demonstram ca "distance_from_porto_km" este singura coloana care permite o
#    inversare completa de tip Simpson si detectam automat golul natural din
#    distributie (intre 104,0 km si 114,0 km) care separa cele doua regimuri.
# 5. Analizam atat impartirea naturala in 2 grupuri (prag 110 km), cat si impartirea
#    in 3 subregiuni geografice (70-110 km, 110-145 km, 145-205 km) si explicam
#    dezechilibrul de alocare care genereaza paradoxul la comasarea datelor.

from pathlib import Path
import numpy as np
import pandas as pd

# Directorul curent al scriptului, folosit pentru cai relative sigure
DIRECTOR_PROIECT = Path(__file__).resolve().parent

COLOANE_CATEGORICE = [
    "trellis_system",
    "vineyard_age_group",
    "cooperative",
    "grape_variety",
    "organic_certified",
    "harvest_month",
    "slope_aspect",
]

COLOANE_NUMERICE = [
    "soil_depth_cm",
    "sun_hours_july",
    "vine_density_per_ha",
    "annual_rainfall_mm",
    "distance_from_porto_km",
    "elevation_m",
    "soil_ph",
]


def incarca_datele() -> pd.DataFrame:
    return pd.read_csv("douro_vineyard_data.csv")


def calculeaza_statistici_globale(df: pd.DataFrame) -> dict:
    """Calculeaza numarul de parcele, succesele si ratele globale per metoda."""
    # Agregam datele dupa coloana "treatment" fara nicio segmentare prealabila
    agregat = df.groupby("treatment")["harvest_success"].agg(["count", "sum", "mean"])

    nr_control = int(agregat.loc["control", "count"])
    succ_control = int(agregat.loc["control", "sum"])
    rata_control = float(agregat.loc["control", "mean"])

    nr_treat = int(agregat.loc["treatment", "count"])
    succ_treat = int(agregat.loc["treatment", "sum"])
    rata_treat = float(agregat.loc["treatment", "mean"])

    return {
        "total_parcele": len(df),
        "nr_control": nr_control,
        "succ_control": succ_control,
        "rata_control": rata_control,
        "nr_treatment": nr_treat,
        "succ_treatment": succ_treat,
        "rata_treatment": rata_treat,
        "diferenta_pp": (rata_treat - rata_control) * 100.0,
    }


def verifica_coloane_categorice(df: pd.DataFrame) -> list:
    """
    Testeaza fiecare coloana categorica pentru a verifica daca impartirea pe
    categoriile ei produce o inversare completa (control > treatment in fiecare grup).
    """
    rezultate = []
    for col in COLOANE_CATEGORICE:
        # Calculam numarul de parcele si rata de succes pentru fiecare (categorie, metoda)
        tabel = (
            df.groupby([col, "treatment"])["harvest_success"]
            .agg(["count", "sum", "mean"])
            .unstack()
        )
        # Diferenta pozitiva inseamna ca treatment castiga; negativa inseamna control castiga
        diff = tabel["mean"]["treatment"] - tabel["mean"]["control"]
        grupuri_inversate = int((diff < 0).sum())
        total_grupuri = int(len(diff))
        inversare_completa = (grupuri_inversate == total_grupuri) and (total_grupuri > 0)

        rezultate.append(
            {
                "coloana": col,
                "grupuri_inversate": grupuri_inversate,
                "total_grupuri": total_grupuri,
                "inversare_completa": inversare_completa,
                "detalii": tabel,
            }
        )
    return rezultate


def cauta_inversare_simpson_dp(
    df: pd.DataFrame, coloana: str, min_parcele_grup: int = 20
) -> dict:
    """
    Algoritm de programare dinamica (DP) pentru a verifica riguros daca o coloana
    numerica admite O partitionare in k >= 2 intervale contigue astfel incat:
      - fiecare interval contine cel putin `min_parcele_grup` parcele pentru ambele metode;
      - in absolut fiecare interval, rata de succes pentru `control` > rata pentru `treatment`.

    Principiul algoritmului:
      1. Sortam valorile unice v_0 < v_1 < ... < v_{m-1} ale coloanei.
      2. Construim sume prefixate pentru numarul de parcele (pc, pt) si numarul de
         succese (psc, pst) pentru `control` si `treatment`.
      3. Definim dp[j] = numarul maxim de intervale valide in care poate fi impartit
         prefixul de valori [0, j), sau -1 daca prefixul nu are o partitionare valida.
         Tranzitie: pentru orice i < j cu dp[i] >= 0, daca intervalul [i, j) are
         nc >= min_parcele_grup, nt >= min_parcele_grup si (sc / nc) > (st / nt)
         (verificat exact in intregi prin sc * nt > st * nc pentru a evita erori
         de virgula mobila), actualizam dp[j] = max(dp[j], dp[i] + 1).
    """
    # Grupam mai intai la nivel de valoare unica a coloanei numerice
    agregat = (
        df.groupby([coloana, "treatment"])["harvest_success"]
        .agg(["count", "sum"])
        .unstack(fill_value=0)
        .sort_index()
    )
    valori = agregat.index.to_numpy()
    n_c = agregat["count"]["control"].to_numpy(dtype=np.int64)
    n_t = agregat["count"]["treatment"].to_numpy(dtype=np.int64)
    s_c = agregat["sum"]["control"].to_numpy(dtype=np.int64)
    s_t = agregat["sum"]["treatment"].to_numpy(dtype=np.int64)

    # Sume prefixate pentru evaluarea in O(1) a oricarui subinterval [i, j)
    pc = np.concatenate([[0], np.cumsum(n_c)])
    pt = np.concatenate([[0], np.cumsum(n_t)])
    psc = np.concatenate([[0], np.cumsum(s_c)])
    pst = np.concatenate([[0], np.cumsum(s_t)])

    m = len(valori)
    dp = np.full(m + 1, -1, dtype=np.int64)
    parinte = np.full(m + 1, -1, dtype=np.int64)
    dp[0] = 0

    for j in range(1, m + 1):
        for i in range(j):
            if dp[i] < 0:
                continue
            nc = pc[j] - pc[i]
            nt = pt[j] - pt[i]
            if nc < min_parcele_grup or nt < min_parcele_grup:
                continue
            sc = psc[j] - psc[i]
            st = pst[j] - pst[i]
            # Conditia stricta sc/nc > st/nt devine sc * nt > st * nc (aritmetica exacta)
            if sc * nt > st * nc:
                if dp[i] + 1 > dp[j]:
                    dp[j] = dp[i] + 1
                    parinte[j] = i

    # Reconstruim intervalele daca exista o partitionare in cel putin 2 grupuri
    max_intervale = int(dp[m])
    intervale = []
    if max_intervale >= 2:
        curent = m
        while curent > 0:
            prev = int(parinte[curent])
            nc = int(pc[curent] - pc[prev])
            nt = int(pt[curent] - pt[prev])
            sc = int(psc[curent] - psc[prev])
            st = int(pst[curent] - pst[prev])
            intervale.append(
                {
                    "min_val": float(valori[prev]),
                    "max_val": float(valori[curent - 1]),
                    "nc": nc,
                    "sc": sc,
                    "rata_control": sc / nc,
                    "nt": nt,
                    "st": st,
                    "rata_treatment": st / nt,
                }
            )
            curent = prev
        intervale.reverse()

    # Calculam si cel mai mare gol natural (diferenta dintre valori consecutive sortate)
    diferente = np.diff(valori)
    idx_gol = int(np.argmax(diferente)) if len(diferente) > 0 else 0
    gol_max = float(diferente[idx_gol]) if len(diferente) > 0 else 0.0
    capat_stanga = float(valori[idx_gol]) if len(diferente) > 0 else float(valori[0])
    capat_dreapta = float(valori[idx_gol + 1]) if len(diferente) > 0 else float(valori[0])

    return {
        "coloana": coloana,
        "are_inversare_completa": max_intervale >= 2,
        "max_intervale": max_intervale,
        "intervale": intervale,
        "gol_max": gol_max,
        "gol_interval": (capat_stanga, capat_dreapta),
        "corelatie_treatment": float(df[coloana].corr((df["treatment"] == "treatment").astype(int))),
        "corelatie_succes": float(df[coloana].corr(df["harvest_success"])),
    }


def evalueaza_impartire_intervale(df: pd.DataFrame, coloana: str, praguri: list, etichete: list) -> pd.DataFrame:
    """Grupeaza coloana dupa pragurile date si returneaza tabelul comparativ control vs. treatment."""
    grupuri = pd.cut(df[coloana], bins=praguri, labels=etichete, include_lowest=True)
    # Agregam pe fiecare grup si metoda pentru a compara ratele si dimensiunile esantioanelor
    tabel = (
        df.assign(_grup=grupuri)
        .groupby(["_grup", "treatment"], observed=False)["harvest_success"]
        .agg(["count", "sum", "mean"])
        .unstack()
    )
    rezultat = pd.DataFrame(
        {
            "parcele_control": tabel["count"]["control"].astype(int),
            "succese_control": tabel["sum"]["control"].astype(int),
            "rata_control": tabel["mean"]["control"],
            "parcele_treatment": tabel["count"]["treatment"].astype(int),
            "succese_treatment": tabel["sum"]["treatment"].astype(int),
            "rata_treatment": tabel["mean"]["treatment"],
        }
    )
    rezultat["diferenta_treat_minus_ctrl_pp"] = (
        rezultat["rata_treatment"] - rezultat["rata_control"]
    ) * 100.0
    rezultat["control_in_avantaj"] = rezultat["rata_control"] > rezultat["rata_treatment"]
    return rezultat


def analizeaza_distance_from_porto(df: pd.DataFrame) -> dict:
    """
    Analizeaza in detaliu coloana castigatoare `distance_from_porto_km`:
    - detecteaza golul natural intre 104,0 km si 114,0 km;
    - calculeaza tabelele pentru impartirea naturala in 2 grupuri (prag 110 km)
      si pentru impartirea in 3 subregiuni viticole (70-110, 110-145, 145-205 km);
    - cuantifica dezechilibrul de alocare (variabila de confuzie).
    """
    valori = np.sort(df["distance_from_porto_km"].unique())
    diferente = np.diff(valori)
    idx_gol = int(np.argmax(diferente))
    gol_stanga = float(valori[idx_gol])
    gol_dreapta = float(valori[idx_gol + 1])
    prag_natural = (gol_stanga + gol_dreapta) / 2.0

    # 1. Impartirea naturala in 2 grupuri folosind golul natural (prag 110 km, echivalent cu 104..113)
    tabel_2_grupuri = evalueaza_impartire_intervale(
        df,
        "distance_from_porto_km",
        praguri=[0, 110, 300],
        etichete=["Aproape de Porto (70,0-104,0 km)", "Departe de Porto (114,0-204,9 km)"],
    )

    # 2. Impartirea in 3 subregiuni ale Vaii Douro (Baixo Corgo, Cima Corgo, Douro Superior)
    tabel_3_grupuri = evalueaza_impartire_intervale(
        df,
        "distance_from_porto_km",
        praguri=[0, 110, 145, 300],
        etichete=[
            "Baixo Corgo / Aproape (70-110 km)",
            "Cima Corgo / Mijloc (110-145 km)",
            "Douro Superior / Departe (145-205 km)",
        ],
    )

    # 3. Calculam distributia alocarii treatment/control intre cele doua regimuri (<=110 vs >110 km)
    zona = np.where(
        df["distance_from_porto_km"] <= 110,
        "Aproape de Porto (<=104 km)",
        "Departe de Porto (>=114 km)",
    )
    tabel_alocare_col = pd.crosstab(zona, df["treatment"], normalize="columns") * 100.0
    tabel_alocare_rand = pd.crosstab(zona, df["treatment"], normalize="index") * 100.0

    # Verificam si pista falsa soil_depth_cm (are si ea un gol bimodal intre 70 si 110 cm, dar NU inverseaza)
    tabel_decoy_soil = evalueaza_impartire_intervale(
        df,
        "soil_depth_cm",
        praguri=[0, 90, 200],
        etichete=["Sol superficial (40-70 cm)", "Sol adanc (110-160 cm)"],
    )

    return {
        "gol_stanga": gol_stanga,
        "gol_dreapta": gol_dreapta,
        "prag_natural": prag_natural,
        "tabel_2_grupuri": tabel_2_grupuri,
        "tabel_3_grupuri": tabel_3_grupuri,
        "alocare_pe_metoda_pct": tabel_alocare_col,
        "pondere_metoda_in_zona_pct": tabel_alocare_rand,
        "tabel_decoy_soil": tabel_decoy_soil,
    }


def formateaza_tabel_grupuri(tabel: pd.DataFrame) -> list:
    """Formateaza frumos un DataFrame de comparatie intre grupuri pentru raportul text."""
    linii = []
    antet = (
        f"{'Grup':<38} | {'Control (succ/n -> rata)':<26} | "
        f"{'Treatment (succ/n -> rata)':<28} | {'Diff (T-C)':>10} | {'Castigator'}"
    )
    linii.append(antet)
    linii.append("-" * len(antet))
    for nume_grup, r in tabel.iterrows():
        c_str = f"{int(r['succese_control']):>4}/{int(r['parcele_control']):<4} -> {r['rata_control']*100:6.2f}%"
        t_str = f"{int(r['succese_treatment']):>4}/{int(r['parcele_treatment']):<4} -> {r['rata_treatment']*100:6.2f}%"
        diff_pp = r["diferenta_treat_minus_ctrl_pp"]
        castigator = "CONTROL" if r["control_in_avantaj"] else "TREATMENT"
        linii.append(
            f"{str(nume_grup):<38} | {c_str:<26} | {t_str:<28} | {diff_pp:+9.2f}pp | {castigator}"
        )
    return linii


def genereaza_raport(df: pd.DataFrame) -> str:
    """Ruleaza intregul audit statistic, construieste raportul in romana si il salveaza in solution.txt."""
    globale = calculeaza_statistici_globale(df)
    cat_rez = verifica_coloane_categorice(df)
    dp_rez = [cauta_inversare_simpson_dp(df, col, min_parcele_grup=20) for col in COLOANE_NUMERICE]
    dp_minim_1 = [cauta_inversare_simpson_dp(df, col, min_parcele_grup=1) for col in COLOANE_NUMERICE]
    porto = analizeaza_distance_from_porto(df)

    linii = []
    linii.append("=" * 95)
    linii.append("PORTUGALIA - PARADOXUL DOURO | ANALIZA PARADOXULUI LUI SIMPSON")
    linii.append("=" * 95)
    linii.append(f"Total parcele experimentale: {globale['total_parcele']}")
    linii.append("")
    linii.append("1. REZULTATE PER ANSAMBLU (COMASATE)")
    linii.append("-" * 95)
    linii.append(
        f"  treatment (noua irigare): {globale['succ_treatment']:>4} / {globale['nr_treatment']} = "
        f"{globale['rata_treatment']*100:.2f}% ({globale['rata_treatment']:.6f})"
    )
    linii.append(
        f"  control   (metoda veche): {globale['succ_control']:>4} / {globale['nr_control']} = "
        f"{globale['rata_control']*100:.2f}% ({globale['rata_control']:.6f})"
    )
    linii.append(
        f"  Diferenta globala (treatment - control): {globale['diferenta_pp']:+.2f} puncte procentuale -> TREATMENT pare in avantaj!"
    )
    linii.append("")
    linii.append("2. AUDIT EXHAUSTIV PE TOATE COLOANELE CANDIDATE")
    linii.append("-" * 95)
    linii.append("2a) Coloane categorice:")
    for item in cat_rez:
        linii.append(
            f"  - {item['coloana']:<22}: inversare in {item['grupuri_inversate']}/{item['total_grupuri']} grupuri "
            f"-> {'INVERSARE COMPLETA' if item['inversare_completa'] else 'NU inverseaza complet'}"
        )

    linii.append("")
    linii.append("2b) Coloane numerice (cautare exhaustiva prin Programare Dinamica pe intervale contigue):")
    for item_20, item_1 in zip(dp_rez, dp_minim_1):
        status = (
            f"INVERSARE COMPLETA POSIBILA (pana la {item_20['max_intervale']} intervale cu >=20 parcele/brat)"
            if item_20["are_inversare_completa"]
            else f"IMPOSIBIL (max_dp[n>=20]={item_20['max_intervale']}, max_dp[n>=1]={item_1['max_intervale']})"
        )
        linii.append(
            f"  - {item_20['coloana']:<24}: corr(treat)={item_20['corelatie_treatment']:+.4f}, "
            f"corr(succes)={item_20['corelatie_succes']:+.4f} | {status}"
        )

    linii.append("")
    linii.append("  Nota asupra pistei false `soil_depth_cm` (distributie bimodala 40-70 cm vs 110-160 cm):")
    linii.extend(["  " + l for l in formateaza_tabel_grupuri(porto["tabel_decoy_soil"])])

    linii.append("")
    linii.append("3. COLOANA IDENTIFICATA: distance_from_porto_km")
    linii.append("-" * 95)
    linii.append(
        f"  Gol natural detectat in distributia distantei: intre {porto['gol_stanga']:.1f} km si "
        f"{porto['gol_dreapta']:.1f} km (prag natural de separare: {porto['prag_natural']:.1f} km)."
    )
    linii.append("")
    linii.append("3a) Impartire naturala in 2 grupuri (prag 110 km, la ruptura geografica 104-114 km):")
    linii.extend(["  " + l for l in formateaza_tabel_grupuri(porto["tabel_2_grupuri"])])

    linii.append("")
    linii.append("3b) Impartire alternativa in 3 subregiuni viticole (70-110 km, 110-145 km, 145-205 km):")
    linii.extend(["  " + l for l in formateaza_tabel_grupuri(porto["tabel_3_grupuri"])])

    linii.append("")
    linii.append("4. MECANISMUL PARADOXULUI LUI SIMPSON (DEZECHILIBRUL ALOCARII)")
    linii.append("-" * 95)
    aloc = porto["alocare_pe_metoda_pct"]
    pond = porto["pondere_metoda_in_zona_pct"]
    linii.append(
        f"  Din toate parcelele `treatment` (n={globale['nr_treatment']}): "
        f"{aloc.loc['Aproape de Porto (<=104 km)', 'treatment']:.2f}% sunt aproape de Porto vs "
        f"{aloc.loc['Departe de Porto (>=114 km)', 'treatment']:.2f}% departe de Porto."
    )
    linii.append(
        f"  Din toate parcelele `control`   (n={globale['nr_control']}): "
        f"{aloc.loc['Aproape de Porto (<=104 km)', 'control']:.2f}% sunt aproape de Porto vs "
        f"{aloc.loc['Departe de Porto (>=114 km)', 'control']:.2f}% departe de Porto."
    )
    linii.append(
        f"  Compozitia zonelor: aproape de Porto are {pond.loc['Aproape de Porto (<=104 km)', 'control']:.2f}% control; "
        f"departe de Porto are {pond.loc['Departe de Porto (>=114 km)', 'treatment']:.2f}% treatment."
    )
    linii.append("")
    linii.append("  Explicatie intr-o propozitie:")
    linii.append(
        "  Comasarea grupurilor produce inversarea deoarece 92,38% dintre parcelele `treatment` au fost "
        "amplasate departe de Porto (>=114 km), unde rata de succes de baza este ridicata (~32%), in timp "
        "ce 74,59% dintre parcelele `control` au ramas aproape de Porto (<=104 km), unde rata de succes "
        "de baza este scazuta (~25%), astfel incat media globala favorizeaza artificial `treatment` desi "
        "`control` este superior in fiecare zona."
    )
    linii.append("")
    linii.append("=" * 95)
    linii.append("FLAG FINAL: SIGMOID_DISTANCE_FROM_PORTO_KM")
    linii.append("=" * 95)

    raport = "\n".join(linii) + "\n"
    (DIRECTOR_PROIECT / "solution.txt").write_text(raport, encoding="utf-8")
    return raport


if __name__ == "__main__":
    df_date = incarca_datele()
    raport_text = genereaza_raport(df_date)
    print(raport_text)
