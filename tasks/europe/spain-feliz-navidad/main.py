# Испания – ¡Feliz Navidad!
# Алгоритм: применяем правило офиса к чекам и восстанавливаем рождественское табло.
#  - цена за единицу = Total_paid_eur / Quantity
#  - чек засчитывается, только если цена в [Fair_min, Fair_max] (границы включены)
#    и товар куплен в своём родном городе (Home_city)
#  - табло: столбец = Pilot_no (1–21), строка = List_no (1–6) товара
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AICI = Path(__file__).resolve().parent
FIS = AICI / "files"

# читаем оба CSV-файла
ghid = pd.read_csv(FIS / "office_price_guide.csv")
bonuri = pd.read_csv(FIS / "purchases.csv")

# присоединяем к каждому чеку данные из справочника (честный диапазон, номер в списке, родной город)
df = bonuri.merge(ghid, on="Product", how="left")

# цена за единицу
df["unit_price"] = df["Total_paid_eur"] / df["Quantity"]

# правило офиса: честная цена (не подделка и не ловушка для туристов)
corect = (df["unit_price"] >= df["Fair_min_eur"] - 1e-9) & (df["unit_price"] <= df["Fair_max_eur"] + 1e-9)
# опционально: ещё и куплено в родном городе (аргумент "--any-city" отключает фильтр)
if "--any-city" not in sys.argv:
    corect &= df["City"] == df["Home_city"]
bune = df[corect]
print(f"валидных чеков: {len(bune)} из {len(df)}")

# табло: матрица 6 x 21 – отметка, если пилот p привёз товар с List_no n
panou = [[" "] * 21 for _ in range(6)]
for _, r in bune.iterrows():
    panou[int(r["List_no"]) - 1][int(r["Pilot_no"]) - 1] = "#"
for rand in panou:
    print("".join(rand))

# рисуем табло: x = пилот, y = номер в списке (1 сверху)
fig, ax = plt.subplots(figsize=(12, 4))
ax.scatter(bune["Pilot_no"], bune["List_no"], s=400, marker="s", c="crimson")
ax.set_xticks(range(1, 22))
ax.set_yticks(range(1, 7), ghid.sort_values("List_no")["Product"])
ax.set_xlim(0.5, 21.5)
ax.set_ylim(6.5, 0.5)
ax.set_xlabel("Pilot_no")
ax.set_title("Рождественское табло – сувениры по честной цене")
fig.tight_layout()
fig.savefig(AICI / "board.png", dpi=120)
print("рисунок сохранён в board.png")
