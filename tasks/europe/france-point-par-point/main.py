# France - Point par point: find the letters hidden in the Paris listings
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

EPS = 0.24          # DBSCAN radius in standardized units
MIN_SAMPLES = 14
MIN_LETTER = 150    # smaller clusters are ordinary dense spots of the city

df = pd.read_csv("files/paris_listings.csv")

# place + "what a traveller weighs": price, reviews, availability
cols = ["latitude", "longitude", "price", "reviews_per_month", "availability_365"]
Z = StandardScaler().fit_transform(df[cols])

# real listings are spread out (noise), the artist's dots are dense
raw = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit_predict(Z)

# keep big clusters, order them by mean price (cheapest first)
sizes = pd.Series(raw[raw >= 0]).value_counts()
letters = [c for c in sizes.index if sizes[c] >= MIN_LETTER]
letters.sort(key=lambda c: df.price[raw == c].mean())

df["label"] = -1    # -1 = the rest of the city (grey)
for i, c in enumerate(letters):
    df.loc[raw == c, "label"] = i
df[["listing_id", "label"]].to_csv("labels.csv", index=False)

# each letter is flat (2D) inside the 5D space, so PCA to 2D shows it
fig, ax = plt.subplots(1, len(letters) + 1, figsize=(5 * (len(letters) + 1), 5))
city = df.label == -1
ax[0].scatter(df.longitude[city], df.latitude[city], s=1, c="lightgrey")
ax[0].scatter(df.longitude[~city], df.latitude[~city], s=3, c=df.label[~city], cmap="tab10")
ax[0].set_title("Paris: grey = city, colour = letters")
for i in range(len(letters)):
    m = (df.label == i).values
    p = PCA(2).fit_transform(Z[m])
    ax[i + 1].scatter(p[:, 0], p[:, 1], s=6)
    ax[i + 1].set_title(f"letter {i + 1}  (n={m.sum()}, mean price={df.price[m].mean():.0f})")
    ax[i + 1].set_aspect("equal")
plt.tight_layout()
plt.savefig("letters.png", dpi=110)
print(df.label.value_counts().sort_index())
