# Elvetia - "Stratul slab": mesajul e ascuns in structura unui arbore de decizie.
# Antrenam un DecisionTreeClassifier cu setarile implicite si citim prima litera
# a coloanei folosite la fiecare nod de decizie, in ordinea nodurilor.
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Citim profilurile de zapada din directorul curent
df = pd.read_csv("test-input.csv")
# Tinta e "slab_released"; daca lipseste, folosim ultima coloana ca rezerva
target = "slab_released" if "slab_released" in df.columns else df.columns[-1]
# X = toate celelalte coloane (trasaturile), y = tinta
X = df.drop(columns=[target])
y = df[target]

# Arbore cu parametri impliciti: creste pana cand frunzele sunt pure.
# random_state conteaza doar la egalitati intre split-uri.
clf = DecisionTreeClassifier(random_state=0).fit(X, y)
# Structura interna a arborelui (vectori indexati dupa numarul nodului)
tree = clf.tree_
names = list(X.columns)

# Nodurile sunt numerotate in pre-ordine (radacina, subarborele stang, apoi dreptul),
# deci indicii 0, 1, 2, ... sunt deja ordinea de citire.
# Un nod e frunza daca children_left == children_right (ambele -1) -> il sarim.
# Pentru nodurile interne, tree.feature[i] = indexul coloanei folosite la split;
# luam prima litera a numelui coloanei, cu majuscula.
word = "".join(
    names[tree.feature[i]][0].upper()
    for i in range(tree.node_count)
    if tree.children_left[i] != tree.children_right[i]
)

# Scriem doar flag-ul
with open("test-output.txt", "w") as f:
    f.write(f"SIGMOID_{word}\n")
