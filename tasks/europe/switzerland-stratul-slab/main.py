# Elvetia - "Stratul slab": mesajul e ascuns in structura unui arbore de decizie.
# Antrenam un DecisionTreeClassifier cu setarile implicite si citim prima litera
# a coloanei folosite la fiecare nod de decizie, in ordinea nodurilor.
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Citim profilurile de zapada din directorul curent
profiles = pd.read_csv("test-input.csv")
# Tinta e "slab_released"; daca lipseste, folosim ultima coloana ca rezerva
target = "slab_released" if "slab_released" in profiles.columns else profiles.columns[-1]
# X = toate celelalte coloane (trasaturile), y = tinta
features = profiles.drop(columns=[target])
labels = profiles[target]

# Arbore cu parametri impliciti: creste pana cand frunzele sunt pure.
# random_state conteaza doar la egalitati intre split-uri.
tree_model = DecisionTreeClassifier(random_state=0).fit(features, labels)
# Structura interna a arborelui (vectori indexati dupa numarul nodului)
tree_struct = tree_model.tree_
column_names = list(features.columns)

# Nodurile sunt numerotate in pre-ordine (radacina, subarborele stang, apoi dreptul),
# deci indicii 0, 1, 2, ... sunt deja ordinea de citire.
# Un nod e frunza daca children_left == children_right (ambele -1) -> il sarim.
# Pentru nodurile interne, tree_struct.feature[i] = indexul coloanei folosite la split;
# luam prima litera a numelui coloanei, cu majuscula.
hidden_word = "".join(
    column_names[tree_struct.feature[node_id]][0].upper()
    for node_id in range(tree_struct.node_count)
    if tree_struct.children_left[node_id] != tree_struct.children_right[node_id]
)

# Scriem doar flag-ul
with open("test-output.txt", "w") as out_file:
    out_file.write(f"SIGMOID_{hidden_word}\n")
