import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("test-input.csv")
target = "slab_released" if "slab_released" in df.columns else df.columns[-1]
X = df.drop(columns=[target])
y = df[target]

clf = DecisionTreeClassifier(random_state=0).fit(X, y)
tree = clf.tree_
names = list(X.columns)

# sklearn stores nodes in depth-first pre-order: read the split questions in that order
word = "".join(
    names[tree.feature[i]][0].upper()
    for i in range(tree.node_count)
    if tree.children_left[i] != tree.children_right[i]
)

with open("test-output.txt", "w") as f:
    f.write(f"SIGMOID_{word}\n")
