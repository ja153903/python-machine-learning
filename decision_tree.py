import matplotlib.pyplot as plt 
import numpy as np 

def gini(p):
    return p * (1 - p) + (1 - p) * (1 - (1 - p))

def entropy(p):
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

def error(p):
    return 1 - np.max([p, 1 - p])

x = np.arange(0.0, 1.0, 0.01)

ent = [entropy(p) if p != 0 else None for p in x]

sc_ent = [e*0.5 if e else None for e in ent]

err = [error(i) for i in x]

fig = plt.figure()
ax = plt.subplot(111)

impurites = [ent, sc_ent, gini(x), err]
labels = ['Entropy', 'Entropy (Scaled)', 'Gini Impurity', 'Misclassification Error']
linestyle = ['-', '-', '--', '-.']
colors =  ['black', 'lightgray', 'red', 'green', 'cyan']

for i, lab, ls, c, in zip(impurites, labels, linestyle, colors):
    line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)
    
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
        ncol=5, fancybox=True, shadow=False)
    
ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')

plt.ylim([0, 1.1])
plt.xlabel('p(i=1)')
plt.ylabel('Impurity Index')
plt.show()

from data_prep import get_data, plot_decisions_regions

data_obj = get_data()

X_train, X_test, y_train, y_test = data_obj["non-std"]
X_train_std, X_test_std = data_obj["std"]

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
tree.fit(X_train, y_train)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

plot_decisions_regions(X_combined, y_combined, classifier=tree,
                        test_idx=range(105, 150))

plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.show()                

from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
dot_data = export_graphviz(tree, filled=True, rounded=True,
                            class_names=['Setosa', 'Versi', 'Virginica'],
                            feature_names=['petal length', 'petal width'],
                            out_file=None)
graph = graph_from_dot_data(dot_data)
graph.write_png('tree.png')

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(criterion='gini', n_estimators=25, 
        random_state=1, n_jobs=2)

forest.fit(X_train, y_train)

plot_decisions_regions(X_combined, y_combined, classifier=forest, test_idx=range(105, 150))
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc='upper left')
plt.show()