import pandas as pd 
import numpy as np 

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
                header=None)

from sklearn.preprocessing import LabelEncoder

X = df.loc[:, 2:].values
y = df.loc[:, 1].values

le = LabelEncoder()
y = le.fit_transform(y)

# le.classes_ will output B and M 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.20,
                                                    stratify=y,
                                                    random_state=1)

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pipe_svc = make_pipeline(
    StandardScaler(),
    SVC(random_state=1)
)

pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)

confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)

print(confmat)

import matplotlib.pyplot as plt 

fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j],
            va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')
plt.show()

from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score, f1_score

# precision score = tp / (tp + fp)
# recall scores = tp / p = (tp / (fn + tp))
# f1 = 2 * (pre * rec) / (pre + rec)

print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))

from sklearn.metrics import make_scorer

scorer = make_scorer(f1_score, pos_label=0)

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{
    'svc__C': param_range,
    'svc__kernel': ['linear']
},
{
    'svc__C': param_range,
    'svc__gamma': param_range,
    'svc__kernel': ['rbf']
}]

gs = GridSearchCV(
    estimator=pipe_svc,
    param_grid=param_grid,
    scoring=scorer,
    cv=10
).fit(X_train, y_train)

print(gs.best_score_)
print(gs.best_params_)

from sklearn.metrics import roc_curve, auc 
from scipy import interp 
from sklearn.decomposition import PCA 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

pipe_lr = make_pipeline(
    StandardScaler(),
    PCA(n_components=2),
    LogisticRegression(penalty='l2', random_state=1, C=100.0)
)

X_train2 = X_train[:, [4, 14]]
cv = list(StratifiedKFold(n_splits=3, random_state=1).split(X_train, y_train))

fig = plt.figure(figsize=(7, 5))
mean_tpr = 0.0 
mean_fpr = np.linspace(0, 1, 100)

all_tpr = []

for i, (train, test) in enumerate(cv):
    probas = pipe_lr.fit(X_train2[train], y_train[train]).predict_proba(X_train2[test])
    fpr, tpr, thresholds = roc_curve(y_train[test], probas[:, 1], pos_label=1)

mean_tpr += interp(mean_fpr, fpr, tpr)
mean_tpr[0] = 0.0 
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label='ROC fold %d (area=%0.2f)' % (i+1, roc_auc))
plt.plot([0, 1], [0, 1], linestyle='--', color=(0.6, 0.6, 0.6), label='random guessing')

mean_tpr /= len(cv)
mean_tpr[-1] = 1.0 
mean_auc = auc(mean_fpr, mean_tpr)

plt.plot(mean_fpr, mean_tpr, 'k--', label='mean ROC (area = %0.2f)' % mean_auc, lw=2)
plt.plot([0, 0, 1], [0, 1, 1], linestyle=':', color='black', label='perfect performance')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('false positive rate')
plt.ylabel('true positve rate')
plt.legend(loc='lower right')
plt.show()