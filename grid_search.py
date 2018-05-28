import pandas as pd 

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

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pipe_svc = make_pipeline(
    StandardScaler(),
    SVC(random_state=1)
)

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
    scoring='accuracy',
    cv=10,
    n_jobs=-1
)

gs = gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)

# the best selected model is available via the 
# best_estimator_ attribute 
clf = gs.best_estimator_
clf.fit(X_train, y_train)
print(f"Test accuracy: {clf.score(X_test, y_test) * 100}%")

from sklearn.model_selection import cross_val_score
import numpy as np 

# Nested cross-validation
# we have an outer k-fold cross-validation loop
# to split the data into training and test folds, and 
# an inner loop is used to select the model using k-fold cross
# validation on the training fold.
# after the model selection, the test fold is then used to evaluate
# the model performance.

gs = GridSearchCV(
    estimator=pipe_svc,
    param_grid=param_grid,
    scoring='accuracy',
    cv=2
)

scores = cross_val_score(
    gs, X_train, y_train,
    scoring='accuracy', cv=5
)

print(f"CV accuracy: {np.mean(scores)} +/- {np.std(scores)}")

# The returned average cross-validation accuracy gives us a goood estimate
# of what to expect if we tune the hyperparameters of a model and use it
# on unseen data.

from sklearn.tree import DecisionTreeClassifier

gs = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=0),
    param_grid=[{
        'max_depth': [1, 2, 3, 4, 5, 6, 7, None]
    }],
    scoring='accuracy',
    cv=2
)

scores = cross_val_score(
    gs, X_train, y_train,
    scoring='accuracy', cv=5
)

print(f"CV accuracy: {np.mean(scores)} +/- {np.std(scores)}")