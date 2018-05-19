import numpy as np 
from data_prep import get_data, plot_decisions_regions
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt 

data = get_data()
X_train, X_test, y_train, y_test = data["non-std"]
X_train_std, X_test_std = data["std"]

knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train_std, y_train)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decisions_regions(X_combined_std, y_combined, classifier=knn,
        test_idx=range(105, 150))
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc='upper left')
plt.show()
