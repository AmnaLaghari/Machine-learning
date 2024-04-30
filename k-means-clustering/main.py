# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
X= -2 * np.random.rand(100,2)
X1 = 1 + 2 * np.random.rand(50,2)
X[50:100, :] = X1
plt.scatter(X[ : , 0], X[ : , 1], s =50, c='b')
plt.show()
Kmean = KMeans(n_clusters=2)
Kmean.fit(X)
centers=Kmean.cluster_centers_
plt.scatter(X[ : , 0], X[ : , 1], s =50, c='b')
plt.scatter(centers[0][0], centers[0][1], s=200, c='g', marker='s')
plt.scatter(centers[1][0], centers[1][1], s=200, c='r', marker='s')
plt.show()
print(Kmean.labels_)

sample_test=np.array([-3.0,-3.0])
second_test=sample_test.reshape(1, -1)
print(Kmean.predict(second_test))