import numpy as np
np.seterr(divide='ignore',invalid='ignore')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets.samples_generator import make_blobs
from sklearn.decomposition import PCA

X, y = make_blobs(n_samples=1000, n_features=3, centers=[[3,3, 3], [1,1,1], [2,2,2],[0,0,0] ], cluster_std=[0.2, 0.5, 0.2, 0.3],
                  random_state=8)
fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=35, azim=25)
plt.scatter(X[:, 0], X[:, 1], X[:, 2])
plt.show()

pca = PCA(n_components=2)
pca.fit(X)
print(pca.explained_variance_ratio_)

X_new = pca.transform(X)
plt.scatter(X_new[:, 0], X_new[:, 1])
plt.show()

pca = PCA(n_components=0.90)
pca.fit(X)
print(pca.explained_variance_ratio_)# 输出为[0.94883173]
print(pca.n_components_)# 输出为1
X_new = pca.transform(X)
