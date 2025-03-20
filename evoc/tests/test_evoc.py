import numpy as np
from ..clustering import EVoC

clusterer = EVoC(n_neighbors=3, verbose=True)
data = np.random.rand(8, 3)
print("data", data)
cluster_labels = clusterer.fit_predict(data)
