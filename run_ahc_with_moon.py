import pandas

import matplotlib.pyplot as pyplot

import scipy.cluster.hierarchy as shc 

from sklearn.cluster import AgglomerativeClustering

dataset = pandas.read_csv("dataset_moon.csv")

print(dataset)


pyplot.scatter(dataset['x1'], dataset['x2'])
pyplot.savefig("scatterplot_moon.png")
pyplot.close()


pyplot.title("Dendrogram")
dendrogram_object = shc.dendrogram(shc.linkage(dataset, method="ward"))
pyplot.savefig("dendrogram_moon.png")
pyplot.close()

machine = AgglomerativeClustering(n_clusters=2, affinity="euclidean", linkage="ward")
results_ahc = machine.fit_predict(dataset)

pyplot.scatter(dataset['x1'], dataset['x2'], c=results_ahc)
pyplot.savefig("scatterplot_moon_color.png")
pyplot.close()













