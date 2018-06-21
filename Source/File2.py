# Needed librarys to complete the lab
import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pandas.read_csv("sample_stocks.csv", skiprows=0)

# creating the K means cluster
k_means = KMeans(n_clusters=3)
k_means.fit(df)

# labeling each point and then predicting closest cluster to X
print(k_means.labels_)
print("------")
k_mean = k_means.predict(df)

print(k_mean)

centers = np.array(k_means.cluster_centers_)
plt.scatter(centers[:,0], centers[:,1], marker="x", color='r')

plt.scatter(df.returns, df.dividendyield, c=k_mean)
plt.show()