import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

data = pd.read_csv('mtcars.csv')
X=data[['mpg','wt']]

wcss=[]
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=1)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss, linestyle='--', marker='o', label='WCSS value')
plt.title('WCSS value- Elbow method')
plt.xlabel('no of clusters- K value')
plt.ylabel('Wcss value')
plt.legend()
plt.show()

kmeans= KMeans(n_clusters=3, random_state=1)
kmeans.fit(X)
pred_Y = kmeans.predict(X)

data['cluster']=kmeans.predict(X)

plt.scatter(data.loc[data['cluster']==0]['mpg'], data.loc[data['cluster']==0]['wt'], marker="*", label='cluster1-0', color='green')
plt.scatter(data.loc[data['cluster']==1]['mpg'], data.loc[data['cluster']==1]['wt'], marker="x", label='cluster2-1', color='red')
plt.scatter(data.loc[data['cluster']==2]['mpg'], data.loc[data['cluster']==2]['wt'], marker="+", label='cluster3-2', color='blue')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c='black', label='center')
plt.xlabel('mpg')
plt.ylabel('wt')
plt.legend()
plt.show()