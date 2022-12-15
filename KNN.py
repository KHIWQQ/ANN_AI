import matplotlib.pyplot as plt
from sklearn.datasets._samples_generator import make_blobs
from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#load data
data = pd.read_csv('mtcars.csv')

#Bivariate analysis
print(data.corr())

plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), square=True, linewidths=0.2)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
# plt.show()

#let try to use Label Encoder first
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['Class']= le.fit_transform(data['model'])

print(data.head())
X1= data.iloc[:,1:12]
Y1= data.iloc[:,-1]
print("====== X Values ======")
print(X1)
print("====== Y Values ======")
print(Y1)

#lets try to plot Decision tree to find the feature importance
from sklearn.tree import DecisionTreeClassifier
tree= DecisionTreeClassifier(criterion='entropy', random_state=1)
tree.fit(X1, Y1)

# Feature importance ออกสอบ
imp= pd.DataFrame(index=X1.columns, data=tree.feature_importances_, columns=['Imp'])
imp_sorted = imp.sort_values(by='Imp', ascending=False)

print(imp_sorted)

sns.barplot(x=imp.index.tolist(), y=imp.values.ravel(), palette='coolwarm')
# จัดกลุ่มรถใช้ 3 ตัวแรกในการจัดกลุ่ม ที่เหลือไม่ต้องสนใจเพราะไม่ค่อยมีผล
plt.show()

#taking only two variable #disp and #qsec as these variable has high importance
X=data[['disp','qsec']]
Y=data.iloc[:,0]

#lets try to create segments using K means clustering
from sklearn.cluster import KMeans
#using elbow method to find no of clusters
wcss=[]
for i in range(1,7):
    kmeans= KMeans(n_clusters=i, init='k-means++', random_state=1)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

print(wcss)

plt.plot(range(1,7), wcss, linestyle='--', marker='o', label='WCSS value')
plt.title('WCSS value- Elbow method')
plt.xlabel('no of clusters- K value')
plt.ylabel('Wcss value')
plt.legend()
plt.show()

#Here we got no of cluster = 2
kmeans= KMeans(n_clusters=2, random_state=1)
kmeans.fit(X)

pred_Y = kmeans.predict(X)
print(pred_Y)

#Cluster Center
print(kmeans.cluster_centers_)

data['cluster']=kmeans.predict(X)
print(data.sort_values(by='cluster'))

# plotting Cluster plot
plt.scatter(data.loc[data['cluster']==0]['disp'], data.loc[data['cluster']==0]['qsec'], c='green', label='cluster1-0')
plt.scatter(data.loc[data['cluster']==1]['disp'], data.loc[data['cluster']==1]['qsec'], c='red', label='cluster2-1')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c='black', label='center')
plt.xlabel('disp')
plt.ylabel('qsec')
plt.legend()
plt.show()


#============================================================================
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i ,init="k-means++",max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title('Elbow Method')
plt.show()

kmeans = KMeans(n_clusters=4, init="k-means++", max_iter=300,n_init=10,random_state=0)
pred_y = kmeans.fit_predict(X)
print(pred_y)
print("this is x",str(X))
plt.scatter(X.iloc[:,0],X.iloc[:,1])
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c='red')
plt.show()