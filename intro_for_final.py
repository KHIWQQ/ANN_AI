#_โมดูลที่ต้องใช้
import  pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import linear_model
from sklearn import metrics
import numpy as np
from math import log
import seaborn as sns

#_/////////////////////////////////////////////////////////////////

#_เรียกดาต้าที่ต้องการใช้
data = pd.read_csv('mtcars.csv')
#_ทดสอบ
print("////////////////////////////////////type////////////////////////////////////")
print(type(data))
print("////////////////////////////////////head////////////////////////////////////")
print(data.head())
print("////////////////////////////////////info////////////////////////////////////")
print(data.info())
print("////////////////////////////////////describe////////////////////////////////////")
print(data.describe())
print("////////////////////////////////////isnull////////////////////////////////////")
print(data.isnull().sum())
print("////////////////////////////////////key////////////////////////////////////")
print(data.keys())

#_/////////////////////////////////////////////////////////////////

#_ทดลองพลอต กราฟจะมีอยู่3แบบ
#_scatter
data.plot(
    kind="scatter",
    x="wt",
    y="mpg",
    figsize=(9,9),
    color="black",
    title="This is fucking scatter plot")
plt.show()
#_box
#_1.กำหนดให้เป็นบล็อก2x2
fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2 ,ncols=2)
#_2.เติมข้อมูลในแต่ละbox
ax1.set_title("fuck mpg")
ax1.boxplot(data['mpg'], labels=['mpg'])
ax2.set_title("fuck mpg")
ax2.boxplot(data['wt'], labels=['fuck'])
ax3.set_title("fuck mpg")
ax3.boxplot(data['hp'], labels=['you'])
ax4.set_title("fuck mpg")
ax4.boxplot(data['qsec'], labels=['bitch'])
#_ออกแบบlayout
plt.tight_layout()
plt.show()
#_กราฟความหนาแน่น
plt1 = pd.DataFrame(data['mpg'])
plt1.plot(
    kind = "density",
    title = "Just plot it bastard"
)
plt2 = pd.DataFrame(data['wt'])
plt2.plot(
    kind = "density",
    title = "Just plot another"
)
plt.show()
#_กราฟ metrix
corrMatrix = data.corr()
print(corrMatrix)
sn.heatmap(corrMatrix, annot=True)
plt.show()

#_/////////////////////////////////////////////////////////////////

#_การใช้กราฟการถดถอยเชิงเส้นมาทำ machine learning
#_simple linear
#_1.เรียกใช้model
regression_model = linear_model.LinearRegression()
#_2.ใส่เนื้อหาให้model
#_ด้วยเหตุผลเหี้ยไรไม่ทราบ ให้ X เป็นพิมพ์ใหญ่นะ
regression_model.fit(
    X = pd.DataFrame(data["wt"]),
    y = data["mpg"]
)
#_3.เช็คโมเดล
print(regression_model.intercept_)
print(regression_model.coef_)
#_4.ดูสกอร์
score = regression_model.score(
    X = pd.DataFrame(data["wt"]),
    y = data["mpg"]
)
print("R-squared fucking value : ")
print(score)
#_การทำนายค่า
print("predic dic dic dictionnnnn")
WT1 = pd.DataFrame([3.52]) #ค่าที่จะอ้างอิง ครั้งนี้สุ่มเอาก่อน
predicted_mpg = regression_model.predict(WT1)
print(predicted_mpg)
print("Predicted mpg of wt = {0} is fucking {1}".format(WT1[0][0], predicted_mpg[0]))

#_จากนี้จะทำแบบนี้ทุกอันเพื่อพล็อตแม่งเป็นกราฟละ
train_prediction = regression_model.predict(X= pd.DataFrame(data["wt"]))
residuals = data["mpg"] - train_prediction
print(residuals.describe())

# อันปกติคือพล็อตจุด
data.plot(
    kind = "scatter",
    x= "wt",
    y= "mpg",
    figsize = (9,9),
    color = "green",
    xlim = (0,7), #กราฟแกน x ให้โชว์แค่ 0-7
    title = "FUCKKKKKKKKK"
)
# คราวนี้พล็อตเส้นจากการทำนาย
plt.plot(
    data["wt"],
    train_prediction,
    color = "blue"
)
# และก็....
plt.show()

# ต่อไปก็เป็นค่าทางคณิตศษสตร์ของโมเดล
print("R squared value : ")
print(score)
y_true = data["mpg"]
y_predict = train_prediction
print("MAE : ")
print(metrics.mean_absolute_error(y_true, y_predict))
print("MSE : ")
print(metrics.mean_squared_error(y_true, y_predict))
print("RMSE: ")
print(np.sqrt(metrics.mean_squared_error(y_true, y_predict)))

num_params = len(regression_model.coef_) + 1
n = len(y_true)
mse = metrics.mean_squared_error(y_true, y_predict)
print("AIC : ")
aic = n*log(mse) + 2*num_params
print(aic)

print("BIC : ")
bic = n*log(mse) + num_params*log(n)
print(bic)

#_/////////////////////////////////////////////////////////////////

#_ จากนี้มาทำ linear หลายตัวรวมกันให้เป็น multiple linear regression
print("Fucking multiple linear!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print("Fucking multiple linear!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print("Fucking multiple linear!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#_ เรียกใช้ขั้นต้น
multi_reg_model = linear_model.LinearRegression()
#_ เทรนโดยใช้ค่า X หลายตัวคราวนี้
multi_reg_model.fit(
    X = data.loc[:,["wt","hp"]],
    y = data["mpg"]
)
#_ ลองเช็คการเทรน
print("Y-intersept and slope : ")
print(multi_reg_model.intercept_)
print(multi_reg_model.coef_)
#_ ได้เวลาดูสกอร์
print("R squred value : ")
score = multi_reg_model.score(
    X = data.loc[:,["wt","hp"]],
    y = data["mpg"]
)
print(score)

#ทดสอบ
print("test")
#_ เรียกใช้ขั้นต้น
multi_reg_model = linear_model.LinearRegression()
#_ เทรนโดยใช้ค่า X หลายตัวคราวนี้
multi_reg_model.fit(
    X = data.loc[:,["wt","qsec","hp"]],
    y = data["mpg"]
)
#_ ลองเช็คการเทรน
print("Y-intersept and slope : ")
print(multi_reg_model.intercept_)
print(multi_reg_model.coef_)
#_ ได้เวลาดูสกอร์
print("R squred value : ")
score = multi_reg_model.score(
    X = data.loc[:,["wt","qsec","hp"]],
    y = data["mpg"]
)
print(score)

#_ ทีนี้เตรียมทำนายค่า
X2 = pd.DataFrame({'wt':[5.08],'qsec':[7.87],'hp':[74.1]})
multi_reg_model_2 = multi_reg_model.predict(X2)
print("fucking predicted value is fucking : ")
print(multi_reg_model_2)

#จบบทแรก ////////////////////     ////////////////////     ////////////////////     ////////////////////
#จบบทแรก ////////////////////     ////////////////////     ////////////////////     ////////////////////
#จบบทแรก ////////////////////     ////////////////////     ////////////////////     ////////////////////
#จบบทแรก ////////////////////     ////////////////////     ////////////////////     ////////////////////


# K-MEAN clustering

#import plt มาเรียบร้อย
from sklearn.datasets._samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

#let try to use Label Encoder first
#_ การฟิตตรงนี้จะทำให้ข้อมูลแสดงได้ทั้งหมด ถ้าตัดไปจะมีตัวแปรเหลือรอดมาไม่กี่ตัว
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['Class']= le.fit_transform(data['model'])

#_ กำหนด X Y ซะ
X1= data.iloc[:,1:12]
Y1= data.iloc[:,-1]
print("====== X Values ======")
print(X1)
print("====== Y Values ======")
print(Y1)

#lets try to plot Decision tree to find the feature importance
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='entropy', random_state=1)
tree.fit(X1, Y1)

# Feature importance ออกสอบ
imp = pd.DataFrame(index=X1.columns, data=tree.feature_importances_, columns=['Imp'])#_ ทำให้มีแท่ง11แท่งเท่ากับจำนวนตัวแปรใน x1
imp_sorted = imp.sort_values(by='Imp', ascending=False)
print(imp_sorted)
sns.barplot(x=imp.index.tolist(), y=imp.values.ravel(), palette='coolwarm')#_ ดูดีๆโค้ดนี้เขียนว่า barplot
plt.show()# จัดกลุ่มรถใช้ 3 ตัวแรกในการจัดกลุ่ม ที่เหลือไม่ต้องสนใจเพราะไม่ค่อยมีผล

#_/////////////////////////////////////////////////////////////////

# ให้ X มี2ค่าพอ
X=data[['disp','qsec']]
Y=data.iloc[:,0]

# ได้เวลาทำกราฟข้อศอกหมา
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,7):
    kmeans= KMeans(n_clusters=i, init='k-means++', random_state=1)#_ nclusterบอกว่ามีค่า7ครั้ง จะเห็นว่ากราฟหักมุม7รอบ
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
print(wcss)
plt.plot(range(1,7), wcss, linestyle='--', marker='o', label='WCSS value')
plt.title('WCSS value- Elbow method')
plt.xlabel('no of clusters- K value')
plt.ylabel('Wcss value')
plt.legend()
plt.show()

#_/////////////////////////////////////////////////////////////////

#Here we got no of cluster = 2
kmeans= KMeans(n_clusters=2, random_state=1) #_ คราวนี้nclusterบอกว่ามีคลัสเตอร์ 2 กลุ่ม
kmeans.fit(X)

pred_Y = kmeans.predict(X)
print(pred_Y)

# ต้องมีคลัสเตอร์เซ็นเตอร์
print(kmeans.cluster_centers_)

data['cluster'] = kmeans.predict(X) #_ เพิ่มคอลัมน์ใหม่เข้าไปในดาต้าเซ็ทเลย
print(data.sort_values(by='cluster')) #_จะเห็นว่าคอลัมน์ cluster มีค่าเป็น 0 หรือ 1 ซึ่งทั้ง2พวกนี้จะเกาะกลุ่มกันเองในแผนภูมิ

# การพล็อตแผนที่คลัสเตอร์
plt.scatter(data.loc[data['cluster']==0]['disp'], data.loc[data['cluster']==0]['qsec'], c='green', label='cluster1-0')
plt.scatter(data.loc[data['cluster']==1]['disp'], data.loc[data['cluster']==1]['qsec'], c='red', label='cluster2-1')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c='black', label='center')
plt.xlabel('disp')
plt.ylabel('qsec')
plt.legend()
plt.show()

#จบบทสอง ////////////////////     ////////////////////     ////////////////////     ////////////////////
#จบบทสอง ////////////////////     ////////////////////     ////////////////////     ////////////////////
#จบบทสอง ////////////////////     ////////////////////     ////////////////////     ////////////////////
#จบบทสอง ////////////////////     ////////////////////     ////////////////////     ////////////////////

#_ จากนี้จะลองทำโจทย์แบบดอกไอริสกัน

#_ ขั้นแรกอิมพอร์ตดาต้ามาก่อน
data2 = pd.read_csv('iris.csv')
#_ ดูขนาดดาต้า
print(data2.shape)
#_ ลองดูซัก20บรรทัดแรก
print(data2.head(20))
#_ describeน่าจะทำให้ทุกข้อมูลอยู่ในหน่วยเดียวกัน (เซนติเมตร)
print(data2.describe())
#_ จัดกลุ่มตามคอลัมน์variety ละก็แสดงจำนวนด้วย
print(data2.groupby('variety').size())

#_ จากนี้จะเป็น data visualization คือการนำเสนอข้อมูล
#_ แบบ box ก่อน
data2.plot(
    kind = "box",
    subplots = True,
    layout = (2,2),
    sharex = False,
    sharey = False
)
from matplotlib import pyplot
pyplot.show()

#_ มาต่อกันที่ฮิสโทรแกรม
data2.hist()
pyplot.show()     #_ แค่เนี้ย.....

#_ scatter ต่อ
from  pandas.plotting import scatter_matrix
scatter_matrix(data2)
pyplot.show()

#_ หลังจากนี้จะแบ่งข้อมูลเป็ส่วนเทรน80% อีก20%จะไว้ตรวจสอบ
from sklearn.model_selection import train_test_split
array = data2.values
X = array[:,0:4]
y = array[:,4]
X_train,X_validation,Y_train,Y_validation = train_test_split(X, y, test_size=0.2)
print("split success!!")

#_ ตอนนี้คงต้อง import พวกที่เหลือมาก่อนละ
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
#_ จากนั้นทำโมเดลในการวิเคราะห์มา 6 แบบ แล้วลองรันเพื่อดูผลกัน
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

#_ ลองรันแต่ละโมเดล
result = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_result = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    result.append(cv_result)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_result.mean(), cv_result.std()))


    #_ แล้วก็โชว์ผลของทั้ง 6 โมเดลซะก่อน
    pyplot.boxplot(result, labels=names)
    pyplot.title('Algorithm Comparison')
    pyplot.show()
    #_จากการรันเห็นว่า LDA เจ๋งสุด อยู่ที่ 98%
    #_จากนี้จะเริ่มการทำนาย(Prediction)โดยมีโมเดล LDA เป็นตัวดำเนินการ แต่เนื่องจากจารย์สอนแค่วิธีของ SVM เพราะงั้นก็ใช้ SVM แม่งนี่ล่ะ
    model = SVC(gamma='auto')
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)#นี่คือการให้มันลองเดาจากพวกค่าที่สงวนเอาไว้แต่แรก

    #_ จากนี้ก็แสดงผลการประเมิน
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))

#_ เสร็จแล้วโว้ยยยยยยยยยยยยยยยยยยยยยยยยยยยยยยยยยยยยย
print("FUCKING FINISHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")