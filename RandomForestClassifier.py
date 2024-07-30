from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

dt = load_iris()
print(dir(dt))
X_train,X_test,Y_train,Y_test = train_test_split(dt.data,dt.target,test_size=0.2)
print("X train length:",len(X_train))
print("X test length:",len(X_test))

model = RandomForestClassifier() # n_estimaters=10,score=96%
#model = RandomForestClassifier(n_estimators=20) # n_estimaters=20 ,score=1
model.fit(X_train,Y_train)
print("model is trained and the score is:",model.score(X_test,Y_test))
y_predicted = model.predict(X_test)
cm = confusion_matrix(Y_test,y_predicted)
sns.heatmap(cm,annot=True)
plt.xlabel("prdicted")
plt.ylabel("truth")
plt.show()