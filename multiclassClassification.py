import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

digits = load_digits()
print(dir(digits))

print("data at index 0:",digits.data[0])
print("image at index 0:",digits.images[0])
print("target at index 0:",digits.target[0])

#for i in range(5):
   # plt.matshow(digits.images[i])
   # plt.show()

X_train,X_test,Y_train,Y_test = train_test_split(digits.data,digits.target,test_size=0.2)
model = LogisticRegression(max_iter=3000)
print("model instance is created")
model.fit(X_train,Y_train)
print("model is trained and the score is:",model.score(X_train,Y_train))

print("digits at index 170 is:",digits.target[170])
print("model prediction of the index is:",model.predict([digits.data[170]]))

print("confusion matrix")
y_predicted = model.predict(X_test)
cm = confusion_matrix(Y_test,y_predicted)
sns.heatmap(cm,annot=True)
plt.xlabel("predcited")
plt.ylabel("truth")
plt.show()