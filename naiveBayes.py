#naive bayes exercise
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB,GaussianNB

data = load_wine()
print(dir(data))
print("data.data",data.data)
print("data.feature_name",data.feature_names)
print("data.target",data.target)
print("data.target_names",data.target_names)
df = pd.DataFrame(data.data,columns=data.feature_names)
df["wine_class"] = data.target
print(df.head())
input = df.drop(["wine_class"],axis="columns")
target = df.wine_class
X_train,X_test,Y_train,Y_test = train_test_split(input,target,test_size=0.2)
model = GaussianNB()
model1 = MultinomialNB()
print("for prediction x_test:\n",X_test[:80])
print("for prediction y_test:\n",Y_test[:80])

model.fit(X_train,Y_train)
print("model score of GausssianNB:",model.score(X_test,Y_test))
print("gaussian model prediction\n",model.predict(X_test[:80]))

model1.fit(X_train,Y_train)
print("model score of MultinomialNB:",model1.score(X_test,Y_test))
print("multinomial model predecitoon\n",model1.predict(X_test[:80]))