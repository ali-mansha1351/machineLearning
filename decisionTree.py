from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("titanic.csv")
n_df = df.drop(["PassengerId","Name","SibSp","Parch","Ticket","Cabin","Embarked"],axis="columns")
print(n_df.head())
inputs = n_df.drop(["Survived"],axis="columns")
target = n_df["Survived"]
print("target dataframe:")
print(target.head())

le_sex = LabelEncoder()
inputs["n_Sex"] = le_sex.fit_transform(inputs["Sex"])
inputs = inputs.drop(["Sex"],axis="columns")
print("inputs dataframe:")
print(inputs)

#age column has nan values we will fill them , with nan values model score is 95%
inputs["Age"] = inputs["Age"].fillna(inputs["Age"].mean(skipna=True))
# print("inputs dataframe after removing nan values:")
# print(inputs)
model = tree.DecisionTreeClassifier()
X_train,X_test,Y_train,Y_test = train_test_split(inputs,target,test_size=0.2,random_state=10)
print("X_test:\n",X_test)
print("Y_test:\n",Y_test)
model.fit(X_train,Y_train)
print("mdoel trained and its score is:",model.score(inputs,target))
print("prediction: ",model.predict([[2,19.0,13.0,1]]))



#with criterion gini ,model score was 0.9539 => without replacing nan values
#with criterion gini ,model score was 0.949494 => with replaced nan values
#with entropy model score is 0.94276 =>with replaced nan values
#with entropy model score is 0.94725 =>without replacing nan values