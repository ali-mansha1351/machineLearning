import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree
import matplotlib.pyplot as plt
df = pd.read_csv("train.csv")
#print(df.columns)

# plt.plot(df["SalePrice"],df.SalePrice)
# plt.show()


#cols_to_use =["MSSubClass","LotFrontage","LotArea","OverallQual","OverallCond","YearBuilt","FullBath","BedroomAbvGr","KitchenQual","TotRmsAbvGrd","Functional","GarageCars","PavedDrive","SaleCondition"]
#ndf = df[cols_to_use]
df.loc[:, "LotFrontage"] = df["LotFrontage"].fillna(0)
print("sum of na values in dataframe\n",df.isna().sum())
#print("untrained dataset\n",df.head())
df = pd.get_dummies(df,drop_first=True,dtype="int")
target = df["SalePrice"]
df = df.drop(["SalePrice"],axis="columns")
print("dummy variable dataset\n",df.head())

print("target dataset\n",target.head())
print("sum of na values in dataframe\n",df.isna().sum())

xtrain,xtest,ytrain,ytest = train_test_split(df,target,test_size=0.3)
model= LinearRegression()
model.fit(xtrain,ytrain)
print("moedl score with training data:",model.score(xtrain,ytrain))
print("moedl score with testing data:",model.score(xtest,ytest))





#next step is to scale the data so that max no of iterations is not reached
# scaler = MinMaxScaler()
# scaler.fit(ndf)
# newData = scaler.transform(ndf)
# scaledData = pd.DataFrame(newData,columns=ndf.columns)
#
# newTarget = scaledData["SalePrice"]
# scaledData = scaledData.drop(["SalePrice"],axis="columns")
# print("scaled train dataset\n",scaledData.head())
# print("scaled target dataset\n",newTarget.head())


#traingin the model with scaled dataset
# X_train,X_test,Y_train,Y_test = train_test_split(scaledData,newTarget,test_size=0.3,random_state=2)
# model = tree.DecisionTreeRegressor()
# model.fit(X_train,Y_train)
# print("moedl score with scaled training data:",model.score(X_train,Y_train))
# print("moedl score with scaled testing data:",model.score(X_test,Y_test))
