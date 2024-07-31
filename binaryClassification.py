import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("HR_comma_sep.csv")
print(df.head())

# doing exploratory analysis
# make a copy of dataset wihtout the columns that have string values

copydf = df.copy()
copydf.drop(['Department','salary'],axis="columns",inplace=True)
print(copydf.head())

# groupby the reamining columns by left column and take mean
with pd.option_context('display.max_columns', None):print(copydf.groupby('left').mean())

#from above we conclude that satisfaction_level, average_montly_hours and promotion_last_5years have impact on retention of employees
#now we will visualize the effect of department and salary on retention of employees

pd.crosstab(df.salary,df.left).plot(kind="bar")
#plt.show()

# high salary emloyees have low leaving rate
# creating a ddatframe containing only affecting factors

sub_df = df[["satisfaction_level","average_montly_hours","promotion_last_5years","salary"]]
#print(sub_df.head())

# create dummy varaibles for salary
dummy = pd.get_dummies(sub_df.salary,dtype="int")
#print(dummy.head())

final = pd.concat([sub_df,dummy],axis="columns")
#with pd.option_context('display.max_columns', None):print(final.head())

# droping extra medium column and salary column
final.drop(['salary','medium'],axis='columns',inplace=True)
with pd.option_context('display.max_columns', None):print(final.head())

# splitting datset
X_train,X_test,Y_train,Y_test = train_test_split(final,df.left,train_size=0.8)

# training model
model = LogisticRegression(max_iter=1000)
model.fit(X_train,Y_train)
print("model's score is:",model.score(X_train,Y_train))
print("the Y_testing data is:",Y_test)
print("models testing is:",model.predict(X_test))
