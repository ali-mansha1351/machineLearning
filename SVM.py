from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()
print(dir(digits))
print(digits.target[0])
#X_train,X_test,Y_train,Y_test = train_test_split(digits.data,digits.target,test_size=0.2)
#model = SVC() # score -> 97.77777
#model = SVC(C=2) #score -> 98.8888
#model = SVC(C=3) #score -> 99.7222
#model = SVC(kernel="rbf") #linear andply gives same result ,sigmoid ->score drops to 89%

plt.imshow(digits.images[0],cmap="gray")
plt.show()