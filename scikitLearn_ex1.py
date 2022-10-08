import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np
#  Другие методы МL, еще больше - в библиотеке.
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC

rawData = open("learn.txt")
svmClass = SVC()
model = LogisticRegression()
dataset = np.loadtxt(rawData, delimiter=",")

print(dataset)
# print(dataset[:, :-1])
# print(dataset[:, -1])

# 1,0,0,1
# 1,0,0 - вектор признаков, 1 - результат, второй параметр функции обучения fit
svmClass.fit(dataset[:, :-1], dataset[:, -1])
model.fit(dataset[:, :-1], dataset[:, -1])

testValuses = np.array([[0, 1, 0]], float)
svmPredict = svmClass.predict(testValuses)
print(svmPredict)
regressionPredict = model.predict(testValuses)
print(regressionPredict)

# Save - load Models
# save the model to disk
filename = 'model.dat'
pickle.dump(svmPredict, open(filename, 'wb'))
#
# ...
#
# load the model from disk
# model = pickle.load(open(filename, 'rb'))
result = model.score(dataset[:, :-1], dataset[:, -1])
print(result)