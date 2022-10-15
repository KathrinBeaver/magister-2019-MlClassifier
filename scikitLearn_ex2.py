from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np

rawData = open("learnCoords.txt")
svmClass = SVC()
dataset = np.loadtxt(rawData, delimiter=",")

svmClass.fit(dataset[:, :-1], dataset[:, -1])

testValues = np.array([[2000, 100]], float)
svmPredict = svmClass.predict(testValues)
print(svmPredict)

model = LogisticRegression()
model.fit(dataset[:, :-1], dataset[:, -1])

regressPredict = model.predict(testValues)
print(regressPredict)