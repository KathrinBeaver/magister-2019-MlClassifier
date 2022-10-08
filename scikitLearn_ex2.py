from sklearn.svm import SVC
import numpy as np

rawData = open("learnCoords.txt")
svmClass = SVC()
dataset = np.loadtxt(rawData, delimiter=",")

svmClass.fit(dataset[:, :-1], dataset[:, -1])

testValues = np.array([[-200, -2]], float)
svmPredict = svmClass.predict(testValues)
print(svmPredict)

