from sklearn.svm import SVC
import numpy as np

rawData = open("learnCoords.txt")
svmClass = SVC()
dataset = np.loadtxt(rawData, delimiter=",")

# print(dataset)

svmClass.fit(dataset[:, :-1], dataset[:, -1])

testValues = np.array([[-0.0001, -50]], float)

svmPredict = svmClass.predict(testValues)
print(svmPredict)
