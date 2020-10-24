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
dataset = np.loadtxt(rawData, delimiter=",")

print(dataset)
# print(dataset[:, :-1])
# print(dataset[:, -1])

# 1,0,0,1
# 1,0,0 - вектор признаков, 1 - результат, второй параметр функции обучения fit
svmClass.fit(dataset[:, :-1], dataset[:, -1])

testValues = np.array([[0, 0, 1]], float)
svmPredict = svmClass.predict(testValues)
print(svmPredict)

# Save - load Models
# save the model to disk
# filename = 'model.dat'
# pickle.dump(model, open(filename, 'wb'))
#
# ...
#
# load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, Y_test)
# print(result)