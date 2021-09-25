import pickle

from sklearn.svm import SVC
import numpy as np

#  Другие методы МL, еще больше - в библиотеке.
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC

# Открыть файл с вектором признаков и преобразовать в датасет

rawData = open("learn.txt")
dataset = np.loadtxt(rawData, delimiter=",")

# Обучить модель для выбранного алгоритма ML
svmClassifier = SVC()
svmClassifier.fit(dataset[:, :-1], dataset[:, -1])

# Сохранить обученную модель
filename = 'model.dat'
pickle.dump(svmClassifier, open(filename, 'wb'))

# Загрузить ранее обученную модель
loadedSvmClassifier = pickle.load(open(filename, 'rb'))

# Классифицировать входные данные
# testValues = np.array([[2,1,1,1,0,2,4,1]], float) # единонорог
testValues = np.array([[3,3,0,1,0,2,2,0]], float) # стервятник
svmPredict = loadedSvmClassifier.predict(testValues)

print(svmPredict)

# print(dataset)
# print(dataset[:, :-1])
# print(dataset[:, -1])
#
# # 1,0,0,1
# # 1,0,0 - вектор признаков, 1 - результат, второй параметр функции обучения fit
# svmClassifier.fit(dataset[:, :-1], dataset[:, -1])
#
# testValues = np.array([[0, 0, 1]], float)
# svmPredict = svmClassifier.predict(testValues)
# print(svmPredict)
#
# # Save - load Models
# # save the model to disk
# filename = 'model.dat'
# pickle.dump(svmPredict, open(filename, 'wb'))
# #
# ...
#
# load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))


# result = loaded_model.score(X_test, Y_test)
# print(result)