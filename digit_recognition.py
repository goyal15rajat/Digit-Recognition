import tkinter
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import datasets

digits = datasets.load_digits()

clf = svm.SVC(gamma=0.001,C=100)

x,y = digits.data[:-10],digits.target[:-10]


d = clf.fit(x,y)
print(d)

print('Prediction',clf.predict(digits.data[-10]))

plt.imshow(digits.images[-10] , cmap = plt.cm.gray_r , interpolation = 'nearest')
plt.show()