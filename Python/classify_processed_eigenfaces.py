import csv

from sklearn import svm

from sklearn.metrics import confusion_matrix

from sklearn import cross_validation

from sklearn import datasets

from sklearn.cross_validation import KFold

import matplotlib.pyplot as plt


import numpy as np

import random

import cv2

import cv
import os

scores = []
first = True
for i in range (1,11):
	filename = 'train' + str(i) + '.csv'
	with open('ProcessedEigenfaces/'+filename , 'rb') as csvfile:
		allLandmarks = csv.reader(csvfile , delimiter=',')
		finalMatrix = []
		rowNum = 0
		for row in allLandmarks:
			finalMatrix.append([]);
			for element in range (0 , len(row)):
				if(element == len(row) - 1):
					a = int(row[element])
				else:
					a = float(row[element])
				finalMatrix[rowNum].append(a)
			rowNum = rowNum + 1

	labels = []
	for row in finalMatrix:	
		labels.append(row[len(row) - 1])

	data = []
	dataRow = 0
	for row in finalMatrix:
		data.append([])
		for element in range (0 , 21):
			data[dataRow].append(row[element])
		
		dataRow = dataRow + 1



	trainingData = data[21:]
	testingData = data[:21]

	trainingLabels = labels[21:]
	testingLabels = labels[:21]

	clf = svm.SVC(kernel='poly')

	clf.fit(trainingData, trainingLabels)
	estimatedLabels = clf.predict(testingData)

	scores.append(clf.score(testingData , testingLabels))

	if first:
		first = False
		confusion_m = confusion_matrix(testingLabels, estimatedLabels , range(1,8))
	else:
		confusion_m =np.add(confusion_m ,  confusion_matrix(testingLabels, estimatedLabels , range(1,8)))


scores = np.asarray(scores)

print scores.mean()
print confusion_m


fig = plt.figure()
#plot_confusion_matrix(confusion_m , labels)
ax = fig.add_subplot(111)
ax.set_aspect(1)

label_names = ["Angry" , "Disgust" , "Fear" , "Happy" , "Sad" , "Surprised" , "Neutral"]
label_spaces = [" " , " " , " " , " " , " " , " ", " ", " "]

res = ax.imshow(np.array(confusion_m), cmap=plt.cm.Reds, interpolation='nearest')

width = len(confusion_m)
height = len(confusion_m)

for x in xrange(width):
    for y in xrange(height):
        ax.annotate(str(confusion_m[x][y]), xy=(y, x), 
                    horizontalalignment='center',
                    verticalalignment='center')



tick_marks = np.arange(len(label_names))
plt.yticks(tick_marks, label_names)
plt.xticks(tick_marks, label_spaces)
plt.ylabel('True label')
#plt.xlabel('Predicted label')
plt.show()


clf = svm.SVC(kernel='poly')
clf.fit(data , labels)

