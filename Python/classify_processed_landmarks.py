import csv


from sklearn import svm

from sklearn.metrics import confusion_matrix

from sklearn import cross_validation

from sklearn import datasets

from sklearn.cross_validation import KFold

import matplotlib.pyplot as plt


import numpy as np

import cv2

import cv
import os

import random


derive = np.gradient

def eye_curves(X, Y):
	top1 = np.mean(derive(derive(Y[36:40])))
	top2 = np.mean(derive(derive(Y[42:46])))
	bottom1 = np.mean(derive(derive(Y[45:48]+[Y[42]])))
	bottom2 = np.mean(derive(derive(Y[39:42]+[Y[36]])))
	return top1,top2,bottom1,bottom2


def eyebrow(Y):
	d1 = derive(Y)
	d2 = derive(d1)
	return [np.mean(d1),np.mean(d2)]

def brow_heights(X,Y):
	leye_h = np.mean([Y[36],Y[39]])
	reye_h = np.mean([Y[42],Y[45]])
	return abs(max(Y[17:22]) - leye_h)/abs(Y[36]-Y[39]),abs(max(Y[22:27]) - reye_h)/abs(Y[42]-Y[45])

def process_points(vector):
	Xs = vector[0:66]
	Ys = vector[66:]
	mouth_ratio = abs(Ys[61]-Ys[64])/abs(Xs[54] - Xs[48])
	
	bot_m_c = np.mean(derive(derive(Ys[54:60]+Ys[48:49])))
	top_m_c = np.mean(derive(derive(Ys[48:55])))
	eye_ratio = abs(Ys[37]-Ys[41])/abs(Xs[36] - Xs[39])
	(l_t,r_t,l_b,r_b) = eye_curves(Xs, Ys)
	#(l_h,r_h) = brow_heights(Xs,Ys)
	return [mouth_ratio,bot_m_c,top_m_c,eye_ratio,l_t,r_t,l_b,r_b] + eyebrow(Ys[17:22]) + eyebrow(Ys[22:27])

def find_mean_neutral_face(finalMatrix):
	neutralMatrix = []
	for row in finalMatrix:
		if(row[len(row) - 1] == 7):
			#neutralMatrix.append([])
			neutralMatrix.append([row[x] for x in range(0 , len(row) - 1) ])
	array1 = np.array([np.array(a) for a in neutralMatrix])
	#print array1
	out =  np.mean(array1, axis=0)
	#print out
	return list(out)


with open('data.csv' , 'rb') as csvfile:
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
	for element in range (0 , 66):
		data[dataRow].append(row[element])
	vec1 = data[dataRow][0:66]	
	data[dataRow][0:66] = (v - np.mean(vec1) for v in vec1)
	#data[dataRow][0:66] = [(v-np.mean(vec1))/np.std(vec1) for v in vec1]
	for element in range (66 , len(row) - 1):
		data[dataRow].append(row[element])
	vec2 = data[dataRow][66:]
	data[dataRow][66:] = (v - np.mean(vec2) for v in vec2)	
	#data[dataRow][66:] = [(v-np.mean(vec2))/np.std(vec2) for v in vec2]
	data[dataRow] = process_points(data[dataRow])
	#print ','.join([ str(x) for x in data[dataRow]])
	dataRow = dataRow + 1


########################

#10 fold cross validation
#Break data into 10 sets of size n/10.
#Train on 9 datasets and test on 1.
#Repeat 10 times and take a mean accuracy.


idx = range(len(labels))
random.shuffle(idx)

data = [data[i] for i in idx]
labels = [labels[i] for i in idx]


scores = []

arrayData = np.asarray(data)


arrayLabels = np.asarray(labels)

#########################################################
first = True
kf = KFold(len(labels), n_folds=10)
for train, test in kf:
	#print("%s %s" % (train, test))

	trainingData = arrayData[train]
	testingData = arrayData[test]

	trainingLabels = arrayLabels[train]
	testingLabels = arrayLabels[test]

	clf = svm.SVC(kernel='poly')

	clf.fit(trainingData, trainingLabels)
	estimatedLabels = clf.predict(testingData)

	if first:
		first = False
		confusion_m = confusion_matrix(testingLabels, estimatedLabels , range(1,8))
	else:
		confusion_m = np.add(confusion_m ,  confusion_matrix(testingLabels, estimatedLabels , range(1,8)))

	#print confusion_matrix(testingLabels, estimatedLabels)
	score = clf.score(testingData , testingLabels)

	estimatedLabels = clf.predict(testingData)
	#print confusion_matrix(testingLabels, estimatedLabels)
	scores.append(score)



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




















	
