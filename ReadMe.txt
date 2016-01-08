This file includes instructions which demonstrate how to run the source code:

Python files :

In order to generate the accuracy and confusion matrix for the plain landmarks run : python classify_plain_landmarks.py
In order to generate the accuracy and confusion matrix for the processed landmarks run : python classify_processed_landmarks.py
In order to generate the accuracy and confusion matrix for the plain eigenfaces run : python classify_eigenfaces.py
In order to generate the accuracy and confusion matrix for the processed landmarks + eigenfaces run : python classify_processed_eigenfaces.py
In order to generate the accuracy and confusion matrix for the plain fisherfaces run : python classify_fisherfaces.py
In order to generate the accuracy and confusion matrix for the fisherfaces + processed landmarks run : python classify_processed_fisherfaces.py

Some of the files pull data from csv files. All these files have been included. 

Some files were not added to this folder , since they run the CSIRO-face-analysis-sdk to get the landmarks. This software was installed on our devices so that we could use it. 


Matlab files

The Matlab files were all used to generate Eigenfaces and Fisherfaces , as well as concantenate them with the processed landmarks. 
