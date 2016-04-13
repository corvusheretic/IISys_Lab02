'''
Created on Mar 21, 2016

@author: kalyan
'''

# Import datasets, classifiers and performance metrics
import numpy as np
import cv2

from sklearn import datasets, neighbors, metrics
from sklearn.decomposition import PCA
from sklearn import manifold

import matplotlib.pyplot as plt

from sklearn import cross_validation

'''
Global Configs :: BEGIN
'''
VIEW_DataImg    =True
SIMPLE_EMBEDDING=False
PROJECT_TO_2D   =True
MANUAL_SPLIT    =False
'''
Global Configs :: END
'''

'''
Auxiliary support functions :: BEGIN
'''

def holdOut(fnDigits,fnData,nSamples,percentSplit=0.8):
    if(MANUAL_SPLIT):
        #Split the data into training and test set split
        n_trainSamples = int(nSamples*percentSplit) 
        trainData   = fnData[:n_trainSamples,:]
        trainLabels = fnDigits.target[:n_trainSamples]
        
        testData    = fnData[n_trainSamples:,:]
        expectedLabels  = fnDigits.target[n_trainSamples:]
    else:
        trainData, testData, trainLabels, expectedLabels = cross_validation.train_test_split(fnData, fnDigits.target, 
                                                                             test_size=(1.0-percentSplit), random_state=0)
    # k-NearestNeighbour Classifier instance
    n_neighbors = 10
    kNNClassifier = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
    # trains the model
    kNNClassifier.fit(trainData, trainLabels) 
    
    predictedLabels = kNNClassifier.predict(testData)
    
    #Display classifier results
    print("Classification report for classifier %s:\n%s\n"
          % ('k-NearestNeighbour', metrics.classification_report(expectedLabels, predictedLabels)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expectedLabels, predictedLabels))
    
    print('holdOut :: Done.')

def kFoldCrossValidation(fnDigits,fnData,kFold=5):
    
    # k-NearestNeighbour Classifier instance
    n_neighbors = 10
    kNNClassifier = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
    
    scores = cross_validation.cross_val_score(kNNClassifier, fnData, fnDigits.target, cv=kFold)
    
    print(scores)
    
    print('kFoldCrossValidation :: Done.')
    
'''
Auxiliary support functions :: END
'''


'''
 Main program flow :: BEGIN
'''
# The digits dataset
digits = datasets.load_digits()

#Code to view data
if(VIEW_DataImg):
    i = 0
    for image in digits.images:
        if(i < 10):
            imMax = np.max(image)
            image = 255*(np.abs(imMax-image)/imMax)
            
            res = cv2.resize(image,(100, 100), interpolation = cv2.INTER_CUBIC)
            
            cv2.imwrite('digit_'+str(i)+'.png',res)
            i+=1
        else:
            break

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

if(PROJECT_TO_2D):
    nComp = 2
else:
    nComp = 3
    
if(SIMPLE_EMBEDDING):
    #Linear Embedding
    #PCA for dimensionality reduction
    pca = PCA(n_components=nComp)
    X_trans = pca.fit_transform(data)
    
else:
    #Manifold Embedding
    tsne = manifold.TSNE(n_components=nComp, init='pca', random_state=0)
    X_trans = tsne.fit_transform(data)

if(PROJECT_TO_2D):
    plt.scatter(X_trans[:,0], X_trans[:,1])
    plt.show()
else:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_trans[:,0], X_trans[:,1], X_trans[:,2])
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    plt.show()

print('Data Visualization :: Done.')

holdOut(digits,data,n_samples,0.7)
kFoldCrossValidation(digits,data,5)

'''
 Main program flow :: END
'''
