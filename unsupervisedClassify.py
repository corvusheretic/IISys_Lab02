'''
Created on Mar 21, 2016

@author: kalyan
'''
# Import datasets, classifiers and performance metrics
import numpy as np
import cv2

from sklearn import datasets
from sklearn.cluster import KMeans

# The digits dataset
digits = datasets.load_digits()

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

kclass = KMeans(init='k-means++', n_clusters=10, n_init=10)

y_pred = kclass.fit(data)

viewDataImg=True
if(viewDataImg):
    for i in range(y_pred.cluster_centers_.shape[0]):
        centroid = y_pred.cluster_centers_[i,:]
        imMax = np.max(centroid)
        imMin = np.min(centroid)
        image = (255*(np.abs(imMax-centroid+imMin)/imMax)).astype(np.uint8)
        image = image.reshape(8,-1)
        res = cv2.resize(image,(100, 100), interpolation = cv2.INTER_CUBIC)    
        cv2.imwrite('digit_'+str(i)+'.png',res)
        
print('Done.')

