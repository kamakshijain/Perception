#!/usr/bin/env python
# coding: utf-8

# In[14]:


# #######################Project3_GMM_Fitting################################
# Team Members (Group Name - Project Group)
# Kamakshi Jain
# Abhinav Modi
# Rohan Singh
##############################################################################

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
import copy


def calculateGaussianEquation(xcoor, mean, std):
    return (1/(std*math.sqrt(2*math.pi)))*np.exp(-np.power(xcoor - mean, 2.) / (2 * np.power(std, 2.)))

def prepareHistogramData(imagePath, numberOfImages, channel):
    l1 = np.zeros((1, 256))

    for i in range(1,numberOfImages+1):
        image = cv.imread(str(imagePath) +str(i) +".jpg") # "./red_buoy/" +str(i) +".jpg"
        image2 = cv.GaussianBlur(image,(5,5),0)    
        image1 = image2[:,:,channel]
               
        for j in range(0, image1.shape[0]):
            for k in range(0, image1.shape[1]):
                l1[0][int(image1[j,k])] = l1[0][int(image1[j,k])] + 1
    
    l1 = l1 / numberOfImages
    return np.squeeze(l1)
 
def calculateMeanStd(data):
        mean = sum(data*range(0,256))/sum(data)
        std = (np.sum(data*(np.array(range(0,256)) - mean)**2)/sum(data))**(1/2)
        return mean, std

redbuoy_b = prepareHistogramData("./red_buoy/", 131, 0)
redbuoy_g = prepareHistogramData("./red_buoy/", 131, 1)
redbuoy_r = prepareHistogramData("./red_buoy/", 131, 2)
plt.plot(range(0,256), redbuoy_b, 'b')
plt.plot(range(0,256), redbuoy_g, 'g')
plt.plot(range(0,256), redbuoy_r, 'r')
plt.show()

m3, s3 = calculateMeanStd(redbuoy_r)
plt.plot(range(0,256), calculateGaussianEquation(range(0,256), m3, s3), 'r')
plt.show()

greenbuoy_b = prepareHistogramData("./green_buoy/", 42, 0)
greenbuoy_g = prepareHistogramData("./green_buoy/", 42, 1)
greenbuoy_r = prepareHistogramData("./green_buoy/", 42, 2)
plt.plot(range(0,256), greenbuoy_b, 'b')
plt.plot(range(0,256), greenbuoy_g, 'g')
plt.plot(range(0,256), greenbuoy_r, 'r')
plt.show()

m2, s2 = calculateMeanStd(greenbuoy_g)
plt.plot(range(0,256), calculateGaussianEquation(range(0,256), m2, s2), 'g')
plt.show()

yellowbuoy_b = prepareHistogramData("./yellow_buoy/", 42, 0)
yellowbuoy_g = prepareHistogramData("./yellow_buoy/", 42, 1)
yellowbuoy_r = prepareHistogramData("./yellow_buoy/", 42, 2)
plt.plot(range(0,256), yellowbuoy_b, 'b')
plt.plot(range(0,256), yellowbuoy_g, 'g')
plt.plot(range(0,256), yellowbuoy_r, 'r')
plt.show()

m2, s2 = calculateMeanStd(yellowbuoy_g)
plt.plot(range(0,256), calculateGaussianEquation(range(0,256), m2, s2), 'g')
plt.show()

m3, s3 = calculateMeanStd(yellowbuoy_r)
plt.plot(range(0,256), calculateGaussianEquation(range(0,256), m3, s3), 'r')
plt.show()


# In[18]:


def calculateProbabilty(xcoor, mean, std):
    return (1/(std*math.sqrt(2*math.pi))) * np.exp(-(xcoor - mean)**2 / (2* std**(2)))

def em_gmm(imagePath, noOfImages, channel, noOfIterations, noOfGaussians, means, stds):
    pixel = []

    for i in range(1, noOfImages+1):
        image = cv.imread(imagePath + str(i) + ".jpg")
        image = image[:, :, channel]
        r, c = image.shape
        
        for j in range(0, r):
            for m in range(0, c):
                im = image[j][m]
                pixel.append(im)
                
    n = 0
    pixel = np.array(pixel, dtype = 'float64')
    tempmeans = means
    tempstds = stds

    while (n != noOfIterations):
        prob = []
        bassianprob = []
        probGau = 1/noOfGaussians

        for j in range(0,noOfGaussians):
            prob.append(calculateProbabilty(pixel, tempmeans[j], tempstds[j]))
        
        denom = np.zeros((1,len(pixel)))
        
        for j in prob:
            denom = denom + (np.array(j) * probGau)
       
        for j in prob:
            bassianprob.append(np.divide(np.array(j) * probGau, np.array(denom[0])))
        
        tempmeans = []
        tempstds = []
        for j in range(0, noOfGaussians):
            tempmeans.append(np.sum(bassianprob[j] * np.array(pixel)) / np.sum(bassianprob[j]))
            tempstds.append(((np.sum(bassianprob[j] * ((np.array(pixel) - tempmeans[j]) ** (2)))) / (np.sum(bassianprob[j]))) ** (1 / 2))
        
        n = n + 1
        
    return tempmeans, tempstds
    
def drawGaussians(means, stds):
    g = []
    y = []
    
    ran = list(range(256))
    
    for i in range(0, len(means)):
        plt.plot(ran, stats.norm.pdf(ran, means[i], stds[i]))
    plt.show()
    
    for i in range(0, len(means)):
        g.append(calculateGaussianEquation(np.array(ran), means[i], stds[i]))
    
    temp = 0
    for i in range(0, len(g[0])):
        for j in range(0, len(g)):
            temp = temp + (1 * g[j][i])
        y.append(temp)
        
    plt.plot(ran, y)
    plt.show()

print('Gaussian Mixture Model of Red Buoy')
means, stds = em_gmm('./red_buoy/', 131, 2, 50, 3, [50, 100, 150], [10, 10, 10])
drawGaussians(means, stds)

print('Gaussian Mixture Model of Green Buoy')
means, stds = em_gmm('./green_buoy/', 42, 1, 50, 3, [50, 100, 150], [10, 10, 10])
drawGaussians(means, stds)


# In[ ]:




