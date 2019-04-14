import os, sys
# try:
# 	os.chdir(os.path.join(os.getcwd(), 'Project3'))
# 	print(os.getcwd())
# except:
# 	pass
#
# try:
#     sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
# except:
#     pass

import pickle
import math
import numpy as np
import numpy.matlib
import numpy.random
import matplotlib.pyplot as plt
import cv2

cap = cv2.VideoCapture('./data/detectbuoy.avi')

def gauss_dist(x,mean,sigma):
    g = (1/math.sqrt(np.absolute((2*math.pi)**D *np.linalg.det(sigma))))*(math.e**(-0.5*np.matmul(np.matmul((x-mean),np.linalg.inv(sigma)),(x-mean).T)))
    return g
# r = orange[2]
# g = orange[1]
# b = orange[0]
# mean_r = np.mean(r)
# mean_g = np.mean(g)
# mean_b = np.mean(b)
# sigma = np.cov((b,g,r))
# with open("orange.pickle","rb") as f:
#     u = pickle._Unpickler(f)
#     u.encoding = 'latin1'
#     orange = u.load()
#     print(orange)
pickle_in = open("orange.pickle","rb")#, encoding='latin1')
orange = pickle.load(pickle_in)

global D
D = 2
K = 9
ld = 1
ud = ld+D     #ud <=3
#Initialize mean values for each gaussian(3)
print(sys.maxsize)
# seed = np.random.randint(1,100000000)
# rng = random.Random(seed)
seed = 499591493
np.random.seed(seed)
print("Seed was:", seed)
pi_ = np.random.rand(K,1)
pi_state = np.random.get_state()
pi_ = pi_/np.sum(pi_)
print(pi_)
mean_ = 20*(np.random.randn(K,D))+200
mean_state = np.random.get_state
# print(mean_state,pi_state)

sigma_ = 100*np.reshape((np.matlib.repmat(np.identity(D),1,K)),[D,K,D])

# pi_ = np.array([[0.80498806],[0.17942201],[0.01558993]])
# mean_ = np.array([[167.12467309, 247.82011282],[233.44306862, 230.75108718],[147.59144439, 173.44794164]])
# sigma_ = np.array([[[1471.45452655, -572.89455568],[ 866.78334693,  593.41319722],[3136.04240966,  498.75328477]],
#         [[-572.89455568,  395.30214626],[ 593.41319722,  487.56287171],[ 498.75328477 , 556.34501654]]])

th = 0.0001

posteriors = np.zeros([orange.shape[0],K])
max_iters = 100
iters = 0
orange_ = orange[:,np.newaxis]
print(orange_.shape)
orange_x = orange_[:,:,ld:ud]
print("test",orange_x.shape)

print('pi',pi_)
print('mean',mean_)
print('sigma',sigma_[:,0,:])

while iters<max_iters:
    for i in range(K):
        for j in range(orange.shape[0]):
            # print(orange_x[j].shape)
            # print(sigma_[:1,:])
            posteriors[j,i] = pi_[i]*gauss_dist(orange_x[j],mean_[i,:],sigma_[:,i,:])
    #         j = j*50
    sum_posteriors = np.sum(posteriors,axis = 1)#[:,0]+posteriors[:,1]+posteriors[:,2]
    print(posteriors.shape)
    for i in range(posteriors.shape[0]):
        posteriors[i,:]=posteriors[i,:]/sum_posteriors[i]
    prev_mean = mean_
    ##Maximization step:
    for i in range(K):
        temp = np.zeros([D,D])
        sum_ = temp
        for j in range(orange.shape[0]):
            temp = posteriors[j,i]*np.matmul((orange_x[j]-mean_[i,:]).T,(orange_x[j]-mean_[i,:]))
            sum_ = sum_+temp

        #Update mean, posterior and covariance
        sigma_[:,i,:] = sum_/np.sum(posteriors[:,i])
        mean_[i,:] = np.sum(np.reshape(posteriors[:,i],(posteriors.shape[0],1))*orange[:,ld:ud],axis=0)/np.sum(posteriors[:,i])
        pi_[i] = np.mean(posteriors[:,i])

    iters = iters+1
    if np.linalg.norm(prev_mean-mean_)<th:
        break
prev_mean=mean_
print('pi',pi_)
print('mean',mean_)
print('sigma',sigma_)

# for i in range(3):
#     for j in range(orange.shape[0]):
#         posteriors[j,i] = pi_[i]*gauss_dist(orange_x[j],mean_[i,:],sigma_[:,i,:])

print(posteriors.shape)

gauss_param = [mean_ , sigma_, pi_]    #mean,covariance,pi

print(gauss_param[1].shape)
i = 0
frame = cv2.imread('./data/frames/1.png')

copy = frame.copy()
copy = cv2.cvtColor(copy,cv2.COLOR_BGR2HSV)
mask = cv2.inRange(copy,np.array([0,80,150]),np.array([40,255,255]))
frame = cv2.bitwise_and(frame,frame,mask = mask)
frame = cv2.medianBlur(frame,5)
cv2.imshow("mask",frame)
cv2.waitKey(0)
while(True):
    # ret,frame = cap.read()
    print(frame.shape)
    print(frame[0,1,:])
    clone = frame.copy()
    clone = cv2.cvtColor(clone,cv2.COLOR_BGR2GRAY)
    # m,n = frame.shape
    mod = np.zeros((frame.shape[0],frame.shape[1]))
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            for k in range(K):
                posterior = gauss_dist(frame[i,j,ld:ud],gauss_param[0][k,:],gauss_param[1][:,k,:])
                mod[i,j] = mod[i,j] + gauss_param[2][k] * posterior
                # print(posterior.shape)
    print(np.amax(mod))
    mod = mod/np.amax(mod)
    print(len(mod[mod > 0.5]))


    # print(np.amax(mod))
    # cv2.imshow('mod',mod)
    minth = 0.45
    maxth = 1
    mask = cv2.inRange(mod,minth,maxth)

    kernel =np.ones((5,5),np.uint8)
    dilate=cv2.dilate(mask,kernel,iterations=2)
    res = cv2.bitwise_and(frame,frame,mask = dilate)
    mod = np.zeros((frame.shape[0],frame.shape[1]))
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            for k in range(K):
                posterior = gauss_dist(frame[i,j,ld:ud],gauss_param[0][k,:],gauss_param[1][:,k,:])
                mod[i,j] = mod[i,j] + gauss_param[2][k] * posterior
                # print(posterior.shape)
    print(np.amax(mod))
    mod = mod/np.amax(mod)
    print(len(mod[mod > 0.5]))


    # print(np.amax(mod))
    # cv2.imshow('mod',mod)
    minth = 0.4
    maxth = 1
    mask = cv2.inRange(mod,minth,maxth)
    res = cv2.bitwise_and(frame,frame,mask = mask)
    cv2.imwrite('D=' + str(D) + '_K=_' + str(K) + '_ld=' + str(ld) + '_seed=' + str(seed) + '_minth=' + str(minth) + '_maxth=' + str(maxth) + '.png', res)
    cv2.imshow('res',res)

    while(True):
        if cv2.waitKey(1) & 0xff==ord('q'):
            cv2.destroyAllWindows()
            break
    break
#
# fig, axs = plt.subplots(1, 3)
# # plt.grid(True)
# # We can set the number of bins with the `bins` kwarg
# patches = axs[0].hist(x=orange[:,0]/max(orange[:,0]), bins=256)
# patches = axs[1].hist(x=orange[:,1]/max(orange[:,1]), bins=256)
# patches = axs[2].hist(x=orange[:,2]/max(orange[:,2]), bins=256)
# a = axs[0].set_title('Blue Channel')
# a = axs[1].set_title('Greeen Channel')
# a = axs[2].set_title('Red Channel')
# # plt.ylim([0,500])
# plt.show()
#
#
# #%%
# orange[0]

