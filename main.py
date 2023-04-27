#!/usr/bin/env python
#import necessary libraries
import numpy as np
import sklearn.linear_model as ln
from scipy.io import wavfile

#check whether this is a file
#rate,frames = wavfile.read('/classes/ece2720/pe4/bach_8820hz.wav')
rate,frames = wavfile.read('bach_8820hz.wav')

#assume frames is the same as samples and also X
samp = frames
X = frames

# The target Y and feature X vectors
# differ by a shift of one sample
Xtrain = X[0:round(len(samp)/2)-1]
Xtrain = np.reshape(Xtrain,(-1,1))
Ytrain = X[1:round(len(samp)/2)]

# find a* and b*
#do linear regression on training set
reg = ln.LinearRegression().fit(Xtrain,Ytrain)

#R^2 score
rSquared = reg.score(Xtrain,Ytrain)

#print values for report
#a* values
#print(reg.coef_)
#b* values
#print(reg.intercept_)
#R^2 value for training
#print(rSquared)

#The test set is the other half of the samples
Xtest = X[round(len(samp)/2):(len(samp)-1)]
Xtest = np.reshape(Xtest,(-1,1))
Ytest = X[(round(len(samp)/2)+1):len(samp)]

#R^2 value for the test set
rSquaredTest = reg.score(Xtest,Ytest)

#print R^2 value
#print(rSquaredTest)


import numpy as np
import sklearn.linear_model as ln
from scipy.io import wavfile
import matplotlib.pyplot as plt

#check whether this is a file
#rate,frames = wavfile.read('/classes/ece2720/pe4/bach_8820hz.wav')
rate,frames = wavfile.read('bach_8820hz.wav')

#assume frames is the same as samples and also X
samp = frames
X = frames

#define arrays for the values of R^2training and R^2test
#19 windows from 2 to 20
#numWindows = 19
R2training = np.zeros(19)
R2test = np.zeros(19)

#iterate through windows two to twenty using a for loop
for i in range(2, 21):
    
    #size/length of the training set/test set
    trainsize = round(len(samp)/(2))
    
    # initialize zero matrices/vectors of
    # the appropriate dimension
    Xtrain = np.zeros((trainsize-i, i))
    Ytrain = np.zeros(trainsize-i)
    Xtest = np.zeros((trainsize-i, i))
    Ytest = np.zeros(trainsize-i)
    
    #create the training and test sets by copying over the values from samp
    #dependent on the window size
    for j in np.arange(0,trainsize-i):
        Xtrain[j,:] = samp[j:j+i]
        Ytrain[j] = samp[j+i]
        Xtest[j,:] = samp[j+trainsize:j+i+trainsize]
        Ytest[j] = samp[j+i+trainsize]
        
    #do linear regression on the training set
    reg = ln.LinearRegression().fit(Xtrain,Ytrain) 
    
    #add R^2 values to the arrays
    R2training[i-2] = reg.score(Xtrain,Ytrain)
    R2test[i-2] = reg.score(Xtest,Ytest)
    
#test by printing out R^2 values
#print(R2training)
#print(R2test)

#get a* for window size 20
#print(reg.coef_)

#compare value to ridge regression for report
#print((np.linalg.norm(reg.coef_))**2)

#graph with window size on x axis with R^2training and R^2test on the y axis
x = np.arange(2, 21)
plt.plot(x, R2training, label='R^2 training')
plt.plot(x, R2test, label='R^2 test')
plt.title('Unregularized Regression on Bach Piece')
plt.xlabel('window size')
plt.ylabel('R^2')
plt.legend()


import sklearn.linear_model as ln
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np

#rate,frames = wavfile.read('/classes/ece2720/pe4/bach_8820hz.wav')
rate,frames = wavfile.read('bach_8820hz.wav')

#assume frames and X are the same as samples
samp = frames
X = frames

#R^2 values for lambda (14 lambda values chosen)
R2training = np.zeros(14)
R2test = np.zeros(14)

#window size 20
windowSize = 20
#lambda range from 10^7 to 10^18 because TA said that 10^19 and 10^20 take too long to compute
lambdaArray = 10**np.arange(7, 21)

#set up arrays for X and Y train and test
trainsize = round(len(samp)/(2))
Xtrain = np.zeros((trainsize-windowSize, windowSize))
Ytrain = np.zeros(trainsize-windowSize)
Xtest = np.zeros((trainsize-windowSize, windowSize))
Ytest = np.zeros(trainsize-windowSize)

arrayIndex = 0


#loop through with the powers of ten in the lambda array
for i in lambdaArray:  
    #create the training and test sets by copying over the values from samp
    #window size this time is constant
    for j in np.arange(0,trainsize-windowSize):
        Xtrain[j,:] = samp[j:j+windowSize]
        Ytrain[j] = samp[j+windowSize]
        Xtest[j,:] = samp[j+trainsize:j+windowSize+trainsize]
        Ytest[j] = samp[j+windowSize+trainsize]
    
    #lambda is alpha in Ridge
    reg = ln.Ridge(alpha = i).fit(Xtrain, Ytrain)
    
    #add R^2 values to the array
    R2training[arrayIndex] = reg.score(Xtrain,Ytrain)
    R2test[arrayIndex] = reg.score(Xtest,Ytest)
    
    #increment the array index to fill R^2 arrays
    arrayIndex = arrayIndex + 1

#find the maximum value of lamda for R^2 test
#print(lambdaArray[np.argmax(R2test)])

#plot the arrays using the semilog scale
plt.semilogx(lambdaArray, R2training, label="R2training")
plt.semilogx(lambdaArray, R2test, label="R2test")
plt.title('Ridge Regression on Bach Piece')
plt.xlabel('Lambda')
plt.ylabel('R^2')
plt.legend()

#print a* and ||a*||^2 for comparison to unregularized
#print(reg.coef_)
#print((np.linalg.norm(reg.coef_))**2)

import sklearn.linear_model as ln
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np

#rate,frames = wavfile.read('/classes/ece2720/pe4/bach_8820hz.wav')
rate,frames = wavfile.read('bach_8820hz.wav')

#assume frames and X are the same as samples
samp = frames
X = frames

#window size 20
windowSize = 9
#lambda value
lambdaVal = 5*(10**11)

#set up arrays for X and Y train and test
trainsize = round(len(samp)/(2))
Xtrain = np.zeros((trainsize-windowSize, windowSize))
Ytrain = np.zeros(trainsize-windowSize)
Xtest = np.zeros((trainsize-windowSize, windowSize))
Ytest = np.zeros(trainsize-windowSize)

#get the training and test sets
for j in np.arange(0,trainsize-windowSize):
    Xtrain[j,:] = samp[j:j+windowSize]
    Ytrain[j] = samp[j+windowSize]
    Xtest[j,:] = samp[j+trainsize:j+windowSize+trainsize]
    Ytest[j] = samp[j+windowSize+trainsize]
    

#lambda is alpha in lasso
#divide by 2 times the training set size to match the Lasso function from sklearn
reg = ln.Lasso(alpha = lambdaVal/(2*trainsize)).fit(Xtrain, Ytrain)

#print a*
#print(reg.coef_)
