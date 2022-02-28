# -*- coding: utf-8 -*-
"""
Assignment 2 - SML - Audit
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
from scipy.stats import multivariate_normal
from numpy.linalg import eig

## Q1. a.
##### Generating 200 Multivariate Bernoulli Distributed Samples

c1x1b = bernoulli.rvs(0.5, size=100)
c1x2b = bernoulli.rvs(0.8, size=100)
c2x1b = bernoulli.rvs(0.9, size=100)
c2x2b = bernoulli.rvs(0.2, size=100)

c1x1 = []
c1x2 = []
c2x1 = []
c2x2 = []

# Converting the distribution into iteratable lists
for x1 in c1x1b:
    c1x1.append(x1)
for x2 in c1x2b:
    c1x2.append(x2)
for x1 in c2x1b:
    c2x1.append(x1)
for x2 in c2x2b:
    c2x2.append(x2)

c1x1_train = []
c1x2_train = []
c2x1_train = []
c2x2_train = []

c1x1_test = []
c1x2_test = []
c2x1_test = []
c2x2_test = []

# Splitting the data into train test splits
c1x1_train = c1x1[0:50]
c1x2_train = c1x2[0:50]
c2x1_train = c2x1[0:50]
c2x2_train = c2x2[0:50]

c1x1_test = c1x1[50:100]
c1x2_test = c1x2[50:100]
c2x1_test = c2x1[50:100]
c2x2_test = c2x2[50:100]

## Q1. b.
#### Creating the MLE function to be used for Class 1 and Class 2
def MLE(X):
    return ((1/len(X)) * np.sum(X))

#### Creating the list of MLEs calcualted after taking n observations at a time where n ranges from 1 to 50 and plotting the observations versus number of samples taken for the MLE for class 1

c1theta1MLE = MLE(c1x1_train)
c1theta2MLE = MLE(c1x2_train)
thetac1 = [c1theta1MLE, c1theta2MLE]

c1theta1MLEob = [] # Observations taking n samples at a time, where n = 1, 2, 3, ..., n for theta 1 of class 1
for i in range(50):
    c1theta1MLEob.append(MLE(c1x1_train[0:i+1]))

c1theta2MLEob = [] # Observations taking n samples at a time, where n = 1, 2, 3, ..., n for theta 2 of class 1
for i in range(50):
    c1theta2MLEob.append(MLE(c1x2_train[0:i+1]))

n = [i for i in range(1,51)]

# Plotting the MLE observations vs n samples
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(n, c1theta1MLEob, label='theta1 MLE vs n')
ax.plot(n, c1theta2MLEob, label='theta2 MLE vs n')
ax.legend(loc='upper right')
plt.xlabel('n')
plt.ylabel('thetaMLE')
plt.title('thetaMLE vs n for Class 1')
plt.show()

## Q1. c.
#### Creating the list of MLEs calcualted after taking n observations at a time where n ranges from 1 to 50 and plotting the observations versus number of samples taken for the MLE for class 2

c2theta1MLE = MLE(c2x1_train)
c2theta2MLE = MLE(c2x2_train)
thetac2 = [c2theta1MLE, c2theta2MLE]

c2theta1MLEob = [] # Observations taking n samples at a time, where n = 1, 2, 3, ..., n for theta 1 of class 2
for i in range(50):
    c2theta1MLEob.append(MLE(c2x1_train[0:i+1]))

c2theta2MLEob = [] # Observations taking n samples at a time, where n = 1, 2, 3, ..., n for theta 2 of class 2
for i in range(50):
    c2theta2MLEob.append(MLE(c2x2_train[0:i+1]))

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(n, c2theta1MLEob, label='theta1 MLE vs n')
ax.plot(n, c2theta2MLEob, label='theta2 MLE vs n')
ax.legend(loc='lower right')
plt.xlabel('n')
plt.ylabel('thetaMLE')
plt.title('thetaMLE vs n for Class 2')
plt.show()

## Q1. d.
#### Plotting the samples using a scatter plot

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.scatter(n, c1x1_train, label='theta1 MLE vs n')
ax.scatter(n, c1x2_train, label='theta2 MLE vs n')
ax.legend(loc='center')
plt.xlabel('n')
plt.ylabel('c1x1 , c1x2')
plt.title('N training samples for Class 1')
plt.show()

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.scatter(n, c2x1_train, label='theta1 MLE vs n')
ax.scatter(n, c2x2_train, label='theta2 MLE vs n')
ax.legend(loc='center')
plt.xlabel('n')
plt.ylabel('c2x1 , c2x2')
plt.title('N training samples for Class 2')
plt.show()

## Q1. e.
#### Classification of the test samples

# Function to find the discriminant of the Multivariate Bernoulli Distribution
def discriminant(x1, x2, thetac1, thetac2):
    nr = (x1 * np.log(thetac1[0])) + ((1 - x1) * np.log((1 - thetac1[0]))) + (x2 * np.log((thetac1[1]))) + ((1 - x2) * np.log((1 - thetac1[1])))
    dr = (x1 * np.log(thetac2[0])) + ((1 - x1) * np.log((1 - thetac2[0]))) + (x2 * np.log((thetac2[1]))) + ((1 - x2) * np.log((1 - thetac2[1])))
    g = nr/dr
    if g > 0:
        return 1
    else:
        return 2

# Finding out the results for the samples
cc = 0 # Count to keep track of correct classification
for x1, x2 in zip(c1x1_test, c1x2_test):
    g = discriminant(x1, x2, thetac1, thetac2)
    if g == 1:
        cc += 1
for x1, x2 in zip(c2x1_test, c2x2_test):
    g = discriminant(x1, x2, thetac1, thetac2)
    if g == 2:
        cc += 1

n = len(c1x1_test) + len(c2x1_test)
print("The number of correct classifications predicted out of {} test samples is {}".format(n ,cc))

## Q3.c.
#### Computing part a and c

X = np.array([[1, 0],[0, 1]])
mu = np.array([0.5, 0.5]) # Mean of X
Xc = np.array(X-mu) # Centralized Matrix of X
print("Centralized Xc = {}".format(Xc))
SXc = np.cov(Xc)
print("Covariance SXc = {}".format(SXc))
v, U = eig(SXc)
print("Eigan values = {}".format(v))
print("Eigan vectors = {}".format(U))
Y = np.dot(np.matrix.transpose(U),SXc) # U'Xc
print("Y = U'Xc = {}".format(Y))
UYdc = np.array(np.dot(U,Y) + mu) # Decentralized matrix of UY
print("UY + mean(X) = {}".format(UYdc))
d = np.subtract(UYdc, X) # Difference
ds = np.square(d) # Squared difference
MSE = ds.mean() # MSE
print("MSE = {}".format(MSE))

## Q3. d,e.

X = []
r=42
for i in range(2):
    x = multivariate_normal.rvs(mean = 0.5, cov=0.5, size=2, random_state=r)
    r += 1
    X.append(x)
mu = np.array([0.5, 0.5]) # Mean of X
Xc = np.array(X-mu) # Centralized Matrix of X
print("Centralized Xc = {}".format(Xc))
SXc = np.cov(Xc)
print("Covariance SXc = {}".format(SXc))
v, U = eig(SXc)
print("Eigan values = {}".format(v))
print("Eigan vectors = {}".format(U))
Y = np.dot(np.matrix.transpose(U),SXc) # U'Xc
print("Y = U'Xc = {}".format(Y))
UYdc = np.array(np.dot(U,Y) + mu) # Decentralized matrix of UY
print("UY + mean(X) = {}".format(UYdc))
d = np.subtract(UYdc, X) # Difference
ds = np.square(d) # Squared difference
MSE = ds.mean() # MSE
print("MSE = {}".format(MSE))

## Q 3. f.

Up1 = U[0] # For 2 principal component
Up2 = U # For 2 principal components
Y1 = np.dot(np.matrix.transpose(Up1),SXc) # Up1'Xc
Y2 = np.dot(np.matrix.transpose(Up2),SXc) # Up2'Xc
UpY1 = np.array(np.dot(Up1,Y1) + mu) # Decentralized matrix of UY
UpY2 = np.array(np.dot(Up2,Y2) + mu) # Decentralized matrix of UY
MSE1 = (np.square(np.subtract(UpY1,X))).mean()
MSE2 = (np.square(np.subtract(UpY1,X))).mean()
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.scatter(1, MSE1, label='MSE for d = 1')
ax.scatter(2, MSE2, label='MSE for d = 2')
ax.legend(loc='center')
plt.xlabel('nPC')
plt.ylabel('MSE')
plt.title('MSE vs nPC')
plt.show()