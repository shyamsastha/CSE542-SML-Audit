"""
Assignment 1 SML - Audit
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

"""
Q1
"""
#create a randomly linear spaced distribution
X = np.linspace(-4,10,140)

#generate the likelihoods
w1_pdf = norm.pdf(X, loc=2, scale=1)
w2_pdf = norm.pdf(X, loc=5, scale=1)

#generate likelihood ratio
ll_ratio =w1_pdf/w2_pdf

#plot P(X/W1) vs X
plt.plot(X, w1_pdf)
plt.xlabel('X')
plt.ylabel('P(X/w1)')
plt.xlim(-4,10)
plt.ylim(0,0.5)
plt.title('P(X/w1) vs X')
plt.show()

#plot P(X/W2) vs X
plt.plot(X, w2_pdf)
plt.xlabel('X')
plt.ylabel('P(X/w2)')
plt.xlim(-4,10)
plt.ylim(0,0.5)
plt.title('P(X/w2) vs X')
plt.show()

#plot P(X/W1)/P(X/W2) vs X
plt.plot(X, ll_ratio)
plt.xlabel('X')
plt.ylabel('P(X/w1)/P(X/w2)')
plt.xlim(-4,10)
plt.ylim(0,0.5)
plt.title('P(X/w1)/P(X/w2) vs X')
plt.show()

#plot decision boundary for reference
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(X, w1_pdf, label='P(X/w1)')
ax.plot(X, w2_pdf, label='P(X/w2)')
ax.legend(loc='upper left')
plt.xlabel('X')
plt.title('Decision Boundary')
plt.show()

"""
Q2
"""

#create P(W1/X) and P(w2/X) using bayes rule P(W1/X) = P(X/W1) * P(W1)/P(X)
#Assume equal probability; i.e., P(W1) = P(W2) = 1/2
#Since evidence is the summation of all probabilities, P(X) = 1

w1_pt = (1/np.pi) * (1/(1 + np.power((X-3),2))) * 1/2
w2_pt = (1/np.pi) * (1/(1 + np.power((X-5),2))) * 1/2

#plot P(W1/X) and P(W2/X)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(X, w1_pt, label='P(w1/X)')
ax.plot(X, w2_pt, label='P(w2/X)')
ax.legend(loc='upper left')
plt.xlabel('X')
plt.show()