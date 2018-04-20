# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 19:09:05 2018

@author: admin
"""

import numpy as np
import time
from matplotlib import pyplot as plt
from multiprocessing.pool import ThreadPool as Pool
#matplotlib inline

np.random.seed(1)

###SIZE
plt.figure(figsize=(8,6), dpi=80)
sz = 10
pSize = int(sz/4)
#Input data - Of the form [x, y, bias]
x1 = np.random.randint(-10,0, (int(sz/2),3)).tolist() + np.random.randint(0,10, (int(sz/2),3)).tolist()
X = np.array(x1)

yval = [-1 for i in range(int(sz/2))] + [1 for i in range(int(sz/2))]
y = np.array(yval)

def plotter(X,w):
    for d, sample in enumerate(X):
        if (sample[0], sample[1]) == (0,0):
            continue
        if d < int(sz/2):
            plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2, color='green')
        else:
            plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2, color='blue')
    plt.scatter(-2,-5, s=120, marker='_', linewidths=2, color='green')
    plt.scatter(4,3, s=120, marker='+', linewidths=2, color='blue')
    
    x2x3 = np.array([[w[0], w[1], -w[1],w[0]], [w[0], w[1], w[1],-w[0]]])
    X,Y,U,V = zip(*x2x3)
#    print(X,Y,U,V)
    ax = plt.gca()
    ax.quiver(X,Y,U,V, scale=1, color='red')
#    plt.show()




def workk(X,Y,w,errors,epoch,eta):
    error = 0
    for i, x in enumerate(X):
        #Misclassific
        if (Y[i]* np.dot(X[i], w)) < 1:
            w = w + eta * ((X[i] * Y[i]) + (-2 * (1/epoch) * w))
            error = 1
        else:
        #Correct classific
            w = w + eta * (-2 * (1/epoch) * w)
        errors.append(error)
    return errors, w
            
def svm_sgd_plot(X, Y, eta):
    
    w = np.zeros(len(X[0]))
    epochs = 10000 * sz
    tim = 0
    errors = []
    #training GD
    for epoch in range(1, epochs):
        if epoch % 10000 == 0:
            print("Epoch %d " % (epoch))
#        st = time.clock()
        errors, w = workk(X,Y,w,errors,epoch,eta)
#        en = time.clock()
#        tim = (en - st) * 1000000
#        plt.plot(errors, '|')
#        plt.xlabel('Epoch|' + 'eta=' + str(eta))
#        plt.xlim(0,epoch)
#        plt.ylim(0.5,1.5)
#        plt.ylabel('Misclassified')
#        plotter(X, w)
    
    
    return w



ww = svm_sgd_plot(X,y,0.01)
print(ww)
plotter(X,ww)
plt.show()




        
        
    
                