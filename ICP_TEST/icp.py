import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from functools import partial

max_iternum=100
target=np.loadtxt('data/face2.txt')
source=np.loadtxt('data/face3.txt')
A=source
B=target

def visualize(iteration, error,X, Y, ax):
    plt.cla()
    ax.scatter(X[:,0],  X[:,1], X[:,2], color='red', label='Target')
    ax.scatter(Y[:,0],  Y[:,1], Y[:,2], color='blue', label='Source')
    ax.text2D(0.87, 0.92, 'Iteration: {:d}\nError: {:06.4f}'.format(iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)

def icp(X,Y):
    ax = fig1.add_subplot(111, projection='3d')
    visualize(0, 0, Y, X, ax)
    for i in range(max_iternum):
        centerA=np.mean(X,axis=0)
        centerB=np.mean(Y,axis=0)
        AA=X-centerA
        BB=Y-centerB
        H=np.dot(AA.T,BB)
        U,_,V=np.linalg.svd(H)
        R=np.dot(V.T,U.T)
        t=centerB.T-np.dot(R,centerA.T)
        Y=np.dot(X,R)+t
        callback(**{'iteration': i, 'error': 0, 'X': Y, 'Y': X})

if __name__ == '__main__':
    fig1 = plt.figure()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    callback = partial(visualize, ax=ax)
    icp(A,B)
