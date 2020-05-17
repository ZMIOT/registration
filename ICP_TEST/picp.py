import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from functools import partial

source=np.loadtxt('data/face2.txt')
target=np.loadtxt('data/face3.txt')
R=0
t=0
D=source.shape[1]
Nx=source.shape[0]
Ny=target.shape[0]
pi=1/Nx
max_iternum=100
lmt=1.2



class picp(object):
    def __init__(self):
        self.R=np.array([[1,0,0],[0,1,0],[0,0,1]])
        self.t=np.array([0,0,0])
        self.t=self.t.T
        self.p0=1/Nx
        self.X = source
        self.s=1
        self.Y = target
        self.m = self.X.shape[1]
        self.lanmta=1.3
        self.RMS=0
    def nearest_neighbor(self,src, dst):
        '''
        Find the nearest (Euclidean) neighbor in dst for each point in src
        Input:
            src: Nxm array of points
            dst: Nxm array of points
        Output:
            distances: Euclidean distances of the nearest neighbor
            indices: dst indices of the nearest neighbor
        '''

        #assert src.shape == dst.shape

        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(dst)
        distances, indices = neigh.kneighbors(src, return_distance=True)
        return distances.ravel(), indices.ravel()
    def best_fit_transform(self,A, B,k):
        '''
        Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
        Input:
          A: Nxm numpy array of corresponding points
          B: Nxm numpy array of corresponding points
        Returns:
          T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
          R: mxm rotation matrix
          t: mx1 translation vector
        '''
        assert A.shape == B.shape

        # get number of dimensions
        m = A.shape[1]

        # translate points to their centroids
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        AA = A - centroid_A
        BB = B - centroid_B

        # rotation matrix
        H = np.dot(AA.T, BB)
        sumH=np.zeros(H.shape)
        for i in range(1,Nx):
            if k==0:
                tempH=(1/i)*H
            else:
                tempH=self.px[i]*H
            sumH=sumH+tempH
        U, S, Vt = np.linalg.svd(sumH)
        R = np.dot(Vt.T, U.T)
        # special reflection case
        if np.linalg.det(R) < 0:
           Vt[m-1,:] *= -1
           R = np.dot(Vt.T, U.T)

        # translation
        t = centroid_B.T - np.dot(R,centroid_A.T)

        # homogeneous transformation
        T = np.identity(m+1)
        T[:m, :m] = R
        T[:m, m] = t
        return T, R, t



    def updateParam(self,X,Y,indices,distances,k):
        P = np.zeros(Nx)
        Xt=self.X.T
        sumsig =0
        sig=0
        if k==0:
            for i in range(1,Nx):
                sig=sig+(1/i)*sum(distances)
            self.sigma2=sig/D
        tempRMS=self.RMS
        for i in range(1,Nx):
            self.RMS=0
            diff=X[i,:]-Y[indices,:]
            diff=np.linalg.norm(diff,ord=2)
            pm=np.exp(-diff/(2*self.sigma2))
            P[i]=pm
            if k==0:
                tempsig=((1/i)*diff)/D
                self.RMS=self.RMS+np.sqrt((1/i)*diff)
            else:
                tempsig=(self.px[i]*diff)/D
                self.RMS = self.RMS+np.sqrt((self.px[i]) * diff)
            sumsig=sumsig+tempsig
        self.sigma2=sumsig if sumsig>(sumsig/self.lanmta) else sumsig/self.lanmta
        den = sum(P)
        self.px = P / den

        err=np.abs(self.RMS-tempRMS)
        if err<0.0001:
            return err
        return err

    def emfunc(self):
        ax = fig1.add_subplot(111, projection='3d')
        visualize(0, 0, self.X, self.Y, ax)
        src = np.ones((self.m + 1, self.X.shape[0]))
        dst = np.ones((self.m + 1, self.Y.shape[0]))
        A=self.X
        B=self.Y

        # # Add noise
        B += np.random.randn(Ny, D) * 5

        src[:self.m, :] = np.copy(A.T)
        dst[:self.m, :] = np.copy(B.T)

        for k in range(100):
            #这里的indices表示对应的最近点的索引，distances表示的每个点对应点的最短距离
            distances, indices = self.nearest_neighbor(src[:self.m, :].T, dst[:self.m, :].T)
            self.R1=self.R
            self.t1=self.t

            T, self.R,self.t = self.best_fit_transform(src[:self.m, :].T, dst[:self.m, indices].T,k)
            #T, self.R, self.t = self.best_fit_transform(src[:self.m, :].T, dst[:self.m, indices].T, k)

            #error,T,self.R,self.t=self.fasticp(normals,src[:self.m, :].T, dst[:self.m, indices].T,self.s,k,distances,indices)
            error=self.updateParam(src[:self.m, :].T, dst[:self.m, indices].T,indices,distances,k)
            src = np.dot(T, src)
            B = np.dot(B, self.R)
            callback(**{'iteration': k, 'error': error, 'X': A, 'Y': B})
            if np.abs(error)<0.0001:
                break
def visualize(iteration, error,X, Y, ax):
    plt.cla()
    ax.scatter(X[:,0],  X[:,1], X[:,2], color='red', label='Target')
    ax.scatter(Y[:,0],  Y[:,1], Y[:,2], color='blue', label='Source')
    ax.text2D(0.87, 0.92, 'Iteration: {:d}\nError: {:06.4f}'.format(iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)

if __name__ == '__main__':
    picp=picp()
    fig1 = plt.figure()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    callback = partial(visualize, ax=ax)
    picp.emfunc()
    plt.show()

