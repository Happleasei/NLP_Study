import numpy as np
import matplotlib.pyplot as plt

x = np.empty((100,2))
x[:,0] = np.random.uniform(0.,100.,size=100)
x[:,1] = 0.75*x[:,0]+3.+np.random.normal(0,10.,size=100)
plt.scatter(x[:,0],x[:,1])
plt.show()

def demean(x):
    return x-np.mean(x,axis=0)
x_demean = demean(x)
plt.scatter(x_demean[:,0],x_demean[:,1])
plt.show()

def f(w,x):
   return np.sum((x.dot(w)**2))/len(x)

def df_math(w,x):
    return x.T.dot(x.dot(w))*2./len(x)

def direction(w):
    return w/np.linalg.norm(w)

def gradient_ascent(df,x,initial_w,eta,n_iters=1e4,epsilon=1e-8):
    w = direction(initial_w)
    cur_iter =  0
    while cur_iter < n_iters:
        gradient=df(w,x)
        last_w = w
        w = w+eta*gradient
        w = direction(w)
        if(abs(f(w,x)-f(last_w,x))<epsilon):
            break
        cur_iter += 1
    return w

initial_w = np.random.random(x.shape[1])
eta = 0.001

w = gradient_ascent(df_math,x_demean,initial_w,eta)
plt.scatter(x_demean[:,0],x_demean[:,1])
plt.plot([0,w[0]*30],[0,w[1]*30],color='r')
plt.show()
