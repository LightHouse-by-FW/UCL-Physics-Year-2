import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def update_3d(theta_n, theta_nm1, r):
    del_sqr = np.roll(theta_n, 1, axis=0) + np.roll(theta_n, -1, axis=0) + np.roll(theta_n, 1, axis=1) + np.roll(theta_n, -1, axis=1) - 4*theta_n
    theta_np1 = 2*theta_n - theta_nm1 + r**2 * del_sqr
    return theta_np1

freq = 1
wlength = 1
velocity = freq*wlength
ang_freq = freq * 2*np.pi
N = 50
k = 2*np.pi/wlength

dh = wlength/N
r = 0.2
dt = r*dh/velocity
waveX = np.arange(0, 3*wlength, dh)
waveY = np.arange(0, 3*wlength, dh)
x2d,y2d = np.meshgrid(waveX, waveY)
sigma = 1.0
theta_0 = np.sin(k*x2d - ang_freq * 0) * np.exp(-(y2d-1.5*wlength)**2/sigma)
theta_t = np.sin(k*x2d - ang_freq*dt)  * np.exp(-(y2d-1.5*wlength)**2/sigma)
theta_0[:,N:]=0
theta_t[:,N:]=0
theta_0[:0:] = 0
theta_t[:0:] = 0

#theta_t[N:]=0

fig = plt.figure(figsize=(12,8))
"""
subplot1 = fig.add_subplot(1,3,1, projection='3d')
subplot1.plot_surface(x2d,y2d, theta_t)
subplot1.set_xlabel("x")
subplot1.set_ylabel("y")
"""
theta_n = None
sol100 = None
graph = 1
for i in range(351):
    theta_n = update_3d(theta_t, theta_0, r)
    theta_0 = theta_t
    theta_t = theta_n
    #theta_0[:0:] = 0
    theta_t[0,:] = 0
    theta_t[-1,:] = 0
    theta_t[int(1.5*N),int(1.5*N)]=0

    if(i%50 == 0):
        subplot = fig.add_subplot(2,4,graph, projection='3d')
        subplot.plot_surface(x2d,y2d, theta_t, cmap='viridis')
        subplot.set_xlabel("x")
        subplot.set_ylabel("y")
        graph+=1
    #theta_t[::-1] = 0

    #theta_0[::N] = 0
    #theta_t[-1:] = 0

"""
sp2 = fig.add_subplot(1,3,2, projection='3d')
sp2.plot_surface(x2d,y2d, sol100)
sp2.set_xlabel("x")
sp2.set_ylabel("y")
sp3 = fig.add_subplot(1,3,3, projection='3d')
sp3.plot_surface(x2d,y2d, theta_n)
sp3.set_xlabel("x")
sp3.set_ylabel("y")
"""
plt.show()
