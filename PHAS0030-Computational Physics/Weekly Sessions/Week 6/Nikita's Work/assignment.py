import sys
sys.path.append('C:\\Users\\nikit\\AppData\\Local\\Programs\\Python\\python38\\lib\\site-packages')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def UpdateWave1D(theta_n, theta_nm1, r):
    """
    Calculates the value of theta at the next time step based on the
    previous two time steps
    Inputs:
        theta_n: array like, The wave at the previous time step
        theta_nm1: array like, the wave from two time steps previous
        r: constant/array like, the value of r at the given position of the wave
    Outputs:
        theta_np1: The wave at the next time step.
    """
    theta_ip1 = np.roll(theta_n, 1)
    theta_im1 = np.roll(theta_n,-1)
    theta_np1 = 2*(1-r**2)*theta_n - theta_nm1 + r**2 * (theta_ip1 + theta_im1)
    return theta_np1


#Initiate wave values
freq = 1 #Hz
wlength = 1 #m
velocity = freq*wlength #m s^-1
ang_freq = freq * 2*np.pi # rad/s
k = 2*np.pi/wlength #Wave vector

N = 50 #Number of points per wavelength
Length = 5 #Size of system in wave lengths

#Initiate other values
dx = wlength/N
r = np.full((Length*N), 0.3)
#r = np.ones(Length*N)*0.3
r[2*N:3*N]=0.15
#Constant dt based on initial r
dt = r[0]*dx/velocity
waveX = np.arange(0, Length*wlength, dx)

#Create first two time steps
theta_0 = np.sin(k*waveX - ang_freq * 0)
theta_t = np.sin(k*waveX - ang_freq*dt)
theta_0[N:]=0
theta_t[N+1:]=0
time = [0, dt]
timeSolutions = [theta_0, theta_t]

total_time_steps = 800

for i in range(0, total_time_steps):
    #Find wave_(n+1) based on wave_(n) and wave_(n-1)
    theta_n = UpdateWave1D(theta_t, theta_0, r)
    #Set new values
    theta_0 = theta_t
    theta_t = theta_n
    #Store solution at time t
    timeSolutions.append(theta_n)
    #Append time array, ensures time array has same size as timeSolutions
    time.append(time[-1] + dt)

#Create required 2d numpy arrays
#timeSolutions = np.array(timeSolutions)
print(timeSolutions)
x2d,t2d = np.meshgrid(waveX, time)
#
# #Create and display 3d surface plot
# surf_fig = plt.figure()
# surf_ax = surf_fig.add_subplot(111, projection='3d')
# surf_ax.set_xlabel(r"x ($\lambda$)")
# surf_ax.set_ylabel("t (s)")
# surf_ax.set_zlabel(r"$\psi(x,t)$")
# surf_ax.view_init(90,270)
#
# surf = surf_ax.plot_surface(x2d,t2d, timeSolutions, cmap='viridis')
# surf_col = surf_fig.colorbar(surf)
# surf_col.ax.set_ylabel(r"$\psi(x,t)$", rotation=270)
#
# plt.show()
import matplotlib.animation as animation

def update_line(num, data, line):
    line.set_ydata(data[num])
    return line,
fig1 = plt.figure()
ax = fig1.add_subplot(111)

l, = ax.plot(waveX, timeSolutions[0], 'r-')
line_ani = animation.FuncAnimation(fig1, update_line, len(timeSolutions), fargs=(timeSolutions, l),
                                   interval=4, blit=True)

#line_ani.save('myAnimation.gif', writer='PillowWriter', fps=120)
plt.show()
