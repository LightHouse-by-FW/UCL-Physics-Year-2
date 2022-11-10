import numpy as np
import matplotlib.pyplot as plt

def explicit_wave_eq_update(theta_n, theta_nm1, r):
    theta_ip1 = np.roll(theta_n, 1)
    theta_im1 = np.roll(theta_n,-1)
    theta_np1 = 2*(1-r**2)*theta_n - theta_nm1 + r**2 * (theta_ip1 + theta_im1)
    return theta_np1

freq = 1
wlength = 1
velocity = freq*wlength
ang_freq = freq * 2*np.pi
N = 50
k = 2*np.pi/wlength

dx = wlength/N
r = 0.1
dt = r*dx/velocity
waveX = np.arange(0, 3*wlength, dx)
