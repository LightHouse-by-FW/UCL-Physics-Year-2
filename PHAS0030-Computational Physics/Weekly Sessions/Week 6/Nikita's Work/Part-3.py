import numpy as np
import matplotlib.pyplot as plt

def calc_M(N, zeta, V, dt):
    onDiag = np.ones(N)*(2*(2+zeta*1j)) + 2*dt*1j*V
    offDiag = np.ones(N-1)*(-zeta*1j)
    return np.diag(onDiag) + np.diag(offDiag, k=1) + np.diag(offDiag, k=-1)

def calc_N(N, zeta, V, dt):
    onDiag = np.ones(N)*(2*(2-zeta*1j)) - 2*dt*1j*V
    offDiag = np.ones(N-1)*1j
    return np.diag(onDiag) + np.diag(offDiag, k=1) + np.diag(offDiag, k=-1)

dx=0.5
N= int(200/dx)
x = np.arange(-100, 100, dx)
V=np.zeros(N)
dt = 0.1
zeta = dt/(dx**2)
matr_N = calc_N(N,zeta,V,dt)
matr_M = calc_M(N,zeta,V,dt)
M_inv_N = np.dot(np.linalg.inv(matr_M),matr_N)


k=1
x_0 = -75
sig = 10
wave_func = np.exp(1j*k*x)*np.exp(-(x-x_0)**2/sig**2)
#plt.plot(x,np.real(wave_func))
#plt.plot(x,np.imag(wave_func))
fig = plt.figure(figsize=(12,6))
graph = 0
for i in range(1151):
    wave_func_next = np.dot(M_inv_N,wave_func)
    wave_func = wave_func_next
    if(i%150==0):
        graph+=1
        subplot = fig.add_subplot(2,4,graph)
        subplot.plot(x, np.real(wave_func), label="real")
        subplot.plot(x, np.imag(wave_func), label="imag")
        subplot.set_title("index=" + str(i))

#plt.plot(x,np.real(wave_func))
#plt.plot(x,np.imag(wave_func))
plt.show()
