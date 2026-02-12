# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 23:33:41 2026

@author: rodri
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 06:41:23 2025

@author: rhizmeri
"""

import matplotlib.pyplot as plt
import numpy as np

def f(state):
    x, v = state
    xdot = v
    vdot = -(K/mass) * x
    return np.array([xdot, vdot])

def rkf45(f, state, dt):
    #constants for function
    dt_min = 1e-9
    dt_max = 1
    tol = 1e-9 #use smaller for high precision
    
    for attempt in range(10):#so no inf loop, change if need greater accuracy
        #base values for k
        k1 = f(state)
        k2 = f(state + dt/4 * k1)
        k3 = f(state + dt * (3/32 * k1 + 9/32 * k2))
        k4 = f(state + dt * (1932/2197 * k1 - 7200/2197 * k2 + 7296/2197 * k3))
        k5 = f(state + dt * (439/216 * k1 - 8 * k2 + 3680/513 * k3 - 845/4104 * k4))
        k6 = f(state + dt * (-8/27 * k1 + 2 * k2 - 3544/2565 * k3 + 1859/4104 * k4 - 11/40 * k5))
        
        #computing rk4 and rk5
        coeff = [25/216, 0, 1408/2565 , 2197/4104  , -1/5 , 0, 
                 16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55] #coefficiants
       
        
        #error estimator
        rk4 = state + dt * (coeff[0]*k1 + coeff[1]*k2 + coeff[2]*k3 + coeff[3]*k4 + coeff[4]*k5 + coeff[5]*k6)
        #actual solution
        rk5 = state + dt * (coeff[6]*k1 + coeff[7]*k2 + coeff[8]*k3 + coeff[9]*k4 + coeff[10]*k5 + coeff[11]*k6)
        
        #compute error
        error = np.max(np.abs(rk5 - rk4))
        if error > 0:
            safety = 0.84 * (tol/error)**0.25
        else:
            safety = 2
        
        #compute new stepsize
        dt_new = np.clip(dt*safety, dt_min, dt_max)
        
        if (error <= tol) or (dt <= dt_min):
            return rk5, dt_new, True #end loop
        else:
            dt = dt_new #retry loop
            
            
    return rk5, dt_min, True #loop ended just use lowest dt

plt.close('all')

#main program     
dt     = 0.1
K    = 1.0
mass = 10.0
x0   = 2.0
v0   = 1.0
tf   = 1

tellNum = 10000

state = np.array([x0, v0])
nstep = int(tf/(dt * 10**-3) + 1) + 100
path  = np.zeros((nstep, 2))
time  = np.zeros(nstep)
path[0, :] = [x0, v0]
dt_now = dt

time[0] = 0
j = 0
dt_now = dt
run_sim = True

while run_sim == True:
    oldstate  = path[j, :]
    newstate, dt_new, accepted = rkf45(f, oldstate, dt_now)
    time[j+1] = time[j] + dt_now
    path[j+1, :] = newstate
    dt_now = dt_new
    j += 1
    if time[j] >= tf or j >= nstep - 2:
        jfinal = j + 1
        run_sim = False
        print("simulation complete")
    elif np.mod(j, max(1, int((nstep-1)/10))) == 0:
        print(f"{time[j]:.1f} units of time")
        
        
time = time[:jfinal]
xpath = path[:jfinal, 0]
vpath = path[:jfinal, 1]
    
omega = np.sqrt(K/mass)
disp  = x0*np.cos(omega * time) + (v0/omega)*np.sin(omega * time)
vel = -x0*omega*np.sin(omega*time) + v0*np.cos(omega*time)

#--------------------------------------------
plt.figure(1)
plt.title('RKF45 vs Analytical - Displacement')
plt.plot(time, xpath, 'b-', label='RKF45', linewidth=2)
plt.plot(time, disp, 'r--', label='Analytical', linewidth=2, alpha=0.7)
plt.xlabel('time')
plt.ylabel('displacement')
plt.legend()
plt.grid('on')
#--------------------------------------------
plt.figure(2)
plt.title('RKF45 vs Analytical - Velocity')
plt.plot(time, vpath, 'b-', label='RKF45', linewidth=2)
plt.plot(time, vel, 'r--', label='Analytical', linewidth=2, alpha=0.7)
plt.xlabel('time')
plt.ylabel('velocity')
plt.legend()
plt.grid('on')
#--------------------------------------------
plt.figure(3)
plt.title('Error in Displacement')
plt.plot(time, xpath - disp, 'g-', linewidth=1.5)
plt.xlabel('time')
plt.ylabel('error')
plt.grid('on')
#--------------------------------------------

plt.show()
