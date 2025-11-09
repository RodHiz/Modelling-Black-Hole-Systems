# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 06:41:23 2025

@author: rhizmeri
"""

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.animation as animation


def f(state):
    x, v = state
    xdot = v
    vdot = -(K/mass) * x
    return np.array([xdot, vdot])

def update(state):
    x = state[0]
    v = state[1]
    xnew = x + v*dt
    vnew = v - (K/mass)*x*dt
    return np.array([xnew, vnew])

def rk2(f, state, dt):
    k1 = f(state)
    k2 = f(state + 1 * dt * k1)
    return state + (dt/2.0) * (k1 + k2)

def rk3(f, state, dt):
    k1 = f(state)
    k2 = f(state + 0.5 * dt * k1)
    k3 = f(state + 1 * dt * k2)
    return state + (dt/3.0) * (k1 + k2 + k3)

def rk4(f, state, dt):
    k1 = f(state)
    k2 = f(state + 0.5 * dt * k1)
    k3 = f(state + 0.5 * dt * k2)
    k4 = f(state + dt * k3)
    return state + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)



plt.close('all')

#main program 
    
dt   = 0.9
K    = 1.0
mass = 1.0
x0   = 2.0
v0   = 1.0
tf   = 50

tellNum = 10000

state = np.array([x0, v0])
nstep = int(tf/dt + 1)
path  = np.zeros((nstep, 2))
time  = np.zeros(nstep)
solution   = np.zeros((nstep, 2))
rk2sol = np.zeros((nstep, 2))
rk3sol = np.zeros((nstep, 2))
path[0, :] = [x0, v0]

time[0] = 0
solution[0] = state
rk2sol[0] = state
rk3sol[0] = state
for j in range(nstep-1):
    time[j+1] = time[j] + dt
    oldstate  = path[j, :]
    newstate  = update(oldstate)
    path[j+1, :] = newstate
    solution[j+1] = rk4(f, solution[j], dt)
    rk2sol[j+1] = rk2(f, rk2sol[j], dt)
    rk3sol[j+1] = rk3(f, rk3sol[j], dt)
    
    if np.mod(j, max(1, int((nstep-1)/10))) == 0:
        print(f"{time[j]:.1f} units of time")
    elif j == nstep - 2:
        print("simulation complete")

xpath = path[:, 0]
vpath = path[:, 1]
    
omega = np.sqrt(K/mass)
disp  = x0*np.cos(omega * time) + v0*np.sin(omega * time)
#harmonic form
#disp = np.sqrt(x0**2 + v0**2) * np.sin(time + np.arctan(x0/v0))
vel = -x0*omega*np.sin(omega*time) + v0*omega*np.cos(omega*time)

xnew = solution[:, 0]
vnew = solution[:, 1]

xrk2 = rk2sol[:, 0]
vrk2 = rk2sol[:, 1]

xrk3 = rk3sol[:, 0]
vrk3 = rk3sol[:, 1]

fig, (ax1, ax2) = plt.subplots(1, 2) #plotes them horizontally, flip to (2,1) for vertical

#plot and style for top plot
ax1.plot(time, xpath)
ax1.set_xlabel('time')
ax1.set_ylabel('displacememt')
ax1.set_title('Euler - displacement')
ax1.axis('tight')
ax1.grid('on')
ax1.set_xlim(0, tf)

#plot and style for bottom plot
ax2.plot(time, xnew)
ax2.set_xlabel('time')
ax2.set_ylabel('displacement')
ax2.set_title('rk4 - displacement')
ax2.axis('tight')
ax2.grid('on')
ax2.set_xlim(0, tf)


plt.tight_layout() #plots don't overlap
'''
#--------------------------------------------
plt.figure(1)

plt.title('rk4 - displacement')
plt.plot(time, xnew)
plt.xlabel('time')
plt.ylabel('xpath')
plt.grid('on')
#--------------------------------------------

#--------------------------------------------
plt.figure(2)

plt.title('Euler - displacement')
plt.plot(time, xpath)
plt.xlabel('time')
plt.ylabel('xpath')
plt.grid('on')
#--------------------------------------------
'''
'''
#--------------------------------------------
plt.figure(3)

plt.title('rk4 - velocity')
plt.plot(time, vnew)
plt.xlabel('time')
plt.ylabel('vpath')
plt.grid('on')
#--------------------------------------------

#--------------------------------------------
plt.figure(4)

plt.title('Euler - velocity')
plt.plot(time, vnew)
plt.xlabel('time')
plt.ylabel('vpath')
plt.grid('on')
#--------------------------------------------
'''

fig, (ax1, ax2) = plt.subplots(1, 2)

#plot and style for top plot
ax1.plot(time, vpath)
ax1.set_xlabel('time')
ax1.set_ylabel('velocity')
ax1.set_title('Euler - velocity')
ax1.axis('tight')
ax1.grid('on')
ax1.set_xlim(0, tf)

#plot and style for bottom plot
ax2.plot(time, vnew)
ax2.set_xlabel('time')
ax2.set_ylabel('velocity')
ax2.set_title('rk4 - velocity')
ax2.axis('tight')
ax2.grid('on')
ax2.set_xlim(0, tf)


plt.tight_layout() #plots don't overlap
#--------------------------------------------
plt.figure(5)

plt.title('rk4 - Energy')
energy = 0.5 * mass * vnew**2 + 0.5 * K * xnew**2
plt.plot(time, energy)
plt.xlabel('time')
plt.ylabel('energy')
plt.axis('tight')
plt.grid('on')
#--------------------------------------------

#--------------------------------------------
plt.figure(6)

plt.title('Euler - Energy')
energy = 0.5 * mass * vpath**2 + 0.5 * K * xpath**2
plt.plot(time, energy)
plt.xlabel('time')
plt.ylabel('energy')
plt.axis('tight')
plt.grid('on')
#--------------------------------------------

#--------------------------------------------
plt.figure(7)

plt.title('true displacement')
plt.plot(time, disp)
plt.xlabel('time')
plt.ylabel('dis')
plt.grid('on')
#--------------------------------------------

#--------------------------------------------
plt.figure(8)

plt.title('true velocity')
plt.plot(time, vel)
plt.xlabel('time')
plt.ylabel('vel')
plt.grid('on')
#--------------------------------------------

#--------------------------------------------
plt.figure(9)

plt.title('true Energy')
energy = 0.5 * mass * vel**2 + 0.5 * K * disp**2
plt.plot(time, energy)
plt.xlabel('time')
plt.ylabel('energy')
plt.axis('tight')
plt.grid('on')
#--------------------------------------------

#--------------------------------------------
plt.figure(10)

plt.title('displacement')
#plt.plot(time, xpath,'r-')
plt.plot(time, xrk2, 'k--')
plt.plot(time, xrk3, 'm-')
plt.plot(time, xnew, 'b-')
plt.plot(time, disp, 'g-.')
plt.legend([ 'rk2', 'rk3', 'rk4', 'true value'])
plt.axis('tight')
plt.grid('on')
#--------------------------------------------


fig, ax = plt.subplots(figsize=(6.5, 4))

eulerE= 0.5 * mass * vpath**2 + 0.5 * K * xpath**2
rk2E  = 0.5 * mass * vrk2**2 + 0.5 * K * xrk2**2
rk3E  = 0.5 * mass * vrk3**2 + 0.5 * K * xrk3**2
rk4E  = 0.5 * mass * vnew**2 + 0.5 * K * xnew**2
trueE = 0.5 * mass * vel**2 + 0.5 * K * disp**2

trueEuler = (disp - xpath)**2
truerk2   = (disp - xrk2 )**2
truerk3   = (disp - xrk3 )**2
truerk4   = (disp - xnew )**2

#line1, = ax.plot([], [], 'b-', label='Euler')
line2, = ax.plot([], [], 'r-', label='rk2')
line3, = ax.plot([], [], 'g-', label='rk3')
line4, = ax.plot([], [], 'm-', label='rk4')

ax.set_xlabel('time')
ax.set_ylabel('R² Error')
ax.set_title('Squared Error vs Time')
ax.legend()
ax.grid('on')

# Set axis limits based on your data
ax.set_xlim(0, tf)
ax.set_ylim(0, max(truerk2.max(), truerk3.max(), truerk4.max()) * 1.1)
#ax.set_ylim(0, max( truerk3.max(), truerk4.max()) * 1.1)

def init():
    # Initialize all elements
    #line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    line4.set_data([], [])
    return   line2, line3, line4  # Return all animated elements

def animate(frame):
    # Animate by showing data up to current frame
    idx = int(frame * len(time) / 200)  # Map frame to data index
    
    #line1.set_data(time[:idx], trueEuler[:idx])
    line2.set_data(time[:idx], truerk2[:idx])
    line3.set_data(time[:idx], truerk3[:idx])
    line4.set_data(time[:idx], truerk4[:idx])
    
    return line2, line3, line4

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=nstep, interval=50, blit=True)

plt.show()
