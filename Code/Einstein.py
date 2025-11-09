# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 14:36:32 2025

@author: rhizmeri
"""

#imports
import numpy as np
import matplotlib.pyplot as plt

def rk4(f, state, dt): #Runge-Kutta integration, 4th degree
    k1 = f(state)
    k2 = f(state + 0.5 * dt * k1)
    k3 = f(state + 0.5 * dt * k2)
    k4 = f(state + dt * k3)
    return state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

def f(state): #system of equations
    r = state[0]
    #theta = state[1] #used overall but needed in defining the equation
    rdot   = state[2]
    thetadot  = state[3]
    rddot     = -GM / r**2 + r * thetadot**2 - 3*GM*thetadot**2 / c**2
    thetaddot = -2 * rdot * thetadot / r
    return(np.array([rdot, thetadot, rddot, thetaddot]))

plt.close('all') #don't want overlapping plots

#main program

GM = 1      #gravitational constant * mass of blackhole,mass of body is negligeable in comparison
c  = 1      #speed of light
r0 = 4     #initial radius
theta0 = 0  #initial phase
rdot0  = 0  #rate of change of radius
circular_orbit = np.sqrt(GM*c**2 /(r0**2 * (r0*c**2 - 3*GM))) #always gives circular orbit
thetadot0      = circular_orbit * 1.0001 #remove circ_orb for a constant

#initial state of constants
state0  = np.array([r0, theta0, rdot0, thetadot0]) 
dt = 0.000001         #time-step, diff in time between each point
tf = 10     #final time. 1 is the time for [fill in after research done]
nstep = int(tf/dt + 1)       #maximum number of points. +1 so that even number is given
path  = np.zeros((nstep, 4)) #4D array of initial path with state fille with zeros
path[0,:] = state0           #initial value for path
time  = np.zeros(nstep)      #array of each time interval, filled with zeros

if tf <= 50000:
    tf_split = 50  #prints tf, split into divisions
else:
    tf_split = 10

jfinal  = nstep #in case object doesn't fall in black hole
time[0] = 0     #where do you want time to begin?
for j in range(nstep-1):
    time[j+1] = time[j] + dt #filling in the time array
    oldstate  = path[j, :]
    newstate  = rk4(f, oldstate, dt)
    path[j+1, :] = newstate
    if newstate[0] < 0.1: #if radius < 0.2 units
        jfinal = j        #object falls in blackhole, no need for zeros to be plotted on graphs
        print("End of simulation")
        print("Object fell into black hole")
        break
    elif np.mod(j, max(1, int((nstep-1)/tf_split))) == 0:
              #.[number of decimals]f
        print(f"{time[j]:.0f} units of time") #in case of long sim to know how much time is left
    elif j == nstep - 2:
        print("simulation complete")  #simulation completed and didn't fall in BH

#Better names
#Stopping plots at jfinal
rpath = path[:jfinal, 0]
thetapath = path[:jfinal, 1]
rdotPath  = path[:jfinal, 2]
thetadotPath = path[:jfinal, 3]
time  = time[:jfinal]

#----------------------------------------------
plt.figure('time / radius',figsize=(12,6))

plt.xlabel('time')
plt.ylabel('radius')
plt.title('rate of radius')
plt.grid('on')
plt.plot(time, rpath)


#----------------------------------------------
plt.figure('time / phase')

plt.xlabel('time')
plt.ylabel('phase')
plt.title('rate of phase')
plt.grid('on')
plt.plot(time, thetapath)

#----------------------------------------------
plt.figure('cartesian sim')

xpath = rpath * np.cos(thetapath)
ypath = rpath * np.sin(thetapath)
plt.plot(xpath, ypath)
plt.plot(0,0, 'ok') #center of BH is black circle
plt.plot(xpath[jfinal-1], ypath[jfinal-1], 'or') #where is the object
plt.title("Cartesian Simulation of object around BH",fontweight='bold',fontsize=15,pad=10)

#creates the event horizon
phi = np.linspace(0, 2*np.pi, 500)
plt.plot(2*np.cos(phi), 2*np.sin(phi), 'k')
plt.grid('on')
plt.axis('scaled')

#----------------------------------------------
#creates vertical graphs in single figure
fig, (ax1, ax2) = plt.subplots(2, 1)

#defines an array of ang mo and energy
AngMo_path = rpath**2 * thetadotPath
Energy = 0.5 * rdotPath**2 + 0.5 * rpath**2 * thetadotPath**2 * (1 - (2*GM / (c**2 * rpath ))) - GM/rpath

#plot and style for top plot
ax1.plot(time, AngMo_path)
ax1.set_xlabel('time')
ax1.set_ylabel('angular momentum')
ax1.set_title('rate of angular momentum')
ax1.axis('tight')
ax1.grid('on')
ax1.set_xlim(0, tf)

#plot and style for bottom plot
ax2.plot(time, Energy)
ax2.set_xlabel('time')
ax2.set_ylabel('energy')
ax2.set_title('rate of energy')
ax2.axis('tight')
ax2.grid('on')
ax2.set_xlim(0, tf)


plt.tight_layout() #plots don't overlap

#----------------------------------------------
plt.figure('radi / theta')

r0 = np.linspace(0, 100, 10000)
thetadot0 = np.sqrt(GM*c**2 /(r0**2 * (r0*c**2 - 3*GM)))

plt.plot(r0, thetadot0) #radius against phase
plt.xlabel('radius')
plt.ylabel('thetadot')

#----------------------------------------------
plt.figure('polar sim')

plt.polar(thetapath, rpath)
plt.polar(phi, 2*(np.cos(phi))**2 + 2*(np.sin(phi))**2,'k') #plots event horizontally

plt.plot(thetapath[jfinal-1], rpath[jfinal-1], 'or') #where is the object
plt.title("Polar Simulation of object around BH",fontweight='bold',fontsize=15,pad=10)

plt.plot(0,0,'ok') #center of BH
plt.grid('on')
#----------------------------------------------

plt.show()