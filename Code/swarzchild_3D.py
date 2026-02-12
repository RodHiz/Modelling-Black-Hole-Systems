# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 20:48:10 2026

@author: rodri
"""

import numpy as np
import matplotlib.pyplot as plt

def rkf45(f, state, dt):
    #constants for function
    dt_min = 1e-12
    dt_max = 1
    tol = 1e-12 #use smaller for high precision
    
    for attempt in range(10):#so no inf loop, change if you need greater accuracy
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

def f(state):
    # State: [r, theta, phi, r_dot, theta_dot, phi_dot]
    r = state[0]
    theta = state[1]
    #phi = state[2]
    rdot = state[3]
    thetadot = state[4]
    phidot = state[5]
    #tau = state[6]
    
    rs = 2 * GM / c**2  # Schwarzschild radius
    factor = 1 - rs/r
    
    taudot = np.sqrt((c**2 + rdot**2 / factor + r**2 * thetadot**2 + r**2 * np.sin(theta)**2 * phidot**2) / (c**2 * factor))
    # Geodesic equations for Schwarzschild metric
    # d²r/dλ²
    rddot = (-(GM * c**2 / r**2) * factor * taudot**2
             + (GM / (r**2 * factor)) * rdot**2
             + r * factor * thetadot**2
             + r * factor * np.sin(theta)**2 * phidot**2)
    
    # d²θ/dλ²
    thetaddot = -2 * rdot * thetadot / r + np.sin(theta) * np.cos(theta) * phidot**2
    
    # d²φ/dλ²
    phiddot = -2 * rdot * phidot / r - 2 * (np.cos(theta)/np.sin(theta)) * thetadot * phidot
    
    return np.array([rdot, thetadot, phidot, rddot, thetaddot, phiddot, taudot])

plt.close('all')

# constants -----------------------------------------------------------------
GM = 1
c = 1
rs = 2 * GM / c**2  # Schwarzschild radius

# Initial conditions
r0 = 20
theta0 = np.pi/2  # Equatorial plane
phi0 = 0
rdot0 = 0
thetadot0 = 0.01  # Small out-of-plane velocity for 3D orbit
phidot0 = np.sqrt(GM / (r0**3 - 3 * GM * r0**2 / c**2)) * 1.01  # Near-circularGM = 1
tau0 = 0
c = 1
rs = 2 * GM / c**2  # Schwarzschild radius

state0 = np.array([r0, theta0, phi0, rdot0, thetadot0, phidot0, tau0])
# constants -----------------------------------------------------------------

# constraints ---------------------------------------------------------------

dt = 0.1         #time-step, diff in time between each point
tf = 50000       #final time
       
if tf < 10000:
    nstep = 100101
else:
    nstep = int(tf/(dt) + 101) #maximum number of points. +1 so that even number is given
    
path  = np.zeros((nstep, 7)) #4D array of initial path with state filled with zeros
path[0,:] = state0           #initial value for path
time  = np.zeros(nstep)      #array of each time interval, filled with zeros

tf_split = 0.1

time[0] = 0     #where do you want time to begin?
j = 0
dt_now = dt
run_sim = True

print(f"array of size {nstep} used ")
# constraints ---------------------------------------------------------------

# main iterative process ----------------------------------------------------

while run_sim == True:
    oldstate  = path[j, :]
    newstate, dt_new, accepted = rkf45(f, oldstate, dt_now)
    time[j+1] = time[j] + dt_now
    path[j+1, :] = newstate
    dt_now = dt_new
    j += 1
    if newstate[0] < 0.1: #if radius < 0.1 units
        jfinal = j + 1      #object falls in blackhole, no need for zeros to be plotted on graphs
        print("End of simulation")
        print("Object fell into black hole")
        run_sim = False
    elif time[j] >= tf*tf_split:
        print(f" time: ({(time[j] / tf) * 100:.0f}%), step {j}, dt={dt_now:.6f}")
        tf_split += 0.1
    if time[j] >= tf or j >= nstep - 2:
        jfinal = j + 1
        run_sim = False
        print("simulation complete")
# main iterative process ----------------------------------------------------

# Extract paths + to cartesian ----------------------------------------------

rpath = path[:jfinal, 0]
thetapath = path[:jfinal, 1]
phipath = path[:jfinal, 2]
rdotpath = path[:jfinal, 3]
thetadotpath = path[:jfinal, 4]
phidotpath = path[:jfinal, 5]
taupath = path[:jfinal, 6]
time = time[:jfinal]

# Convert to Cartesian coordinates
xpath = rpath * np.sin(thetapath) * np.cos(phipath)
ypath = rpath * np.sin(thetapath) * np.sin(phipath)
zpath = rpath * np.cos(thetapath)
# Extract paths + to cartesian ----------------------------------------------

#============================================================================
#radius, phase over time

#----------------------------------------------
plt.figure('time / radius')

plt.xlabel('time')
plt.ylabel('radius')
plt.title('rate of radius')
plt.grid('on')
plt.plot(time, rpath)
#----------------------------------------------

#----------------------------------------------
plt.figure('time / phase')

plt.xlabel('time')
plt.ylabel('phase')
plt.title('rate of phase')
plt.grid('on')
plt.plot(time, thetapath)
#----------------------------------------------

#----------------------------------------------
plt.figure('time / azimuthal')

plt.xlabel('time')
plt.ylabel('azimuthal')
plt.title('rate of azimuthal')
plt.grid('on')
plt.plot(time, phipath)
#----------------------------------------------

#conserved quantities:

# Specific energy (energy per unit mass)
# E = (1 - rs/r) * c² * (dt/dτ)
# From metric: -c²(dt/dτ)²(1-rs/r) + dr²/(1-rs/r) + r²(dθ² + sin²θ dφ²) = -c²
# Solving for dt/dτ:
#dtdtau     = np.sqrt((c**2 + rdotpath**2/(1 - rs/rpath) + rpath**2 * (thetadotpath**2 + np.sin(thetapath)**2 * phidotpath**2)) / (c**2 * (1 - rs/rpath)))
taudotpath = np.sqrt((c**2 + rdotpath**2/(1 - rs/rpath) + rpath**2 * thetadotpath**2 + rpath**2 * np.sin(thetapath)**2 * phidotpath**2) / (c**2 * (1 - rs/rpath)))

Energy = (1 - rs/rpath) * c**2 * taudotpath

# Specific angular momentum (z-component, conserved in equatorial orbits)
# L_z = r² * sin²(θ) * dφ/dτ
L_z = rpath**2 * np.sin(thetapath)**2 * phidotpath

# Total angular momentum magnitude
# L² = r⁴(θ̇² + sin²θ φ̇²)
L_tot = rpath**2 * np.sqrt(thetadotpath**2 + np.sin(thetapath)**2 * phidotpath**2)

# Metric norm (should be -c² = -1 for massive particle with c=1):
#   g_μν ẋ^μ ẋ^ν = -(1-rs/r)c²ṫ² + ṙ²/(1-rs/r) + r²θ̇² + r²sin²θ φ̇²
metric_norm = (-(1 - rs/rpath) * c**2 * taudotpath**2 + rdotpath**2 / (1 - rs/rpath) + rpath**2 * thetadotpath**2 + rpath**2 * np.sin(thetapath)**2 * phidotpath**2)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

#plot and style for top plot
ax1.plot(time, L_z)
ax1.set_xlabel('time')
ax1.set_ylabel('angular momentum -z')
ax1.set_title('Angular Momentum-z over time')
ax1.axis('tight')
ax1.grid('on')
ax1.set_xlim(0, time[jfinal-2])

ax2.plot(time, L_tot)
ax2.set_xlabel('time')
ax2.set_ylabel('angular momentum ')
ax2.set_title('Angular Momentum over time')
ax2.axis('tight')
ax2.grid('on')
ax2.set_xlim(0, time[jfinal-2])

#plot and style for middleplot
ax3.plot(time, Energy)
ax3.set_xlabel('time')
ax3.set_ylabel('energy')
ax3.set_title('Energy over time')
ax3.axis('tight')
ax3.grid('on')
ax3.set_xlim(0, time[jfinal-2])


plt.tight_layout() #plots don't overlap


plt.figure('metric')

plt.plot(time, metric_norm)
plt.xlabel('time')
plt.ylabel('metric norm')
plt.title('norm over time')
plt.axis('tight')
plt.grid('on')
plt.xlim(0, time[jfinal-2])

plt.figure('proper time')

plt.plot(time, taudotpath)
plt.xlabel('time')
plt.ylabel('metric norm')
plt.title('norm over time')
plt.axis('tight')
plt.grid('on')
plt.xlim(0, time[jfinal-2])

# Plot proper time vs coordinate time
plt.figure('Proper Time vs Coordinate Time')
plt.plot(taupath, time, 'b-', linewidth=1)
plt.xlabel('Coordinate Time (t)')
plt.ylabel('Proper Time (τ)')
plt.title('Proper Time vs Coordinate Time')
plt.grid(True)

# Also plot time dilation factor
plt.figure('Time Dilation Factor')
dt_dtau = np.gradient(taupath, time)  # dt/dτ
plt.plot(time, dt_dtau, 'r-', linewidth=1)
plt.xlabel('Proper Time (τ)')
plt.ylabel('dt/dτ')
plt.title('Time Dilation Factor (dt/dτ)')
plt.grid(True)
plt.axhline(1, color='k', linestyle='--', alpha=0.5, label='No dilation')
plt.legend()

#============================================================================

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

angle = np.linspace(0, 2*np.pi, 500)

# Plot 1: x vs y
ax1.set_title('x vs y')
ax1.plot(xpath, ypath, label='Trajectory')
ax1.plot(xpath[jfinal-1], ypath[jfinal-1], 'or', label='Object')
ax1.plot(rs*np.cos(angle), rs*np.sin(angle), 'k', label='Event Horizon')
ax1.plot(0, 0, 'ok', label='Singularity')
ax1.grid(True)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.axis('equal')

# Plot 2: x vs z
ax2.set_title('x vs z')
ax2.plot(xpath, zpath, label='Trajectory')
ax2.plot(xpath[jfinal-1], zpath[jfinal-1], 'or', label='Object')
ax2.plot(rs*np.cos(angle), rs*np.sin(angle), 'k', label='Event Horizon')
ax2.plot(0, 0, 'ok', label='Singularity')
ax2.grid(True)
ax2.set_xlabel('x')
ax2.set_ylabel('z')
ax2.axis('equal')

# Plot 3: y vs z
ax3.set_title('y vs z')
ax3.plot(ypath, zpath, label='Trajectory')
ax3.plot(ypath[jfinal-1], zpath[jfinal-1], 'or', label='Object')
ax3.plot(rs*np.sin(angle), rs*np.cos(angle), 'k', label='Event Horizon')
ax3.plot(0, 0, 'ok', label='Singularity')
ax3.grid(True)
ax3.set_xlabel('y')
ax3.set_ylabel('z')
ax3.axis('equal')

# Legend in the 4th subplot
ax4.axis('off')  # turn off the empty subplot
handles, labels = ax1.get_legend_handles_labels()
ax4.legend(handles, labels, loc='center')

plt.tight_layout()
plt.show()

# ============ 3D PLOT ======================================================

fig = plt.figure('3D Schwarzschild Geodesic', figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot trajectory
ax.plot(xpath, ypath, zpath, 'b-', linewidth=0.5, label='Geodesic')
ax.scatter(xpath[-1], ypath[-1], zpath[-1], color='red', s=50, label='Final Position')

# Plot singularity (point at center)
ax.scatter(0, 0, 0, color='black', s=10, marker='o', 
           linewidths=2, label='Singularity', zorder=10)

# Plot event horizon (sphere)
u = np.linspace(0, 2 * np.pi, 100)
x_sphere = rs * np.outer(np.cos(u), np.sin(u/2))
y_sphere = rs * np.outer(np.sin(u), np.sin(u/2))
z_sphere = rs * np.outer(np.ones(np.size(u)), np.cos(u/2))

# Add wireframe for visibility of event horizon
ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='black', alpha=0.5, 
                  linewidth=1.2, rstride=5, cstride=5, label='event horizon')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Geodesic around Schwarzschild Black Hole', fontweight='bold')
ax.legend()

# Set equal aspect ratio
max_range = np.max([xpath.max() - xpath.min(), ypath.max() - ypath.min(), zpath.max() - zpath.min()]) / 2
mid_x = (xpath.max() + xpath.min()) / 2
mid_y = (ypath.max() + ypath.min()) / 2
mid_z = (zpath.max() + zpath.min()) / 2
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)
# ============ 3D PLOT ======================================================