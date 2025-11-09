import numpy as np
import matplotlib.pyplot as plt

import matplotlib.animation as animation

def rk4(f, state, dt):
    k1 = f(state)
    k2 = f(state + 0.5 * dt * k1)
    k3 = f(state + 0.5 * dt * k2)
    k4 = f(state + dt * k3)
    return state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0

def f (state):
    r = state[0]
    #theta = state[1]
    rdot = state[2]
    thetadot = state[3]
    rddot = -GM / r**2 + r * thetadot**2
    thetaddot = -2 * rdot * thetadot / r
    return(np.array([rdot, thetadot, rddot, thetaddot]))

plt.close('all')
#main program

GM = 1
r0 = 6
theta0 = 0
rdot0 = 0
circular_orbit = np.sqrt(GM/r0**3)
thetadot0 = circular_orbit * 1.3
state0 = np.array([r0, theta0, rdot0, thetadot0])

tellNum = 10000

dt = 0.01
tf = 1000 #units of time
nstep = int(tf/dt + 1)
path = np.zeros((nstep, 4))
time = np.zeros(nstep)
path[0,:] = state0


jfinal  = nstep
time[0] = 0
for j in range(nstep-1):
    time[j+1] = time[j] + dt
    oldstate = path[j, :]
    newstate = rk4(f, oldstate, dt)
    path[j+1, :] = newstate
    
    if newstate[0] > 10 * r0:
        jfinal = j
        print("End of simulation")
        print("Object escaped orbit")
        break

    elif np.mod(j, max(1, int((nstep-1)/10))) == 0:
       print(f"{time[j]:.0f} units of time")
    elif j == nstep - 2:
       print("simulation complete")

rpath = path[:jfinal, 0]
thetapath = path[:jfinal, 1]
rdotPath = path[:jfinal, 2]
thetadotPath = path[:jfinal, 3]
time = time[:jfinal]

#----------------------------------------------
plt.figure(1)

plt.plot(time, rpath)

plt.title('rate of radius')
plt.xlabel('time')
plt.ylabel('radius')
plt.grid('on')

#----------------------------------------------

plt.figure(2)
plt.plot(time, thetapath)

plt.title('rate of phase')
plt.xlabel('time')
plt.ylabel('theta')
plt.grid('on')

#----------------------------------------------

plt.figure(3)
xpath = rpath * np.cos(thetapath)
ypath = rpath * np.sin(thetapath)


plt.plot(xpath, ypath, label='Trajectory')
plt.plot(xpath[jfinal-1], ypath[jfinal-1], 'or', label='Object')

phi = np.linspace(0, 2*np.pi, 500)

plt.plot(2*np.cos(phi), 2*np.sin(phi), 'k-', label='Event Horizon')
plt.plot(0,0, 'ok', label='Singularity')

plt.title('Motion of Object')

plt.grid('on')
plt.legend(loc='upper left')#cahnge to 'best' if needed
plt.tight_layout()
plt.axis('scaled')

#----------------------------------------------
#creates vertical graphs in single figure
fig, (ax1, ax2) = plt.subplots(2, 1)

#defines an array of ang mo and energy
AngMo_path = rpath**2 * thetadotPath
Energy = 0.5 * rdotPath**2 + 0.5 * AngMo_path**2 / rpath**2 - GM/rpath

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

fig, ax = plt.subplots(figsize=(6.5, 6.5))
# Calculate x and y paths
xpath = rpath * np.cos(thetapath)
ypath = rpath * np.sin(thetapath)
# Create plot elements
line1, = ax.plot([], [], 'b-', alpha=0.5, label='Trajectory')
point1, = ax.plot([], [], 'ro', markersize=8, label='Object')
center, = ax.plot([0], [0], 'ko', markersize=10, label='Black Hole')
# Plot reference circle
phi = np.linspace(0, 2*np.pi, 500)
ax.plot(2*np.cos(phi), 2*np.sin(phi), 'k-', alpha=0.3, label='Event Horizon')
ax.set_title('Motion of Object')
ax.legend()
ax.grid('on')
ax.axis('scaled')
# Set axis limits based on trajectory
max_extent = max(np.abs(xpath).max(), np.abs(ypath).max()) * 1.2
ax.set_xlim(-max_extent, max_extent)
ax.set_ylim(-max_extent, max_extent)

# Set the trail length (adjust this number to change how long the trail is)
trail_length = int(len(thetapath) * 2 * np.pi / thetapath[jfinal - 1] * 0.9)  #change end number depending on how many 
                                                                              #oscillation you want to show            
def init():
    line1.set_data([], [])
    point1.set_data([], [])
    return line1, point1

def animate(frame):
    # Show trajectory up to current frame
    idx = int(frame * len(xpath) / 200)
    if idx >= len(xpath):
        idx = len(xpath) - 1
    
    # Calculate start index for the trail
    start_idx = max(0, idx - trail_length)
    
    # Show only the recent trail
    line1.set_data(xpath[start_idx:idx], ypath[start_idx:idx])
    point1.set_data([xpath[idx]], [ypath[idx]])
    
    return line1, point1

anim = animation.FuncAnimation(fig, animate, init_func=init, 
                              frames=200, interval=50, blit=True, repeat=True)
plt.show()
