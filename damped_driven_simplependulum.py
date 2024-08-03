# %%
import math
import matplotlib.pyplot as plt
import numpy as np

#%%
def f(time):
    """
    Returns the external force as a function of time.

    Parameters:
        time (float): The time at which to calculate the force.
    """
    #returns external force as a function of time
    f_0 = 40
    w = 1
    return f_0*math.cos(w*time)
# %%
# Initial and system conditions
x = 0
v = 0
k = 1
b = 0.01
m = 5
q_0 = 1
w=1

tlist = np.linspace(0, 100, 1000)
vlist = []
xlist = []
alist = []
dt = tlist[1] - tlist[0]


for t in tlist:
    """
    Uses basic numerical integration to find the displacement, velocity, and acceleraiton of a simple damped+driven pendulum
    The same technique will be used for a complex system
    """
    a = -(k/m)*x-(b/m)*v+f(t)+(k/m)*q_0*math.cos(w*t)
    v = v + a*dt
    x = x + v*dt
    vlist.append(v)
    xlist.append(x)
    alist.append(a)

# %%
#x vs time
plt.plot(tlist, xlist, color="pink")
plt.xlabel("time")
plt.ylabel("displacement")
plt.title("displacement vs time")
plt.grid(True)
plt.xlim(0, 100)
plt.show()

# %%
#v vs time
plt.plot(tlist, vlist, color="red")
plt.xlabel("Time")
plt.ylabel("Velocity")
plt.title("Velocity vs Time")
plt.grid(True)
plt.xlim(0, 100)
plt.show()

# %%
#a vs time
plt.plot(tlist, alist, color="blue")
plt.xlabel("Time")
plt.ylabel("Acceleration")
plt.title("Acceleration vs Time")
plt.grid(True)
plt.xlim(0, 100)
plt.show()
# %%
