import math
import time

import numpy as np
import pylab as plt


def odeEuler(f, y0, t):
    '''Approximate the solution of y'=f(y,t) by Euler's method.

    Parameters
    ----------
    f : function
        Right-hand side of the differential equation y'=f(t,y), y(t_0)=y_0
    y0 : number
        Initial value y(t0)=y0 where t0 is the entry at index 0 in the array t
    t : array
        1D NumPy array of t values where we approximate y values. Time step
        at each iteration is given by t[n+1] - t[n].

    Returns
    -------
    y : 1D NumPy array
        Approximation y[n] of the solution y(t_n) computed by Euler's method.
    '''
    y = np.zeros(len(t))
    y[0] = y0
    for n in range(0, len(t) - 1):
        y[n + 1] = y[n] + f(y[n], t[n]) * (t[n + 1] - t[n])
    return y


# Finds value of y for a given x using step size h
# and initial value y0 at x0.
def rungeKutta(f, y0, t):
    # Count number of iterations using step size or
    # step height h

    # Iterate for number of iterations
    y = np.zeros(len(t))
    y[0] = y0
    for n in range(0, len(t) - 1):
        "Apply Runge Kutta Formulas to find next value of y"
        h = (t[n + 1] - t[n])
        k1 = h * f(y[n], t[n])
        k2 = h * f(y[n] + 0.5 * h, t[n] + 0.5 * k1)
        k3 = h * f(y[n] + 0.5 * h, t[n] + 0.5 * k2)
        k4 = h * f(y[n] + h, t[n] + k3)

        # Update next value of y
        y[n + 1] = y[n] + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


# comparison between euler and runger
t = np.linspace(0, 10, 200)
y0 = 1
f = lambda y, t: -y
start_euler = time.time()
y_euler = odeEuler(f, y0, t)
end_euler = time.time()
start_runger = time.time()
y_runge = rungeKutta(f, y0, t)
end_runge = time.time()
y_True = np.exp(-t)
plt.plot(t, y_euler, "b.-", t, y_runge, "r-", t, y_True, "y-")
plt.legend(["Euler", "runger", "True"])
plt.axis([0, 3, 0, 3])
plt.grid(True)
plt.title("Solution of $y' = -y , y(0) = 1$")
plt.show()
print("The run time for euler = " + str(end_euler - start_euler))
print("The run time for ruger = " + str(end_runge - start_runger))


# simulation parts

# modify runge to be able to plot the trajectory
def rungeKutta2(r0, phi0, psi0, t):
    # initialize the r plus r minus and phi arrays and fill the first entry of each one with the initial value
    r_plus = []
    r_minus = []
    phi = []
    psi = []
    r_plus.append(r0)
    r_minus.append(r0)
    phi.append(phi0)
    psi.append(psi0)
    # we first use runge Kutta on the positive value of r and get the change of r from the positive side with respect
    # to tao
    for n in range(0, len(t) - 1):
        "Apply Runge Kutta Formulas to find next value of y"
        h = (t[n + 1] - t[n])
        k1 = h * delta_r_plus(r_plus[n], t[n])
        k2 = h * delta_r_plus(r_plus[n] + 0.5 * h, t[n] + 0.5 * k1)
        k3 = h * delta_r_plus(r_plus[n] + 0.5 * h, t[n] + 0.5 * k2)
        k4 = h * delta_r_plus(r_plus[n] + h, t[n] + k3)

        # Update next value of r plus
        r_plus.append(r_plus[n] + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))
        # we apply also the runge kutta on the negative side of r to get the r that is falling into the black hole
    for n in range(0, len(t) - 1):
        "Apply Runge Kutta Formulas to find next value of y"
        h = (t[n + 1] - t[n])
        k1 = h * delta_r_minus(r_minus[n], t[n])
        k2 = h * delta_r_minus(r_minus[n] + 0.5 * h, t[n] + 0.5 * k1)
        k3 = h * delta_r_minus(r_minus[n] + 0.5 * h, t[n] + 0.5 * k2)
        k4 = h * delta_r_minus(r_minus[n] + h, t[n] + k3)
        # we set a condition if the radius is less than 2 then stop counting because the object would have already
        # passed the point of no return
        if r_minus[n] + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4) < 2:
            break
        else:
            # Update next value of y
            r_minus.append(r_minus[n] + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))
    # after we get the value of r we use it to find the value of phi also using runge Kutta however we use a loop to
    # the len of rminus because the length of this array might be less than the tao in case it passed the point of
    # no return before the end of timing sample
    for n in range(0, len(r_minus) - 1):
        "Apply Runge Kutta Formulas to find next value of y"
        h = (t[n + 1] - t[n])
        k1 = h * delta_phi(r_minus[n], t[n])
        k2 = h * delta_phi(r_minus[n] + 0.5 * h, t[n] + 0.5 * k1)
        k3 = h * delta_phi(r_minus[n] + 0.5 * h, t[n] + 0.5 * k2)
        k4 = h * delta_phi(r_minus[n] + h, t[n] + k3)

        # Update next value of y
        phi.append(phi[n] + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))
    # return the arrays that would contain the information of the trajectory of the photon
    for n in range(0, len(r_minus) - 1):
        "Apply Runge Kutta Formulas to find next value of y"
        h = (t[n + 1] - t[n])
        k1 = h * delta_psi(r_minus[n], t[n])
        k2 = h * delta_psi(r_minus[n] + 0.5 * h, t[n] + 0.5 * k1)
        k3 = h * delta_psi(r_minus[n] + 0.5 * h, t[n] + 0.5 * k2)
        k4 = h * delta_psi(r_minus[n] + h, t[n] + k3)

        # Update next value of y
        psi.append(psi[n] + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))

    return r_plus, r_minus, phi, psi


# constants of the system
M = 4
r_M = 5 * M
E_m = 8.032
L_m = 35.82 * M
phi_zero = 0
L_E = L_m / E_m
timesteps = 1000
steptime = 0.01
psi_zero = 0
c = 2.9 * 10 ** 8
# the functions of each trajectory used in Kutta
delta_phi = lambda r, t: L_m / r ** 2
delta_r_minus = lambda r, t: -(E_m ** 2 - (1 + (L_m / r) ** 2) * (1 - (2 * M) / r)) ** 0.5
delta_r_plus = lambda r, t: (E_m ** 2 - (1 + (L_m / r) ** 2) * (1 - (2 * M) / r)) ** 0.5
delta_psi = lambda r, t: (1 / r ** 2) * ((1 / L_E ** 2) - ((1 / r ** 2) * (1 - (2 * M / r)))) ** -0.5

# we start by initializing the trajectory arrays phi and r and then create the time sampling that we need as tao
tao = [0]
# fill the timing array
for i in range(1, timesteps):
    tao.append(tao[i - 1] + steptime)

# take different values of L_m to show multiple trajectories at the same time in the same pic so we do it for
# L_m =35,37,40 represented in x1,y1,x2,y2,x3,y3


# we store the trajectory values we got from rungeKutta in these variables
r_plus_runge, r_minus_runge, phi_runge, psi_runge = rungeKutta2(r_M, phi_zero, psi_zero, tao)

# we use the trajectory of rminus and phi to show the path of spaceship falling into the black hole and we do
# this by transforming the polar coordinates rminus and phi into x y coordinates using trigonometry
x1 = []
y1 = []
for i in range(len(r_minus_runge)):
    x1.append(r_minus_runge[i] * math.cos(phi_runge[i]))
    y1.append(r_minus_runge[i] * math.sin(phi_runge[i]))

L_m = 37.82 * M
r_plus_runge, r_minus_runge, phi_runge, psi_runge = rungeKutta2(r_M, phi_zero, psi_zero, tao)
# we use get the trajectory of the r minus and phi to show the path of spaceship falling into the black hole and we do
# this by transforming the polar coordinates rminus and phi into x y coordinates using trigonometry
x2 = []
y2 = []
for i in range(len(r_minus_runge)):
    x2.append(r_minus_runge[i] * math.cos(phi_runge[i]))
    y2.append(r_minus_runge[i] * math.sin(phi_runge[i]))

L_m = 40.82 * M
r_plus_runge, r_minus_runge, phi_runge, psi_runge = rungeKutta2(r_M, phi_zero, psi_zero, tao)
# we use get the trajectory of the r minus and phi to show the path of spaceship falling into the black hole and we do
# this by transforming the polar coordinates rminus and phi into x y coordinates using trigonometry
x3 = []
y3 = []
for i in range(len(r_minus_runge)):
    x3.append(r_minus_runge[i] * math.cos(phi_runge[i]))
    y3.append(r_minus_runge[i] * math.sin(phi_runge[i]))

# we then write the shape of the black hole to be drawn on the same figure and the trajectory and this black hole has a
# radius r = 2
r = 2.0
phib = [0]
# we take samples of angles with radius 2 and then transform them into polar coordinates as well and plot the black hole
for i in range(1000):
    phib.append(phib[i] + 0.36)
xb = []
yb = []
for i in range(len(phib)):
    xb.append(r * math.cos(phib[i]))
    yb.append(r * math.sin(phib[i]))
# plot the trajectory

# plot the black hole
plt.plot(x1, y1,"-r.", x2, y2,"-g.", x3, y3,"-b.", xb, yb)
plt.legend(["trajectory1", "trajectory2", "trajectory3", "Black hole"])
plt.title("Trajectory of the space ship into the black hole")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

# Now we draw the effective potential with respect to r in order to give us possible trajectories of orbits around a
# black hole which depends on the energy
v = lambda r: (1 - ((2 * M) / r)) * (1 + (L_m ** 2 / r ** 2))
r = np.linspace(2 * M, 20 * M, 1000)
v_plot = []
for i in r:
    v_plot.append(v(i))

plt.plot(r, v_plot)

plt.legend(["Veff"])
plt.title("The variation of Veff with respect to R")
plt.xlabel("R")
plt.ylabel("Veff")
plt.show()

# plot the trajectory of the photon into the black hole
plt.plot(r_minus_runge, psi_runge)
plt.plot(xb, yb)

plt.legend(["trajectory of the photon", "black hole"])
plt.title("Trajectory of the photon into the black hole")
plt.xlabel("R")
plt.ylabel("psi")
plt.show()
