# %%
import numpy as np
import math
from sympy import symbols, Eq, solve, sympify
from sympy import sin, cos
import matplotlib.pyplot as plt

# Results using this program does not match the results given by numerical integrations. 
# I think the equation in the 'solve_for_acceleration_lagrangian' function might have a small error or I am testing using large angles.
#The numerical integration method agrees with experimental solutions.
#%%

"""6 SYSTEM: (with lagrangian)"""


# Uses small angle approximations
# Use the same initial conditions as the numerical integration method

def phi(j, k):
    if j == k:
        return 0
    return 1

def sigma(j, k):
    if j > k:
        return 0
    return 1

def solve_for_acceleration_lagrangian(thetas, velocities, g) -> dict:

    theta1_acceleration, theta2_acceleration, theta3_acceleration, theta4_acceleration, theta5_acceleration, theta6_acceleration = symbols('theta1_acceleration theta2_acceleration theta3_acceleration theta4_acceleration theta5_acceleration theta6_acceleration')
    accelerations = [theta1_acceleration, theta2_acceleration, theta3_acceleration, theta4_acceleration, theta5_acceleration, theta6_acceleration]
    
    all_equations = []

    # Generalization for large angles are commented out
    for j in range(1, 6+1):
        equation = 0
        for k in range(6+1):
            #equation += ((g*l*np.sin(thetas[j-1])*masses[k-1]*sigma(j,k)) + (masses[k-1]*l**2*accelerations[j-1]*sigma(j,k)))
            equation += ((g*l*thetas[j-1]*masses[k-1]*sigma(j,k)) + (masses[k-1]*l**2*accelerations[j-1]*sigma(j,k)))
            for q in range(k, 6+1):
                #equation += masses[q-1]*sigma(j,q)*l**2*np.sin(theta[j-1]-theta[k-1])*velocities[j-1]*velocities[k-1]
                equation += masses[q-1]*sigma(j,q)*l**2*theta[j-1]*(velocities[k-1]**2)
            for q in range(k, 6+1):
                #equation += masses[q-1]*sigma(j,q)*l**2*(np.sin(theta[k-1]-theta[j-1])*(velocities[j-1]-velocities[k-1])*velocities[k-1] + phi(j,k)*np.cos(theta[j-1]-theta[k-1])*accelerations[k-1])
                equation += masses[q-1]*sigma(j,q)*l**2*(-theta[k-1]*velocities[k-1]**2 + phi(j,k)*accelerations[k-1]-phi(j,k)*accelerations[k-1]*theta[j-1]**2/2-phi(j,k)*accelerations[k-1]*theta[k-1]**2/2 + theta[k-1]*theta[j-1]*phi(j,k)*accelerations[k-1])
        all_equations.append(equation)

    simplified_eq = [sympify(eq) for eq in all_equations]
    final_equations = [Eq(eq, 0) for eq in simplified_eq]
    solutions = solve(final_equations, accelerations)
    return solutions, [theta1_acceleration, theta2_acceleration, theta3_acceleration, theta4_acceleration, theta5_acceleration, theta6_acceleration]

def update_v_t_with_time_lagrangian(thetas_initial, velocities_initial, all_acceleration_lists):
    new_velocities = [0,0,0,0,0,0]
    new_thetas = [0,0,0,0,0,0]
    for i in range(6):
        #updating the velocity term
        new_velocities[i] = velocities_initial[i] + all_acceleration_lists[i] * time_step
        new_thetas[i] = thetas_initial[i] + new_velocities[i] * time_step
    return new_thetas, new_velocities

def six_node_pendulum_lagrangian(thetas, velocities):

    thetas_at_time_t = copy.deepcopy(thetas)
    velocities_at_time_t = copy.deepcopy(velocities)

    all_theta_lists = [[], [], [], [], [], []]
    all_velocity_lists = [[], [], [], [], [], []]

    #Updating the theta and velocities and then appending them to the list
    num = 1
    for t in time_span:
        print(num)
        sol_dict, acceleration_symbols = solve_for_acceleration_lagrangian(thetas_at_time_t, velocities_at_time_t, g)
        acceleration_at_time_t = []
        #appending the values to the acceleration dictionary
        for i in range(6):
            acceleration_at_time_t.append(sol_dict[acceleration_symbols[i]])

        #Getting new theta and velocities values and then appending them to the list
        thetas_at_time_t, velocities_at_time_t =  update_v_t_with_time_lagrangian(thetas_at_time_t, velocities_at_time_t, acceleration_at_time_t)
        num += 1
        for i in range(6):
            all_theta_lists[i].append(thetas_at_time_t[i])
            all_velocity_lists[i].append(velocities_at_time_t[i])
    return all_theta_lists, all_velocity_lists
# %%
