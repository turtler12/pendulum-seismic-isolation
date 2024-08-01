# %%

import numpy as np
import math
from sympy import symbols, Eq, solve, sympify
from sympy import sin, cos
import matplotlib.pyplot as plt
import copy
import random
from scipy.fftpack import fft
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Concatenate, Flatten, Reshape

# %%
def euclidean_update(theta_initials, v_initials, u, y, w, loc_force_applied=0):
    """
    Calculates the next position and velocity values based on the given initial conditions.

    Parameters:
        theta_initials (list): Initial positions of the bobs.
        v_initials (list): Initial velocities of the bobs.
        u (float): Disturbance force.
        y (ndarray): Array containing the initial positions and velocities of the bobs.
        w (float): External force.
        loc_force_applied (int, optional): Index of the bob that the external force is acting on. Defaults to 0.

    Returns:
        ndarray: Array containing the next position and velocity values after one unit of time.

    """
    #takes the initial conditions and returns the velocity and positions after 1 unit of time
    
    dt = time_step
    [M1, M2, M3, M4, M5, M6] = masses
    [K1, K2, K3, K4, K5, K6] = ks
    [g1, g2, g3, g4, g5, g6] = gammas
    #Matrix V
    V = np.array([[1-(dt*g2/M1), dt*g2/M1, 0, 0, 0, 0],
                       [dt*g2/M2, 1-dt*(g2+g3)/M2, dt*g3/M2, 0, 0, 0],
                       [0, dt*g3/M3, 1-dt*(g3+g4)/M3, dt*g4/M3, 0, 0],
                       [0, 0, dt*g4/M4, 1-dt*(g4+g5)/M4, dt*g5/M4, 0],
                       [0, 0, 0, dt*g5/M5, 1-dt*(g5+g6)/M5, dt*g6/M5],
                       [0, 0, 0, 0, dt*g6/M6, 1-dt*g6/M6]])
    X = dt * np.array([[-(K1+K2)/M1, K2/M1, 0, 0, 0, 0],
                       [K2/M2, -(K2+K3)/M2, K3/M2, 0, 0, 0],
                       [0, K3/M3, -(K3+K4)/M3, K4/M3, 0, 0],
                       [0, 0, K4/M4, -(K4+K5)/M4, K5/M4, 0],
                       [0, 0, 0, K5/M5, -(K5+K6)/M5, K6/M5],
                       [0, 0, 0, 0, K6/M6, -K6/M6]])
    identity = np.identity(6)

    A = np.block([[V, X],
                  [dt*identity, identity]])

    #loc_force_applied is the index of the bob that the external force is acting on
    B = np.array((K1*dt/M1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))

    C = np.zeros(12)
    C[loc_force_applied] = ks[loc_force_applied]*dt/masses[loc_force_applied]

    #finding the next position and velocity value using the matrices above
    return A @ y + B * u + C * w


# %%
def six_node_pendulum(thetas, velocities, parameters, loc_force_applied=0):
    """
    Simulates the motion of a six-node pendulum system over a given time span.

    Parameters:
        thetas (list): Initial positions of the bobs.
        velocities (list): Initial velocities of the bobs.
        parameters (list): List of parameters for the simulation, described by the external force
        loc_force_applied (int, optional): Index of the bob that the external force is acting on. Defaults to 0.

    Returns:
        tuple: A tuple containing two lists. The first list contains the position values of each bob at each time step, and the second list contains the velocity values of each bob at each time step.
    """
    thetas_at_time_t = copy.deepcopy(thetas)
    velocities_at_time_t = copy.deepcopy(velocities)

    all_theta_lists = [[], [], [], [], [], []]
    all_velocity_lists = [[], [], [], [], [], []]

    #Updating the theta and velocities and then appending them to the list
    for t in time_span:
        # initialize u, y, and w (u is the disturbance force, w is the external force used to nullify the disturbance)
        y = np.append(velocities_at_time_t, thetas_at_time_t).T
        u = disturbance_force(t)
        w = external_force(parameters, t)
        v1, v2, v3, v4, v5, v6, t1, t2, t3, t4, t5, t6 = euclidean_update(thetas_at_time_t, velocities_at_time_t, u, y, w, loc_force_applied)

        thetas_at_time_t = [t1, t2, t3, t4, t5, t6]
        velocities_at_time_t = [v1, v2, v3, v4, v5, v6]

        # Updated the value of the position and velocity to the list holding all the values which will be plotted
        for i in range(6):
            all_theta_lists[i].append(thetas_at_time_t[i])
            all_velocity_lists[i].append(velocities_at_time_t[i])

    return all_theta_lists, all_velocity_lists
        
def six_node_pendulum_function(thetas, velocities, f, loc_force_applied=0):
    """
    This function simulates the motion of a six-node pendulum system with an external force applied at a specific location.
    
    Parameters:
        thetas (list): List of initial positions of the pendulum nodes.
        velocities (list): List of initial velocities of the pendulum nodes.
        f (function): The external force FUNCTION to be applied to the pendulum system.
        loc_force_applied (int, optional): The index of the pendulum node where the external force is applied. Defaults to 0.
        
    Returns:
        tuple: A tuple containing two lists. The first list contains the position values of each pendulum node at each time step,
               and the second list contains the velocity values of each pendulum node at each time step.
    """
    # This function is different from the one above because it takes in a function as an input instead of the parameters of the function. 
    # Use this function unless you're doing specific testing
    # f is the function and this is used because the ML model needs to test the displacements for many different types of external functions

    thetas_at_time_t = copy.deepcopy(thetas)
    velocities_at_time_t = copy.deepcopy(velocities)

    all_theta_lists = [[], [], [], [], [], []]
    all_velocity_lists = [[], [], [], [], [], []]

    #Updating the theta and velocities and then appending them to the list
    for t in time_span:
        # initialize u, y, and w (u is the disturbance force, w is the external force used to nullify the disturbance)
        y = np.append(velocities_at_time_t, thetas_at_time_t).T
        u = disturbance_force(t)
        w = f(t)
        v1, v2, v3, v4, v5, v6, t1, t2, t3, t4, t5, t6 = euclidean_update(thetas_at_time_t, velocities_at_time_t, u, y, w, loc_force_applied)

        thetas_at_time_t = [t1, t2, t3, t4, t5, t6]
        velocities_at_time_t = [v1, v2, v3, v4, v5, v6]

        # Updated the value of the position and velocity to the list holding all the values which will be plotted
        for i in range(6):
            all_theta_lists[i].append(thetas_at_time_t[i])
            all_velocity_lists[i].append(velocities_at_time_t[i])

    return all_theta_lists, all_velocity_lists

def calculate_error(theta_withforce):
    """
    Calculates the square root of the mean of the squared errors for each time step.

    Parameters:
        theta_withforce (list): A list of displacements (OF THE BOTTOM BOB) at each time step

    Returns:
        float: The square root of the mean of the squared errors.
    """
    ##calculates the sqaure root of mean of errors ^2 for each time step
    sd_error = 0
    for i in range(len(theta_withforce)):
        sd_error += theta_withforce[i]**2
    
    sd_error = sd_error/len(theta_withforce)
    sd_error = sd_error**0.5

    return sd_error

def external_force(parameters, t):
    """
    Calculate the external applied force at a given time t based on the given parameters. 
    """
    #fourier analysis case
    if len(parameters) == 4:
        (a1, b1, w1, w2) = parameters
        return a1*np.sin(w1*t) + b1*np.cos(w2*t)
    (a1, b1, a2, b2, w1, w2) = parameters
    return a1*np.sin(w1*t) + b1*np.cos(w1*t) + a2*np.sin(w2*t) + b2*np.cos(w2*t)

def disturbance_force(t):
    """
    Simulated the seismic disturbance force. 
    """
    F_0 = 1
    w_0 = 1
    return F_0*np.sin(w_0*t)
# %%

#Define system parameters
n = 6                       #number of pendulums
g = 9.8                     #m/s^2
overall_length = 1          #length of the overall pendulum          
l = overall_length/n        #length of each pendulum
masses = [1, 1, 1, 1, 1, 2] #mass of each pendulum
gammas = [2, 2, 2, 2, 2, 2] #friction
ks = [1, 1, 1, 1, 1, 6]     #spring constant
loc_force_applied = 1       #location of the external applied force

#Initial conditions: (thetas = [theta1, theta2, theta3, theta4, theta5, theta6], velocity in m/s)
thetas = [np.radians(0), np.radians(0), np.radians(0), np.radians(0), np.radians(0), np.radians(0)]
velocities = [0, 0, 0, 0, 0, 0.03]


num_intervals = 400
end = 40
time_step = end/num_intervals
time_span = np.linspace(0,end,num_intervals)
my_parameters = (3, 3, 3, 3, 4, 4)
all_losses = []


#Testing one single external force
all_theta_lists_withforce, all_velocity_lists_withforce = six_node_pendulum(thetas, velocities, my_parameters, loc_force_applied)
all_theta_lists_noforce, all_velocity_lists_noforce = six_node_pendulum(thetas, velocities, (0, 0, 0, 0, 1, 1))

# %% Manually finding the best external force using a specific general form and a serier of for loops

def for_loop_optimization_method():
    """
    This function uses a for loop to optimize the parameters for a 6-node pendulum system. It iterates over different values of A, B, w1, and w2, and for each combination, it calculates the error by calling the `calculate_error` function. 
    It keeps track of the minimum error and the corresponding parameters. Finally, it returns the parameters with the lowest error.

    Returns:
    dict: A dictionary containing the parameters with the lowest error. The keys are 'alow', 'blow', 'w1low', and 'w2low', and the values are the corresponding minimum values.
    """

    lowest_error = 10**10
    lowest_parameters = {'alow': 0, 'blow': 0, 'w1low': 0, 'w2low': 0}

    #Testing many external forces using the for loop method
    for A in np.linspace(-1.5, 1.5, 15):
        for B in np.linspace(-1.5, 1.5, 15):
            for w1 in np.linspace(0.1, 4, 10):
                for w2 in np.linspace(0.1, 4, 10):
                    all_theta_lists_withforce, all_velocity_lists_withforce = six_node_pendulum(thetas, velocities, (A, B, w1, w2), loc_force_applied)
                    error = calculate_error(all_theta_lists_withforce[5])
                    if error < lowest_error:
                        lowest_error = error
                        lowest_parameters = {'alow': A, 'blow': B, 'w1low': w1, 'w2low': w2}
    return lowest_parameters

lowest_parameters_for_loop = for_loop_optimization_method()
                
all_theta_lists_lowest, all_velocity_lists_lowest = six_node_pendulum(thetas, velocities, (lowest_parameters_for_loop['alow'], lowest_parameters_for_loop['blow'], lowest_parameters_for_loop["w1low"], lowest_parameters_for_loop["w2low"]))


# %%
def plot_theta():
    """
    Plots the thetas of all bobs as a function of time
    """
    for i in range(6):
        plt.plot(time_span, all_theta_lists_lowest[i], label="Mass " + str(i+1))
    plt.xlabel("Time", fontsize=11)  # Add labels with fontsize
    plt.ylabel("Theta", fontsize=12)
    plt.title("Theta vs time", fontsize=16)  # Increase title fontsize
    #plt.ylim(-2*np.pi, 2*np.pi)
    plt.xlim(0, end+1)
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc="upper right", fontsize=12)
    plt.tight_layout()  # Ensures labels are not cut off
    plt.show()
    return 
plot_theta()
# %%
def plot_velocity():
    """
    Plots the velocities of all bobs as a function of time
    """
    for i in range(6):
        plt.plot(time_span, all_velocity_lists_lowest[i], label="Mass " + str(i+1))
    plt.xlabel("Time", fontsize=11)
    plt.ylabel("Velocity", fontsize=12)
    plt.title("Velocity vs time", fontsize=16)
    plt.xlim(0, end+1)
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc="upper right", fontsize=12)
    plt.tight_layout() 
    plt.show()
    return 
plot_velocity()
# %%
def plot_compare_displacement(all_theta_lists_noforce, all_theta_lists_lowest):    
    """
    Plotting the bottommost bob's displacement with and without the external force
    """
    plt.plot(time_span, all_theta_lists_noforce[5], label="Displacement without external force", color='coral', linestyle='--')
    plt.plot(time_span, all_theta_lists_lowest[5], label="Displacement with external force", color='magenta')
    plt.xlabel("Time", fontsize=11) 
    plt.ylabel("Theta", fontsize=12)
    plt.title("Displacement of lowest mass vs time", fontsize=16)  
    plt.xlim(0, end+1)
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='-', alpha=0.7)
    plt.legend(loc="upper right", fontsize=12)
    plt.tight_layout()
    plt.show()
    return
plot_compare_displacement(all_theta_lists_noforce, all_theta_lists_lowest)

#%%
def plot_ml_vs_real_displacement(all_theta_lists_ml, all_theta_lists_real):
    """
    Plotting the bottommost bob's displacement with the ml model and the real optimal force's displacement.
    Ideally, these lines should be aligned
    """
    plt.plot(time_span, all_theta_lists_ml[5], label="ML model's force", color='chocolate', linestyle=':')
    plt.plot(time_span, all_theta_lists_real[5], label="Displacement due to external force", color='royalblue', linestyle=':')
    diff_forces = [all_theta_lists_ml[5][i] - all_theta_lists_real[5][i] for i in range(len(all_theta_lists_ml[5]))]
    plt.plot(time_span, diff_forces, label="Residual displacement", color='black', linewidth=2)
    plt.xlabel("Time", fontsize=12)  # Add labels with fontsize
    plt.ylabel("Theta", fontsize=12)
    plt.title("Displacement of lowest mass vs time", fontsize=16)  # Increase title fontsize
    plt.xlim(0, end+1)
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='-', alpha=0.7)
    plt.legend(loc="upper right", fontsize=12)
    plt.tight_layout()  # Ensures labels are not cut off
    plt.show()
    return

def plot_predicted_vs_real_force(real_function_points, pred_function_points):
    """
    Plots the predicted force vs the real force as a function of time. Ideally, both of these lines should be aligned
    """
    plt.plot(time_span, real_function_points, label="Real function", color="lightsalmon")
    plt.plot(time_span, pred_function_points, label="Predicted function", color="cornflowerblue")
    plt.xlabel("Time")
    plt.ylabel("Force")
    plt.title("Predicted function vs real function")
    plt.legend()
    plt.show()
#%%

def random_fourier_coefficients(num_funcs = 10000):
    """
    Generate a list of random Fourier coefficients for a given number of functions.

    Returns:
        list: A list of lists, where each inner list contains 10 random coefficients between -10 and 10.
    """
    coeffs = []
    for _ in range(num_funcs):
        coeffs.append([random.uniform(-10, 10) for _ in range(10)])
    return coeffs

def fourier_to_function(coeffs, period = (0, end)):
    """
    Returns a function that corresponds to the given list of Fourier coefficients.
    
    Parameters:
    coeffs (list or array): The Fourier coefficients.
    period (tuple): The interval over which the function is defined.
    
    Returns:
    function: A function f(x) that approximates the original function from the Fourier coefficients.
    """
    a, b = period
    L = (b - a) / 2
    n = len(coeffs)
    
    def f(x):
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=complex)
        for k in range(n):
            result += coeffs[k] * np.exp(2j * np.pi * k * x / (2 * L))
        return result.real

    return f

def get_random_location(num_functions):
    """
    Returns a random location that the force will be applied to. 
    Location can be any integer between 0 and 4 (inclusive)
    It is better if the force is applied to the top because there will be less noise
    """
    return [random.randint(0, 4) for _ in range(num_functions)]


# ML functions:

def generate_training_data(functions_coeffs, locations, thetas, velocities):
    """
    Generates training data for a machine learning model that predicts Fourier coefficients based on the positions and velocities of a six-node pendulum.

    Args:
        functions_coeffs (List[List[float]]): A list of Fourier coefficient lists, where each coefficient list corresponds to a function.
        locations (List[int]): A list of location lists, where each location corresponds to a function.
        thetas (List[float]): A list of initial angles for each pendulum node.
        velocities (List[float]): A list of initial velocities for each pendulum node.

    Returns:
        Tuple[List[List[float]], List[List[float]], List[List[int]]]: A tuple containing three lists:
            - theta_all_functions (List[List[float]]): A list of position lists for the 6th pendulum node at all time steps, for each function
            - functions_coeffs (List[List[float]]): The input Fourier coefficient lists.
            - locations (List[int]): The input location lists.
    """

    #training set is: input(theta_all_functions, velocity_all_functions) -> output(fourier coefficients)
    #fourier_coefficients = [[f=f[0]: 10 coefficients], [f=f[1]: 10 coefficients], ...]
    #theta_all_functions = [[f=f[0]: position of the 6th bob at all 400 times], 
    #                       [f=f[1]: position of the 6th bob at all 400 times], ...]
    #locations = [f=f[0]: 1-4, f=f[1]: 1-4, ...]
    
    #converting from fourier coefficients to functions
    functions = []
    for coeff in functions_coeffs:
        functions.append(fourier_to_function(coeff))
    
    theta_all_functions = []
    velocity_all_functions = []
    i = 0
    for f in functions:
        if i%1000 == 0: print(i)
        i+=1
        all_theta_lists_withforce, all_velocity_lists_withforce = six_node_pendulum_function(thetas, velocities, f, 1)
        theta_all_functions.append(all_theta_lists_withforce[5])
        velocity_all_functions.append(all_velocity_lists_withforce[5])
    
    return theta_all_functions, functions_coeffs, locations


# %%
print("Generating training data...")
# Split dataset into training and testing sets, 80/20 split
functions_coeffs = random_fourier_coefficients(100000)
num_samples = len(functions_coeffs)
locations = get_random_location(num_samples)
displacement_dataset, coeffs_dataset, location_dataset = generate_training_data(functions_coeffs, locations, thetas, velocities)

split_index = int(0.8 * num_samples)

x_train_displacement, x_test_displacement = displacement_dataset[:split_index], displacement_dataset[split_index:]
x_train_location, x_test_location = location_dataset[:split_index], location_dataset[split_index:]
y_train, y_test = coeffs_dataset[:split_index], coeffs_dataset[split_index:]

print("Training data generated successfully!")

# %%

print("Defining the model...")
# Define the model
input_displacement = Input(shape=(400,))
input_location = Input(shape=(1,))

# Concatenate location with displacement
merged_input = Concatenate()([input_displacement, input_location])

x = Dense(128, activation='relu')(merged_input)
x = Dense(64, activation='relu')(x)
output = Dense(10)(x)

model = Model(inputs=[input_displacement, input_location], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Convert to arrays and ensure compatibility
x_train_displacement = np.array(x_train_displacement)
x_test_displacement = np.array(x_test_displacement)
x_train_location = np.array(x_train_location)
x_test_location = np.array(x_test_location)
y_train = np.array(y_train)
y_test = np.array(y_test)
print("Successfully defined and compiled the models!")

# %%
print("Training the model...")
# Train the model
history = model.fit([x_train_displacement, x_train_location], y_train, epochs=100, batch_size=32, validation_split=0.2)

print("Model trained successfully!")

#%%
# Evaluate the model
print("Evaluating the model...")
loss = model.evaluate([x_test_displacement, x_test_location], y_test)
print(f'Test loss: {loss}')


# %%
print("Testing specific values: ")
all_losses = {"0":0, "1":0, "2":0, "3":0, "4":0}
# Function to predict Fourier coefficients from displacement and location
def predict_fourier_coeffs(displacement, location):
    displacement = np.array(displacement).reshape(1, 400)  # Reshape for a single prediction
    location = np.array([location]).reshape(1, 1)          # Reshape for a single prediction
    return model.predict([displacement, location]).flatten()

# %%
# Example usage
all_example_locations = [0,1,2,3,4]
for example_location in all_example_locations:
    print("Force was applied to bob:  ", example_location+1)
    example_displacement = all_theta_lists_withforce[5]  # Use a displacement from the test set for example
    predicted_coeffs = predict_fourier_coeffs(example_displacement, example_location)
    print(f'Predicted Fourier coefficients: {np.round(predicted_coeffs, 3)} for location {example_location}')

    pred_function = fourier_to_function(predicted_coeffs)
    real_function_points = []
    pred_function_points = []
    for t in time_span:
        real_function_points.append(external_force(my_parameters, t))
        pred_function_points.append(pred_function(t))

    
    all_theta_lists_predictedforce, all_velocity_lists_predictedforce = six_node_pendulum_function(thetas, velocities, pred_function, example_location)
    diff_forces = [all_theta_lists_predictedforce[5][i] - all_theta_lists_withforce[5][i] for i in range(len(all_theta_lists_predictedforce[5]))]
    print("Loss: ", calculate_error(diff_forces))
    all_losses[str(example_location)] = calculate_error(diff_forces)


    #plot_predicted_vs_real_force(real_function_points, pred_function_points)
    #plot_compare_displacement(all_theta_lists_noforce, all_theta_lists_predictedforce)
    #plot_compare_displacement(all_theta_lists_noforce, all_theta_lists_withforce)
    plot_ml_vs_real_displacement(all_theta_lists_predictedforce, all_theta_lists_withforce)


# %%
