
# %%
import matplotlib.pyplot as plt
import numpy as np
import math

# %%
g = 9.8                             #acceleration due to gravity
l = 0.1                             #length of pendulum
f_0 = 1/(2*math.pi)*math.sqrt(g/l)  #frequency of oscillation of a simple pendulum (not damped)
m = 1                               #mass of pendulum
limit = 500                         #limits the transfer function (as it asymptotically reached infinity) for plotting purposes
number_of_pendulums = 6

# Created an arbitrary list of natural frequencies of pendulums
nat_freqs = []
for i in range(10):
    nat_freqs.append(f_0*(i+1))


#finding the TF with both methods (ideal and lagrangian) for only 1 input frequency
def TF_ideal(f_input):
    """
    Calculate the Transfer Function (TF) for a simple pendulum with only 1 input frequency using the solution of the differential

    Parameters:
        f_input (float): The input frequency.

    Notes:
        - The TF is calculated as the square of the ratio of f_0^2 to the difference between f_input^2 and f_0^2.
        - If the calculated TF exceeds the limit, the limit is returned.
    """
    try:
        if math.pow(f_0**2/(f_input**2-f_0**2), 2) > limit:
            return limit
        return math.pow(f_0**2/(f_input**2-f_0**2), 2)
    except:
        return limit
def lagrangian_output(f_input):
    """
    Calculate the transfer function for a simple pendulum using the Lagrangian equation.

    Returns:
        float: The output frequency calculated using the Lagrangian equation. If an error occurs during the calculation, a large value (10^10) is returned.
     """
    try:
        return math.sqrt(g/(l)-f_input**2)
    except:
        return 10**10
    

# Finds the transfer function from the differential and lagrangian method in the frequency domain
f_list = np.linspace(0, 100, 1000)
TF_ideal_list = [TF_ideal(f_input) for f_input in f_list]
lagrangian_list = [lagrangian_output(f_input) for f_input in f_list]


# %%
#testing specific values
f_input = 0.01
print("f_input =" + str(f_input))
print("Output frequency (without air resistance, diff)= " + str(TF_ideal(f_input)))

omega_output = math.sqrt(g/(l-f_input**2))
print("Using the lagrangian: " + str(omega_output))
# %%
plt.plot(f_list, TF_ideal_list)
plt.plot(f_list, lagrangian_list)
plt.ylim(0, limit+20)
plt.xscale('log')
plt.title("Plot of TF with respect to frequency")
plt.xlabel("frequency")
plt.ylabel("Output/Input")
plt.show()
# %%

def ideal_tf_func(x):
    lis = []
    for f_input in x:
        output = 1
        for num_pend in range(number_of_pendulums):
            output *= nat_freqs[num_pend]**2/abs(nat_freqs[num_pend]**2 - f_input**2)
        lis.append(output)
    return lis

def lag_func(x):
    #only works with small oscillations
    lis = []
    for f_input in x:
        try:
            lis.append(math.sqrt((g/l)-f_input**2))
        except:
            lis.append(0)
    return lis

def with_air(x, gamma=1):
    """
    Calculates the transfer function with air resistance, given by the gamma constant

    Args:
        x (list): A list of input frequencies.
        gamma: The air resistance coefficient. Defaults to 1.

    Returns:
        list: A list of ratios of output/input corresponding to the input frequencies.
    """
    #assuming Q_0 = 1 or plotting the transfer function
    lis = []
    for f_input in x:
        output = 1
        for num_pend in range(number_of_pendulums):
            output *= nat_freqs[num_pend]**2/(math.sqrt(math.pow(nat_freqs[num_pend]**2 - f_input**2, 2) + math.pow(gamma*f_input/m, 2)))
        lis.append(output)
    return lis


# Apply the function to generate y coordinates

ideal_y = ideal_tf_func(f_list)
lag_y = lag_func(f_list)
air_y = with_air(f_list)
print("lagrangian version assumes small oscillations")
print("Number of oscillators: " + str(number_of_pendulums))

# Create the plot
plt.figure()
plt.plot(f_list, ideal_y, label='TF', color='blue', linestyle='-')
#plt.plot(f_list, lag_y, label='Lagrangian version', color='red', linestyle='--')
plt.plot(f_list, air_y, label='With air resistance', color='green', linestyle=':')

# Customizing
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(np.arange(0, 11, 1))
plt.yticks(np.arange(-1, 1.5, 0.5))
plt.title('Plot of TF with respect to frequency', fontsize=16)
plt.xlabel('frequency', fontsize=14)
plt.ylabel('Output/Input', fontsize=14)
plt.yscale('log')
plt.xscale('log')
plt.ylim(0.0001, 500)
plt.xlim(0.01, 110)
plt.legend()
plt.axhline(y=1, color='k', linewidth=0.6)
plt.axvline(x=1, color='k', linewidth=0.6)  
plt.show()

# %%
# Plot of TF with a variety of Gammas
gammas = [0, 0.1, 0.5,1, 5, 10, 100]

plt.figure()
for gamma in gammas:
    air_y = with_air(f_list, gamma)
    plt.plot(f_list, air_y, label="Gamma = " + str(gamma))


plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(np.arange(0, 11, 1))
plt.yticks(np.arange(-1, 1.5, 0.5))
plt.title('Plot of TF wrt frequency for different Gammas', fontsize=16)
plt.xlabel('Frequency [Hz]', fontsize=14)
plt.ylabel('Output/Input', fontsize=14)
plt.yscale('log')
plt.xscale('log')
plt.ylim(0.0001, 500)
plt.xlim(0.01, 110)
plt.legend()
plt.axhline(y=1, color='k', linewidth=0.6)
plt.axvline(x=1, color='k', linewidth=0.6)  
plt.show()

# %%
