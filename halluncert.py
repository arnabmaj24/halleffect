import numpy as np

microV =[-1.1
-0.9,
-0.7,
-1.1,
-1.6,
-2.1,
-2.5,
-1.7,
-1.2]
d = 19.69E-3
mvun = 0.5
area = 3.66E-11
width = 2.49
thick = 3.39E-08
thickuncer = thick * ((0.01/7.90)**2 + (0.01/1.49)**2)**0.5
areauncert = area * ((thickuncer/thick)**2 + (0.01/width)**2)**0.5
print(thick, thickuncer)

def EUncert(microV):
    uncert = []
    for i in range(len(microV)):
        uncert.append(((abs(microV[i])) * 10**-6)/(d) * ((0.5/abs(microV[i]))**2 + (0.02/19.69)**2)**0.5)

    return(uncert)

def Juncert(A):
    uncert = []
    for i in range(len(A)):
        uncert.append((A[i] * 10**-3)/(area) * ((0.01/A[i])**2 + (areauncert/area)**2)**0.5)

    return(uncert)

def sigma_b(x, sigma_y):
    """
    Calculate sigma_b^2 given x values and their uncertainties sigma_y.
    
    Parameters:
    x (array-like): The x values of the dataset.
    sigma_y (array-like): The uncertainties in the y values associated with each x value.
    
    Returns:
    float: The calculated value of sigma_b^2.
    """
    # Convert inputs to numpy arrays
    x = np.array(x)
    sigma_y = np.array(sigma_y)
    
    # Calculate 1 / sigma_y^2 for each element
    inv_sigma_y_squared = 1 / sigma_y**2
    
    # Calculate the terms needed for Delta
    sum_inv_sigma_y_squared = np.sum(inv_sigma_y_squared)
    sum_x_squared_over_sigma_y_squared = np.sum((x**2) * inv_sigma_y_squared)
    sum_x_over_sigma_y_squared = np.sum(x * inv_sigma_y_squared)
    
    # Calculate Delta
    delta = sum_inv_sigma_y_squared * sum_x_squared_over_sigma_y_squared - (sum_x_over_sigma_y_squared)**2
    
    # Calculate sigma_b^2
    sigma_b_squared = (1 / delta) * sum_inv_sigma_y_squared
    
    return sigma_b_squared**0.5


def weighted_average(x, sigma):
    """
    Calculate the weighted average of x values given their uncertainties (sigma).
    
    Parameters:
    x (array-like): The x values of the dataset.
    sigma (array-like): The uncertainties associated with each x value.
    
    Returns:
    float: The calculated weighted average x_wav.
    """
    # Convert inputs to numpy arrays
    x = np.array(x)
    sigma = np.array(sigma)
    
    # Calculate weights (wi) as 1 / sigma^2
    weights = 1 / sigma**2
    
    # Calculate the weighted average
    x_wav = np.sum(weights * x) / np.sum(weights)
    
    sigma_wav = 1/np.sqrt(np.sum(weights))


    return x_wav, sigma_wav

def dmu(mu, delta_L, L, delta_w, w, delta_t, t, delta_R, R):
    """
    Calculate the uncertainty in mu (Delta mu) given mu and uncertainties in L, w, t, and R.

    Parameters:
    mu (float): The value of mu.
    delta_L (float): The uncertainty in L.
    L (float): The value of L.
    delta_w (float): The uncertainty in w.
    w (float): The value of w.
    delta_t (float): The uncertainty in t.
    t (float): The value of t.
    delta_R (float): The uncertainty in R.
    R (float): The value of R.

    Returns:
    float: The calculated value of Delta mu.
    """
    # Calculate the fractional uncertainties and sum them in quadrature
    fractional_uncertainty = np.sqrt(
        (delta_L / L)**2 +
        (delta_w / w)**2 +
        (delta_t / t)**2 +
        (delta_R / R)**2
    )
    
    # Calculate Delta mu
    delta_mu = mu * fractional_uncertainty
    return delta_mu








