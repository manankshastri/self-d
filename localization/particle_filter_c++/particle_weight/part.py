import numpy as np

def gauss(sig_x, sig_y, x_obs, y_obs, mu_x, mu_y):
    norm = 1 / (2 * np.pi * sig_x * sig_y)
    exponent1 = ((x_obs - mu_x)**2) / (2 * sig_x * sig_x)
    exponent2 = ((y_obs - mu_y)**2) / (2 * sig_y * sig_y)
    total_exponent = exponent1 + exponent2
    weight = norm * np.exp(-total_exponent)
    return weight

weight1 = gauss(0.3, 0.3, 6, 3, 5, 3)
weight2 = gauss(0.3, 0.3, 2, 2, 2, 1)
weight3 = gauss(0.3, 0.3, 0, 5, 2, 1)
print("Weight1: ", weight1)
print("Weight2: ", weight2)
print("Weight3: ", weight3)
print("\nTotal Weight: ", weight1*weight2*weight3)
