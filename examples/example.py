# Import the package
from SPSA import SPSA, plot_progress
import numpy as np

# Suppose you want to minimise x^2 + y^2 + z^2, with the conditions
# x >= 0.5
# y >= 0.4
# z >= 0.3
# Which, by inspection, will be x = 0.5, y = 0.4, z = 0.3

# Define your function. Note that you may not want to use more parameters,
# if so, put the parameters you want to change at the front
f = lambda x, y, z, w, m: x**2 + y**2 + z**2

# Carry out the algorithm. Paramters and return values are explained in SPSA.py
params, minimum, progress = SPSA(f, np.array([2, 3, 1]), 1000, report = 5, extra_params = np.array(["Extra parameter", 
	"Another parameter"]), theta_min = np.array([0.5, 0.4, 0.3]), return_progress = 5)

# Output results
print(f"The parameters that minimise the function are {params}\nThe minimum value of f is: {minimum}")

# Plot the progress. Paramters and return values are explained in SPSA.py
plot_progress(progress, title = "Plot", xlabel = r"Iteration", 
	ylabel = r"x$^{2}$ + y$^{2}$ + z$^{2}$", save = "exmaple.png")

# NOTE: To run this example, the SPSA package must be in the same directory as your script, or in a folder
# incuded in your PATH variable.