import numpy as np
import matplotlib.pyplot as plt

# For information about the algorithm, visit:
# https://www.jhuapl.edu/SPSA/PDF-SPSA/Spall_An_Overview.PDF

# Consatants to be used for the gradient descent
constats = {"alpha": 0.602, "gamma": 0.101, "a": 0.6283185307179586, "c": 0.1, "A": False}

# Main minimising function
def SPSA(f, theta, n_iter, extra_params = False, theta_min = None, theta_max = None, 
	report = False, constats = constats, return_progress = False):
	
	# Parameters:
	# 	f: Function to be minimised (func)
	# 	theta: Initial function parameters (np.array)
	# 	n_iter: Number of iterations (int)
	# 	extra_params: Extra parameters taken by f (np.array)
	# 	theta_min: Minimum value of theta (np.array)
	# 	theta_max: Maximum value of theta (np.array)
	# 	report: Print progress. If False, nothing is printed. If int, every
	# 		report iterations you will get the iteration number, function 
	# 		value and parameter values.
	# 	constats: Constants needed for the gradient descent (dict)
	#	return_progress: Return array with all the function values at each itearion (bool)

	# Returns:
	# 	theta: Optimum parameters values to minimise f (np.array)
	#	If return_progress == True:
	#		progress: Array with all the function values at each itearion (np.array)

	# Get value of p from paramters
	p = len(theta)

	# Get constants from dictionary
	alpha = constats["alpha"]
	gamma = constats["gamma"]
	a = constats["a"]
	c = constats["c"]
	A = constats["A"]

	if A == False:
		A = n_iter / 10

	if return_progress:
		progress = np.array([])

	# Carry out the iterations
	for k in range(1, n_iter + 1):
		ak = a / (k + A)**alpha
		ck = c / k**gamma

		delta = 2 * np.round(np.random.rand(p, )) - 1

		theta_plus = theta + ck * delta
		theta_minus = theta - ck * delta

		if extra_params is False:
			y_plus = f(*theta_plus)
			y_minus = f(*theta_minus)
		else:
			y_plus = f(*theta_plus, *extra_params)
			y_minus = f(*theta_minus, *extra_params)

		# Get derivative
		g_hat = (y_plus - y_minus) / (2 * ck * delta)

		# Gradient descent step
		theta = theta - ak * g_hat

		# Make sure theta is within the boundaries
		if theta_min is not None:
			index_min = np.where(theta < theta_min)
			theta[index_min] = theta_min[index_min]

		if theta_max is not None:
			index_max = np.where(theta > theta_max)
			theta[index_max] = theta_max[index_max]

		if return_progress:
			if extra_params is False:
				progress = np.append(progress, f(*theta))
			else:
				progress = np.append(progress, f(*theta, *extra_params))

		# Report progress
		if report:
			if k % report == 0:
				if extra_params == False:
					print(f"Iteration: {k}\tArguments: {theta}\tFunction value: {f(*theta)}")
				else:
					print(f"Iteration: {k}\tArguments: {theta}\tFunction value: {f(*theta, *extra_params)}")

	# Return optimum value
	if not return_progress:
		return theta
	else:
		return theta, progress


# Plot the second return value of SPSA
def plot_progress(progress, title = False, xlabel = False, ylabel = False, save = False):
	plt.plot(progress, color = "#e100ff")
	if xlabel:
		plt.xlabel(xlabel)
	if ylabel:
		plt.ylabel(ylabel)
	if title:
		plt.title(title)
	fig_size = plt.rcParams["figure.figsize"]
	fig_size[0] = 8
	fig_size[1] = 6
	plt.rcParams["figure.figsize"] = fig_size
	plt.grid(b = True, which = "major", linestyle = "-")
	plt.minorticks_on()
	plt.grid(b = True, which = "minor", color = "#999999", linestyle = "-", alpha = 0.2)
	if save:
		plt.savefig(save)
	plt.show()

if __name__ == "__main__":
	# Test it works
	f = lambda x, y, z, w, m: x**2 + y**2 + z**2
	print(SPSA(f, np.array([2, 3, 1]), 1000, report = 5, extra_params = np.array(["Extra parameter", 
		"Another parameter"]), theta_min = np.array([0.5, 0.4, 0.3])))