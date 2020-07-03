import numpy as np

# For information about the algorithm, visit:
# https://www.jhuapl.edu/SPSA/PDF-SPSA/Spall_An_Overview.PDF

# Consatants to be used for the gradient descent
constats = {"alpha": 0.602, "gamma": 0.101, "a": 0.2, "c": 0.2, "A": False}

# Main minimising function
def SPSA(f, theta, n_iter, extra_params = False, theta_min = None, theta_max = None, 
	report = False, constats = constats):
	
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

	# Returns:
	# 	theta: Optimum parameters values to minimise f (np.array)

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

	# Carry out the iterations
	for k in range(1, n_iter + 1):
		ak = a / (k + A)**alpha
		ck = c / k**gamma

		delta = 2 * np.round(np.random.rand(p, )) - 1

		theta_plus = theta + ck * delta
		theta_minus = theta - ck * delta

		if extra_params == False:
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
			theta = max(theta, theta_min)

		if theta_max is not None:
			theta = min(theta, theta_max)

		# Report progress
		if report:
			if k % report == 0:
				if extra_params == False:
					print(f"Iteration: {k}\tArguments: {theta}\tFunction value: {f(*theta)}")
				else:
					print(f"Iteration: {k}\tArguments: {theta}\tFunction value: {f(*theta, *extra_params)}")

	# Return optimum value
	return theta

if __name__ == "__main__":
	# Test it works
	f = lambda x, y, z: x**2 + y**2 + z**2
	print(SPSA(f, np.array([2, 3, -1]), 1000, report = 5))