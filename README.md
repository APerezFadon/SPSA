# Simultaneous Perturbation Stochastic Approximation (SPSA)
Python implementation of the SPSA algorithm. This is a minimisation algorithm based on gradient descent. The advantage of SPSA is that the complexity does not scale too much with number of parameters.

# Documentation
SPSA(f, theta, n_iter, extra_params = False, theta_min = None, theta_max = None, report = False, constats = constats, return_progress = False)
- Parameters:
  - f: Function to be minimised (func)
  - theta: Initial parameter guess (np.array)
  - n_iter: Number of iterations (int)
  - extra_params: Extra parameters taken by f (np.array)
  - theta_min: Minimum value of theta (np.array)
  - theta_max: Maximum value of theta (np.array)
  - report: Print progress. If False, nothing is printed. If n (int), every n iterations print the iteration number, function value and parameter values (bool / int)
  - constats: Constants needed for the gradient descent (dict). Default is {"alpha": 0.602, "gamma": 0.101, "a": 0.6283185307179586, "c": 0.1, "A": False}
  - return_progress: If False, nothing is else is returned. If n (int), return the iteration number, increasing by n, and the function value at each iteration (bool / int)
- Returns:
  - theta: Optimum parameters values to minimise f (np.array)
  - f(theta): Minimum value found (float)
  - If return_progress == True:
    - progress: Array with all the function values at each return_progress iteration (np.array)
- Carries out the SPSA algorithm
  
plot_progress(progress, title = False, xlabel = False, ylabel = False, save = False)
- Parameters:
  - progress: Third output from SPSA (np.array)
  - title: Graph title (str)
  - xlabel: Label for the x axis. Use r"$$" for latex formatting (str)
  - ylabel: Label for the y axis. Use r"$$" for latex formatting (str)
  - save: If not False, save the graph with the name given (bool / str)
- Plots the function value against iteration number

# References
Spall, J. C. An Overview of the Simultaneous Perturbation Method
for Efficient Optimization. Johns Hopkins APL Technical Digest. 1998; 4 (19): 482-492.
