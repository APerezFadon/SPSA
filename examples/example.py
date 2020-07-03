from SPSA import SPSA, plot_progress
import numpy as np

f = lambda x, y, z, w, m: x**2 + y**2 + z**2
params, progress = SPSA(f, np.array([2, 3, 1]), 1000, report = 5, extra_params = np.array(["Extra parameter", 
	"Another parameter"]), theta_min = np.array([0.5, 0.4, 0.3]), return_progress = True)

plot_progress(progress, title = "title", xlabel = r"x$^{2}$")