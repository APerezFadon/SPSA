from SPSA import SPSA

# You will need to save the SPSA folder in the same directory as this file

f = lambda x, y, z, w, m: x**2 + y**2 + z**2
print(SPSA(f, np.array([2, 3, 1]), 1000, report = 5, extra_params = np.array(["Extra parameter", 
	"Another parameter"]), theta_min = np.array([0.5, 0.4, 0.3])))