""" pylase example: Calculate Brewster angle of fused silica

This example illustrates using the materials module of pylase to calculate the
Brewster angle of fused silica (a common optical material) at a few common
wavelengths.
"""

from numpy import pi
from scipy.optimize import minimize
from pylase import materials

# Parameters
wvlnts = [1064e-9, 532e-9, 355e-9, 266e-9]


# Define a function to calculate the power reflectivity
def refl_fs(theta, n):
    # Calculate the reflectivity from the Fresnel equations
    amp_refl = materials.calc_fresnel_reflectivity(theta, 1, n)
    # Return the power reflectivity in s and p polarizations
    return [abs(x)**2 for x in amp_refl]

# Calculate Brewster's angle for the given wavelengths using scipy's minimization
b_angs = []
s_refl = []
for wvlnt in wvlnts:
    # Calculate the index of refraction of FS at this wavelength
    n = materials.mat_sellmeier_fusedsilica(wvlnt)[0]
    # Minimize the p polarization reflectivity for this wavelength
    res = minimize(lambda x: refl_fs(x, n)[1], 55*pi/180)
    # Store the result
    b_angs.append(res.x[0])
    # Calculate the s reflectivity and store it
    s_refl.append(refl_fs(res.x[0], n)[0])

# Print out Brewster's angles for the wavelengths
print("Brewster's Angles:")
print("\n".join(["{0:6.1f} nm: {1:0.3f} deg;   s refl: "
                 "{2:0.2f} %".format(x*1e9, y*180/pi, z*100)
                 for x, y, z in zip(wvlnts, b_angs, s_refl)]))
