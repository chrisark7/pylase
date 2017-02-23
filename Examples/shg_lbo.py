""" pylase Example: Calculate phase-matching and walk-off angles of LBO for SHG

In this pylase example we use the materials module to calculate the
phase-matching angles and the corresponding walk-off angle for second harmonic
generation in lithium triborate with a fundamental wavelength of 1064 nm.

The calculation assumes that the phase matching is done in the xy plane, at
theta=pi/2, of the crystal which is known to be the most commonly used
phase-matching solution.
"""

from numpy import pi
from scipy.optimize import minimize
from pylase import materials


# Parameters
wvlnt = 1064e-9
temp = 50

# Get indices of refraction at both wavelengths
n1x, n1y, n1z = materials.mat_sellmeier_lbo(wvlnt, temp)
n2x, n2y, n2z = materials.mat_sellmeier_lbo(wvlnt/2, temp)

# Calculate the difference between the speed of the two waves
def ind_dif(phi):
    # Calculate principle indices of refraciton
    n1f, n1s = materials.calc_principle_iors(pi/2, phi, n1x, n1y, n1z)
    n2f, n2s = materials.calc_principle_iors(pi/2, phi, n2x, n2y, n2z)
    # Return the difference in ior between the fast harmonic and slow fundamental
    return abs(n2f - n1s)*1e2

# Minimize the difference vs. phi
res = minimize(ind_dif, 0.1)
phi = res.x[0]
print("Phase matching angles at {temp:0.1f} C for {wvlnt1:0.0f}->"
      "{wvlnt2:0.0f}: theta=90, phi={phi:0.2f}".format(temp=temp,
        wvlnt1=wvlnt*1e9, wvlnt2=wvlnt/2*1e9, phi=phi*180/pi))

# Calculate the walk-off angle
wo = materials.walkoff_shg_type1(pi/2, phi, n1x, n1y, n1z, n2x, n2y, n2z)
print("Walk-off angle: {0:0.2f} mrad".format(wo*1e3))











