""" pylase Example: Plot phase-matching possibilities for SFM in LBO

This example uses the materials module of pylase to plot the phase-matching
possibilities for sum-frequency mixing in LBO.  Note that SHG is just a
special case of SFM in which both wavelengths are equal.  Three plots are
produced for the three types of phase-matching possible:
  - Type I (ssf)
  - Type II (fsf)
  - Type III (sff)

"""


import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from pylase import materials

# Parameters
wvlnt1 = 1064e-9
wvlnt2 = wvlnt1/2
temp = 50
n_points = 100 # Careful, scales as O(n**2)

# Produce a grid of angles
theta = np.linspace(0, np.pi/2, n_points)
phi = np.linspace(0, np.pi/2, n_points)
thetaM, phiM = np.meshgrid(theta, phi)

# Calcualte resultant wavelength
wvlnt3 = 1/(1/wvlnt1 + 1/wvlnt2)

# Get Indices of refraction
n1x, n1y, n1z = materials.mat_sellmeier_lbo(wvlnt1, temp)
n2x, n2y, n2z = materials.mat_sellmeier_lbo(wvlnt2, temp)
n3x, n3y, n3z = materials.mat_sellmeier_lbo(wvlnt3, temp)

# Calculate delta k (in units of 1/mm)
types = ['I', 'II', 'III']
phs = dict()
for typen in types:
    phs[typen] = np.zeros(np.shape(thetaM))
for jj in range(len(theta)):
    for kk in range(len(phi)):
        # Calculate Indices of Refraction
        n1f, n1s = materials.calc_principle_iors(theta[jj], phi[kk], n1x, n1y, n1z)
        n2f, n2s = materials.calc_principle_iors(theta[jj], phi[kk], n2x, n2y, n2z)
        n3f, n3s = materials.calc_principle_iors(theta[jj], phi[kk], n3x, n3y, n3z)
        # Type I (s + s -> f)
        phs['I'][kk, jj] = 2*np.pi*(n1s/wvlnt1 + n2s/wvlnt2 - n3f/wvlnt3)*1e-3
        # Type II (f + s -> f)
        phs['II'][kk, jj] = 2*np.pi*(n1f/wvlnt1 + n2s/wvlnt2 - n3f/wvlnt3)*1e-3
        # Type III (s + f -> f)
        phs['III'][kk, jj] = 2*np.pi*(n1s/wvlnt1 + n2f/wvlnt2 - n3f/wvlnt3)*1e-3

# Log( Abs( data))
for typen in types:
    phs[typen] = np.log10(np.abs(phs[typen]))

# Colormap
mx = np.maximum(np.maximum(phs['I'], phs['II']), phs['III'])
cmap = cm.ScalarMappable(cmap='BuPu')
cmap.set_array(mx)
clrs = dict()
for type in types:
    clrs[type] = cmap.to_rgba(phs[type])

# Convert to Cartesian coords
def polar_to_cart(theta, phi, r=1):
    x = r*np.sin(theta) * np.cos(phi)
    y = r*np.sin(theta) * np.sin(phi)
    z = r*np.cos(theta)
    return x, y, z

xM, yM, zM = polar_to_cart(thetaM, phiM)

# Plot
fig = plt.figure(figsize=(16, 7))
ax = dict()
ax['I'] = plt.subplot(131, projection='3d')
ax['II'] = plt.subplot(132, projection='3d')
ax['III'] = plt.subplot(133, projection='3d')
surf = dict()
for type in types:
    surf[type] = ax[type].plot_surface(xM, yM, zM, rstride=1, cstride=1,
                                       facecolors=clrs[type], shade=False)
    # Set axis properties
    ax[type].set_xticklabels([])
    ax[type].set_yticklabels([])
    ax[type].set_zticklabels([])
    ax[type].view_init(elev=45, azim=45)

# Colorbar
plt.colorbar(cmap, shrink=1)
ax['I'].set_title('Log$_{10}(|\Delta k|)$ Type I (ssf)')
ax['II'].set_title('Log$_{10}(|\Delta k|)$ Type II (sff)')
ax['III'].set_title('Log$_{10}(|\Delta k|)$ Type III (fsf)')

# Title
plt.suptitle("SFM {0:0.1f} + {1:0.1f} -> {2:0.1f}".format(
    wvlnt1*1e9, wvlnt2*1e9, wvlnt3*1e9))

# Show figure
fig.show()
