""" Useful routines for calculating properties of optical materials

This module defines a number of useful routines for calculating properties
of optical materials.  It contains:
  - Fresnel equations for calculating the reflectivity of an uncoated optical
    surface based on its index of refraction
  - Sellmeier equations for calculating the index of refraction as a function
    of wavelength for some common materials
  -
"""

import cmath
import warnings
from scipy.integrate import quad

__author__ = "Chris Mueller"
__status__ = "Development"


###############################################################################
# Common Calculations
###############################################################################
def calc_fresnel_reflectivity(theta, n_init, n_fin):
    """ Calculate the Fresnel reflectivity vs. angle for given iors

    This routine returns the Fresnel reflectifity for both s and p polarized
    light.  The inputs are the input angle of incidence in radians, and the
    initial and final indices of refraction

    Note that the amplitude reflectivity is returned as a possibly complex
    number, so the power reflectivity, if desired, must be calculated with
    R=|r|**2.

    :param theta: The angle of incidence of the radiation in radians
    :param n_init: The initial index of refraction
    :param n_fin: The final index of refraction
    :type theta: float
    :type n_init: float
    :type n_fin: float
    :return: (s reflection coefficient, p reflection coefficient
    :rtype: (complex, complex)
    """
    # Calculate the transmitted angle from Snell's law
    theta_t = cmath.asin(n_init/n_fin * cmath.sin(theta))
    # Check for total internal reflection
    if not theta_t.imag == 0:
        warnings.warn("Warning: The chosen values give total internal reflection")
    # Calculate the reflection coefficients
    rs = (n_init*cmath.cos(theta) - n_fin*cmath.cos(theta_t))/\
         (n_init*cmath.cos(theta) + n_fin*cmath.cos(theta_t))
    rp = (n_fin*cmath.cos(theta) - n_init*cmath.cos(theta_t))/\
         (n_fin*cmath.cos(theta) + n_init*cmath.cos(theta_t))
    # Return
    return rs, rp

###############################################################################
# Materials Properties
###############################################################################
def mat_sellmeier_lbo(wvlnt, temp):
    """ Gives the principal indices of refraction for LBO with temp dependence

    The Sellmeier equation is an empirical relationship for the indices of
    refraction of optical materials versus wavelength.  The coefficients of
    the Sellmeier equation are generally found by fitting empirical data
    measured with e.g. a spectrometer.  Note that the wavelength in the
    equation is the vacuum wavelength and not the wavelendth in the material
    itself.  For materials with differing indices of refraction along the
    different axes, the Sellmeier coefficients are measured for each axis
    independently.

    The Sellmeier coefficients for LBO are taken from data on the Castech
    website which is valid at room temperature.  There is also data, which is
    commented out currently, from [1] below.  The temperature dependence is
    taken from [2] below.  The range of validity for the temperature
    dependence if from 20-200 C.

      [1]: New nonlinear optical crystal: LiB3O5" J. Opt. Soc. Am. B. 6, 4
           (1989).
      [2]: Tang, Y., Cui, Y., & Dunn, M. H. (1995). Thermal dependence of the
           principal refractive indices of lithium triborate. Journal of the
           Optical Society of America B, 12(4), 638. doi:10.1364/JOSAB.12.000638

    :param wvlnt: The wavelength of the radiation in meters
    :param temp: The temperature of the LBO in C
    :type wvlnt: float
    :type temp: float
    :return: (nx, ny, nz)
    :rtype: (float, float, float)
    """
    # Room temp (C)
    rmTemp = 21
    # Convert wvlnt to microns
    wvlnt *= 1e6
    # Tang et. al. data
    '''
    nx = (2.4517 - 0.01177/( 0.00921  - wvlnt**2) - 0.00960 * wvlnt**2)
    ny = (2.5279 + 0.01652/( 0.005459 + wvlnt**2) - 0.01137 * wvlnt**2)
    nz = (2.5818 - 0.01414/( 0.01186  - wvlnt**2) - 0.01457 * wvlnt**2)
    '''
    # Castech Data
    nx0 = (2.454140 + 0.011249 / (wvlnt**2 - 0.011350) -
           0.014591 * wvlnt**2 - 6.60e-5 * wvlnt**4)**(1/2)
    ny0 = (2.539070 + 0.012711 / (wvlnt**2 - 0.012523) -
           0.018540 * wvlnt**2 + 2.0e-4 * wvlnt**4)**(1/2)
    nz0 = (2.586179 + 0.013099 / (wvlnt ** 2 - 0.011893) -
           0.017968 * wvlnt**2 - 2.26e-4 * wvlnt**4)**(1/2)
    # Temperature dependence (equations 19, 20, 21 in [2])
    def dnxdT(T):
        return 2.0342e-7 - 1.9697e-8 * T - 1.4415e-11 * T**2
    def dnydT(T):
        return -1.0748e-5 - 7.1034e-8 * T - 5.7387e-11 * T**2
    def dnzdT(T):
        return -8.5998e-7 - 1.5476e-7 * T + 9.4675e-10 * T**2 - 2.2375e-12 * T**3
    # Integrate the three contributions
    nx = nx0 + quad(dnxdT, rmTemp, temp)[0]
    ny = ny0 + quad(dnydT, rmTemp, temp)[0]
    nz = nz0 + quad(dnzdT, rmTemp, temp)[0]
    # Return
    return nx, ny, nz
