""" Useful routines for calculating properties of optical materials

Many of the routines defined in this package are focused on calculating the
optical properties of common nonlinear optical materials.

The module defines a number of useful routines for calculating properties
of optical materials.  It contains:
  - Fresnel equations for calculating the reflectivity of an uncoated optical
    surface based on its index of refraction
  - Sellmeier equations for calculating the index of refraction as a function
    of wavelength for some common materials
  - A routine for calculating the principle indices of refraction in an
    arbitrary direction through a uniaxial or biaxial optical crystal
  - A group of routines to calculate the walk-off angle between the k-vector
    and the Poynting vector in uniaxial and biaxial optical crystals.
"""

import cmath
import warnings
from numpy import sin, cos, arccos, arctan, mean, isclose
from scipy.integrate import quad

__author__ = "Chris Mueller"
__status__ = "Development"


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


def mat_sellmeier_bbo(wvlnt):
    """ Principal indices of refraction of BBO vs. wavelength

    This function calculates the three principle indices of refraction for
    BBO from the Sellmeier equations given in [1].

      [1]: Eimerl, D., Davis, L., Velsko, S., Graham, E. K., & Zalkin, a.
           (1987). Optical, mechanical, and thermal properties of barium
           borate. Journal of Applied Physics, 62(5), 1968-1983.
           doi:10.1063/1.339536.  Retrieved from http://refractiveindex.info/

    :param wvlnt: The wavelength of the radiation in meters
    :type wvlnt: float
    :return: (nx, ny, nz)
    :rtype: (float, float, float)
    """
    # Convert wvlnt to microns
    wvlnt *= 1e6
    # Calculate the three indices
    nx = (2.3753 + 0.01224/(wvlnt**2 - 0.01667) - 0.01516 * wvlnt**2)**(1/2)
    ny = nx
    nz = (2.7359 - 0.01878/(wvlnt**2 - 0.01822) - 0.01354 * wvlnt**2)**(1/2)
    # Return
    return nx, ny, nz


def mat_sellmeier_clbo(wvlnt):
    """ Principle indices of refraction of CLBO vs. wavelength

    This function calculates the three principle indices of refraction for
    BBO from the Sellmeier equations given in [1].

      [1]: , D., Davis, L., Velsko, S., Graham, E. K., & Zalkin, a.
           (1987). Optical, mechanical, and thermal properties of barium
           borate. Journal of Applied Physics, 62(5), 1968-1983.
           doi:10.1063/1.339536.  Retrieved from http://refractiveindex.info/

    :param wvlnt: The wavelength of the radiation in meters
    :type wvlnt: float
    :return: (nx, ny, nz)
    :rtype: (float, float, float)
    """
    # Convert wvlnt to microns
    wvlnt *= 1e6
    # Calculate the three indices
    nx = (1.458830 + 0.748813/(1 - 0.013873/wvlnt**2) + 0.358461/(1 - 35.0/wvlnt**2))**(1/2)
    ny = nx
    nz = (1.422750 + 0.634640/(1 - 0.013382/wvlnt**2) + 0.170604/(1 - 35.0/wvlnt**2))**(1/2)
    # Return
    return nx, ny, nz


def mat_sellmeier_mgf2(wvlnt):
    """ Principle indices of refraction of MgF2 vs. wavelength

    This function calculates the three principle indices of refraction for
    magnesium fluoride (MgF2) from the Sellmeier equations given in [1].  Note
    that the measurement of the Sellmeier coefficients were made at 19 C.

      [1]: Dodge, M. J. (1984). Refractive properties of magnesium fluoride.
           Applied Optics, 23(12), 1980–1985. Retrieved from
           http://refractiveindex.info/

    :param wvlnt: The wavelength of the radiation in meters
    :type wvlnt: float
    :return: (nx, ny, nz)
    :rtype: (float, float, float)
    """
    # Convert wvlnt to microns
    wvlnt *= 1e6
    # Calculate the three indices
    nx = (1 + 0.48755108*wvlnt**2/(wvlnt**2 - 0.04338408)
            + 0.39875031*wvlnt**2/(wvlnt**2 - 0.09461442)
            + 2.3120358*wvlnt**2/(wvlnt**2 - 23.793604))**(1/2)
    ny = nx
    nz = (1 + 0.41344023*wvlnt**2/(wvlnt**2 - 0.03684262)
            + 0.50497499*wvlnt**2/(wvlnt**2 - 0.09076162)
            + 2.4904862*wvlnt**2/(wvlnt**2 - 23.771995))**(1/2)
    # Return
    return nx, ny, nz


def mat_sellmeier_quartz(wvlnt):
    """ Principle indices of refraction of alpha quartz vs. wavelength

    This function calculates the three principle indices of refraction for
    alpha quartz from the Sellmeier equations given in [1].

      [1]: Ghosh, G. Dispersion-equation coefficients for the refractive index
           and birefringence of calcite and quartz crystals. Opt. Commun. 163,
           95-102 (1999). Retrieved from
           http://refractiveindex.info/?shelf=main&book=SiO2&page=Ghosh-o

    :param wvlnt: The wavelength of the radiation in meters
    :type wvlnt: float
    :return: (nx, ny, nz)
    :rtype: (float, float, float)
    """
    # Convert wvlnt to microns
    wvlnt *= 1e6
    # Calculate the three indices
    nx = (1.28604141 + (1.07044083 * wvlnt**2)/(wvlnt**2 - 1.00585997e-2)
          + (1.10202242 * wvlnt**2)/(wvlnt**2 - 100))**(1/2)
    ny = nx
    nz = (1.28851804 + (1.09509924 * wvlnt**2)/(wvlnt**2 - 1.02101864e-2)
          + (1.15662475 * wvlnt**2)/(wvlnt**2 - 100))**(1/2)
    # Return
    return nx, ny, nz


def mat_sellmeier_fusedsilica(wvlnt):
    """ The principle indices of refraction for fused silica vs. wavelength

    This function calculates the three principle indices of refraction (all
    the same) for fused silica from the Sellmeier equations given in [1].

      [1]: Malitson, I. H. Interspecimen Comparison of the Refractive Index of
           Fused Silica. J. Opt. Soc. Am. 55, 1205 (1965).  Retreived from:
           http://refractiveindex.info/?shelf=main&book=SiO2&page=Malitson

    :param wvlnt: The wavelength of the radiation in meters
    :type wvlnt: float
    :return: (nx, ny, nz)
    :rtype: (float, float, float)
    """
    # Convert lambda0 to microns
    wvlnt *= 1e6
    # Calculate the index of refraction
    n = (1 + (0.6961663 * wvlnt**2)/(wvlnt**2 - 0.0684043**2) +
             (0.4079426 * wvlnt**2)/(wvlnt**2 - 0.1162414**2) +
             (0.8974794 * wvlnt**2)/(wvlnt**2 - 9.896161**2))**(1/2)
    # Return
    return n, n, n

def mat_sellmeier_caf2(wvlnt, temp):
    """ The index of refraction of calcium fluoride vs. wavelength and temp

    Calcium Fluoride is an isotropic crystal so, like fused silica, it has the
    same index of refraction for all 3 crystal axes.  However, to be
    consistent with the other sellmeier functions in this module, this
    function returns three values for the index of refraction. The equations
    for both the index of refraction and dn/dT are taken from [1].

    The base temperature for for the index of refraction is 20 C, and dn/dT is
    also from data taken at this temperature.  This measured dn/dT is
    simply multiplied by the temperature difference from the base temperature
    and summed with the index of refraction at the base temperature.  So,
    temperatures far from the base temperature have increasing amounts of error
    because the temperature dependence of dn/dT is not taken into account.

    Note that a small amount of birefringence is often observed in calcium
    fluoride due to internal stress from thermal gradients during growth of
    the crystal.  This can cause design problems in highly polarization
    dependent optical systems, but is not accounted for by this function.

      [1]: H. H. Li. Refractive index of alkaline earth halides and its
           wavelength and temperature derivatives. J. Phys. Chem. Ref. Data
           9, 161-289 (1980).  Retrieved from:
           https://refractiveindex.info/?shelf=main&book=CaF2&page=Li

    :param wvlnt: The wavelength of the radiation in meters
    :param temp: The temperature in Celcius
    :type wvlnt: float
    :type temp: float
    :return: (n, n, n)
    :rtype: (float, float, float)
    """
    # Base temperature for index measurements
    base_temp = 20.0
    # Convert wvlnt to microns
    wvlnt *= 1e6
    # Calculate room temperature index of refraction
    n0 = (1 + 0.33973 + 0.69913/(1 - (0.09374/wvlnt)**2)
          + 0.11994/(1 - (21.18/wvlnt)**2) + 4.35181/(1 - (38.46/wvlnt)**2))**(1.0/2)
    # Calculate dn/dT
    def dndT(wvlnt, n0):
        return 1/(2*n0) * (-16.6 - 57.3*(n0**2 - 1) +
               44.9*wvlnt**4/(wvlnt**2 - 0.09374**2)**2 +
               151.54*wvlnt**2/(wvlnt**2 - 38.46**2) +
               1654.6*wvlnt**4/(wvlnt**2 - 38.46**2)**2) * 1e-6
    # Integrate the temperature contribution
    n = n0 + dndT(wvlnt, n0) * (temp - base_temp)
    # Return
    return n, n, n


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
    :return: (s reflection coefficient, p reflection coefficient)
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


def calc_principle_iors(theta, phi, n1, n2, n3):
    """ Calculates the principle indices of refraction of a crystal

    This function takes in the principle indices of refraction of a uniaxial
    or biaxial optical crystal, and calculates the principle indices of
    refraction in an arbitrary direction through the crystal.  The direction
    is specified by theta and phi which are in radians.

    Theta and phis are defined in the usual way for crystalline optics; theta
    is the angle with respect to the z axis (with ior n3) and phi is the angle
    with respect to the x axis (with ior n1) in the xy plane.  Note that, as
    usual for crystalline optics, the axes are defined by n1 < n2 < n3 where
    x corresponds to n1, y to n2, and z to n3.

    :param theta: The angle with the z axis in radians
    :param phi: The angle with the x axis in radians (in the xy plane)
    :param n1: The index of refraction of the x axis (n1<n2<n3)
    :param n2: The index of refraction of the y axis (n1<n2<n3)
    :param n3: The index of refraction of the z axis (n1<n2<n3)
    :type theta: float
    :type phi: float
    :type n1: float
    :type n2: float
    :type n3: float
    :return: (smaller ior, larger ior)
    :rtype: (float, float)
    """
    # Calculate the squared components of the unit vector
    u1s = (sin(theta) * cos(phi))**2
    u2s = (sin(theta) * sin(phi))**2
    u3s = cos(theta)**2
    # Square the components of the indicatrix
    n1s, n2s, n3s = n1**2, n2**2, n3**2
    # Calculate the A, B, and C elements of the quadratic equation in n**2
    a = n1s * (u2s + u3s - 1) + n2s * (u1s + u3s - 1) + n3s * (u1s + u2s - 1)
    b = n1s * n2s * (1 - u3s) + n2s * n3s * (1 - u1s) + n3s * n1s * (1 - u2s)
    c = -1 * n1s * n2s * n3s
    # Find the two roots of the quadratic equation (and take the sqrt)
    if (b**2 - 4*a*c) < 0:
        np1 = (-b / (2*a))**(1/2)
        np2 = (-b / (2*a))**(1/2)
    else:
        np1 = (-b / (2*a) + (b**2 - 4*a*c)**(1/2) / (2*a))**(1/2)
        np2 = (-b / (2*a) - (b**2 - 4*a*c)**(1/2) / (2*a))**(1/2)
    # Return
    return np1, np2


###############################################################################
# Walk-off Calculations
###############################################################################
def walkoff_fromprop(theta, phi, n1, n2, n3):
    """ Calculate the walk-off angle between k-vector and Poynting vector

    The calculations are taken from [1].

    This function takes in the principle indices of refraction along the
    three axes as well as an arbitrary direction defined by theta and phi.
    Generally n1 < n2 < n3.  Theta and phi are referenced in the usual way;
    theta is the angle with respect to n3 and phi is the angle between the
    plane containing the n3 axis and the direction of propagation and the n1
    axis.

    The function returns two angles (in radians).  The first one corresponds
    to the polarization with the lower index of refraction and the second
    corresponds to the polarization with the higher index of refraction.  These
    indices are calculated with `calc_principle_iors`.

    Note that the walk-off angles returned by this function are generally
    not the walk-off angles of interest.  Most often one is actually interested
    in the walk-off angles between two different beams and not between the beam
    and its direction of propagation.  In this sense, this function if mostly a
    helper function for the functions which calculate the more interesting
    walk-off angles.  Also note that this function returns the angle of the
    walk-off and the direction of the walk-off is always in the plane of
    the polarization.

      [1]: Brehat, F., & Wyncke, B. (1999). Calculation of double-refraction
           walk-off angle along the phase-matching directions in non-linear
           biaxial crystals. Journal of Physics B, 22, 1891–1898.
           doi:10.1088/0953-4075/22/11/020

    :param theta: The angle with the z axis in radians
    :param phi: The angle with the x axis in radians (in the xy plane)
    :param n1: The index of refraction of the x axis (n1<n2<n3)
    :param n2: The index of refraction of the y axis (n1<n2<n3)
    :param n3: The index of refraction of the z axis (n1<n2<n3)
    :type theta: float
    :type phi: float
    :type n1: float
    :type n2: float
    :type n3: float
    :return: dictionary with ior as key and walk-off as value
    :rtype: dict
    """
    # Define a function to do the bulk of the calculation
    def wo_int(npn, s1, s2, s3, n1, n2, n3):
        val = arctan(
            ((n1**2 - npn**2) * (n2**2 - npn**2) * (n3**2 - npn**2)) *
            (s1**2 * n1**4 * (n2**2 - npn**2)**2 * (n3**2 - npn**2)**2 +
             s2**2 * n2**4 * (n1**2 - npn**2)**2 * (n3**2 - npn**2)**2 +
             s3**2 * n3**4 * (n1**2 - npn**2)**2 * (n2**2 - npn**2)**2)**(-1/2))
        return val
    # Calculate the components of the unit vector
    u1s = sin(theta) * cos(phi)
    u2s = sin(theta) * sin(phi)
    u3s = cos(theta)
    # Calculate the principle indices of refraction
    nps = calc_principle_iors(theta, phi, n1, n2, n3)
    # Ensure that np1 is greater than np2
    nps = list(nps)
    nps.sort()
    # Calculate the two angles
    ang = dict()
    for indp in nps:
        bool_val = (isclose(indp, n1, rtol=1e-10, atol=1e-10) or
                    isclose(indp, n2, rtol=1e-10, atol=1e-10) or
                    isclose(indp, n3, rtol=1e-10, atol=1e-10))
        if bool_val:
            ang[indp] = 0
        else:
            ang[indp] = wo_int(indp, u1s, u2s, u3s, n1, n2, n3)
    # Return
    return ang[nps[0]], ang[nps[1]]


def walkoff_shg_type1(theta, phi, n1x, n1y, n1z, n2x, n2y, n2z):
    """ Calculate the walk-off angle for Type I SHG

    This function calculates the walk-off angle of interest in type 1 second
    harmonic generation.  This is the walk-off angle between the higher index
    of refraction for the fundamental and the lower index of refraction for the
    second harmonic.  The calculation relies on `walkoff_fromprop` and is taken
    from [1].

    The function also checks that the two indices of refraction are near
    each other and prints a warning to the user if not.

      [1]: Brehat, F., & Wyncke, B. (1999). Calculation of double-refraction
           walk-off angle along the phase-matching directions in non-linear
           biaxial crystals. Journal of Physics B, 22, 1891–1898.
           doi:10.1088/0953-4075/22/11/020

    :param theta: The angle with the z axis in radians
    :param phi: The angle with the x axis in radians (in the xy plane)
    :param n1x: The index of refraction of the x axis, fundamental
    :param n1y: The index of refraction of the y axis, fundamental
    :param n1z: The index of refraction of the z axis, fundamental
    :param n2x: The index of refraction of the x axis, second harmonic
    :param n2y: The index of refraction of the y axis, second harmonic
    :param n2z: The index of refraction of the z axis, second harmonic
    :type theta: float
    :type phi: float
    :type n1x: float
    :type n1y: float
    :type n1z: float
    :type n2x: float
    :type n2y: float
    :type n2z: float
    :return: The walk-off angle between the two beams in radians
    :rtype: float
    """
    # Get the principle indices of refraction
    np11, np12 = calc_principle_iors( theta, phi, n1x, n1y, n1z)
    np21, np22 = calc_principle_iors( theta, phi, n2x, n2y, n2z)
    np1 = max([np11, np12])
    np2 = min([np21, np22])
    # Check that the upper of the fundamental and the lower of the harmonic are
    # within 1e-3 of each other
    dif = abs(np1 - np2)
    if dif > 1e-3:
        warnings.warn("Warning: fundamental ({0:0.4f}) and harmonic ({1:0.4f}) "
                      "indices disagree by {2:0.4f}".format(np1, np2, dif))
    # Calculate the individual walk-off angles
    unused, ang1 = walkoff_fromprop(theta, phi, n1x, n1y, n1z)
    ang2, unused = walkoff_fromprop(theta, phi, n2x, n2y, n2z)
    # Calculate the differential walk-off angle
    ang = arccos(cos(ang1) * cos(ang2))
    # Return
    return ang


def walkoff_shg_type2(theta, phi, n1x, n1y, n1z, n2x, n2y, n2z):
    """ Calculate the walk-off angle for Type I SHG

    This function calculates the walk-off angle of interest in type 2 second
    harmonic generation.  This is the walk-off angle between the average index
    of refraction for the fundamental and the lower index of refraction for the
    second harmonic.    The calculation relies on `walkoff_fromprop` and is taken
    from [1].

    The function also checks that the two indices of refraction are near
    each other and prints a warning to the user if not.

      [1]: Brehat, F., & Wyncke, B. (1999). Calculation of double-refraction
           walk-off angle along the phase-matching directions in non-linear
           biaxial crystals. Journal of Physics B, 22, 1891–1898.
           doi:10.1088/0953-4075/22/11/020

    :param theta: The angle with the z axis in radians
    :param phi: The angle with the x axis in radians (in the xy plane)
    :param n1x: The index of refraction of the x axis, fundamental
    :param n1y: The index of refraction of the y axis, fundamental
    :param n1z: The index of refraction of the z axis, fundamental
    :param n2x: The index of refraction of the x axis, second harmonic
    :param n2y: The index of refraction of the y axis, second harmonic
    :param n2z: The index of refraction of the z axis, second harmonic
    :type theta: float
    :type phi: float
    :type n1x: float
    :type n1y: float
    :type n1z: float
    :type n2x: float
    :type n2y: float
    :type n2z: float
    :return: The walk-off angle between the two beams in radians
    :rtype: float
    """
    # Get the principle indices of refraction
    np11, np12 = calc_principle_iors( theta, phi, n1x, n1y, n1z)
    np21, np22 = calc_principle_iors( theta, phi, n2x, n2y, n2z)
    np1 = mean([np11, np12])
    np2 = min([np21, np22])
    # Check that the upper of the fundamental and the lower of the harmonic are
    # within 1e-3 of each other
    dif = abs(np1 - np2)
    if dif > 1e-3:
        warnings.warn("Warning: fundamental ({0:0.4f}) and harmonic ({1:0.4f}) "
                      "indices disagree by {2:0.4f}".format(np1, np2, dif))
    # Calculate the individual walkoff angles
    unused, ang1 = walkoff_fromprop(theta, phi, n1x, n1y, n1z)
    ang2, unused = walkoff_fromprop(theta, phi, n2x, n2y, n2z)
    # Calculate the differential walkoff angle
    ang = arccos(cos(ang1) * cos(ang2))
    # Return
    return ang