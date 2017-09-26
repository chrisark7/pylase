__author__ = 'cmueller'

import scipy.fftpack as spfft
import scipy.misc
import scipy.special
import numpy as np
import GaussianBeams.gaussBeamFunctions as gb
import warnings

def propagate(e0, d, scale=1, wvlnt=266e-9):
    """ Propagates the field defined in e0 by a distance d

    This method is used to numerically propagate an electric field defined by :code:`e0`, with
    coordinate matrices defined by :code:`x` and :code:`y` by a distance :code:`d`.

    :param e0: A 2d complex array defining the electic field
    :param d: The distance to propagate the field
    :param scale: The physical size of each element of e0
    :param wvlnt: The wavelength of the radiation represented by e0
    :type e0: 2d ndarray
    :type d: float
    :type scale: float
    :type wvlnt: float
    :return: The electric field at the new location
    :rtype: 2d ndarray
    """
    # Size and wavenumber
    Ny, Nx = e0.shape
    k0 = 2*np.pi/wvlnt
    # Create phase propagator
    kvec_y = 2*np.pi * spfft.fftfreq(Ny, scale)
    kvec_x = 2*np.pi * spfft.fftfreq(Nx, scale)
    phs = np.outer(np.exp(1j/(2*k0) * d * kvec_y**2), np.exp(1j/(2*k0) * d * kvec_x**2))
    # Propagate and return
    return spfft.ifft2(spfft.fft2(e0) * phs)

def field_gaussian(q, size, wvlnt, nPoints=(1024, 1024), mode=(0, 0)):
    """ Returns a 2d array representing the Gaussian field described by the given q parameter.

    :param q: The q parameter which defines the Gaussian beam
    :param size: The physical size of the grid (in the same units as q)
    :param wvlnt: The wavelength of the radiation
    :param nPoints: The number of points of the grid
    :param mode: Describes the mode in the Hermite-Gauss basis
    :type q: complex
    :type size: 2 element tuple or int
    :type wvlnt: float
    :type nPoints: 2 element tuple of int
    :type mode: 2 element tuple
    :return: 2d array representing the electric field of the
             Gaussain beam as well as two arrays with the x and y coordinates
    :rtype: (2d complex ndarray, 2d real ndarray, 2d real ndarray)
    """
    # Parse inputs
    if hasattr(size, '__len__'):
        if len(size) == 2:
            dx, dy = size
        else:
            raise TypeError('size should be a two element tuple or an int')
    elif type(size) is float or type(size) is int:
        dx, dy = size, size
    else:
        raise TypeError('size should be a two element tuple or an int')
    if hasattr(nPoints, '__len__'):
        if len(nPoints) == 2:
            sx, sy = nPoints
        else:
            raise TypeError('nPoints should be a two element tuple or an int')
    elif type(nPoints) is int:
        sx, sy = nPoints, nPoints
    else:
        raise TypeError('nPoints should be a two element tuple or an int')
    n, m = mode
    # Create the physical grid
    x_list = np.linspace(-dx/2, dx/2, sx)
    y_list = np.linspace(-dy/2, dy/2, sy)
    x, y = np.meshgrid(x_list, y_list)
    # Calculate some parameters
    q0 = 1j*np.imag(q)
    w0 = gb.qOmega0(q, wvlnt)
    w = gb.qOmega(q, wvlnt)
    k = 2*np.pi/wvlnt
    x_herm = np.polynomial.hermite.hermval(2**(1/2)*x_list/w, [0 for x in range(n-1)] + [1])
    y_herm = np.polynomial.hermite.hermval(2**(1/2)*y_list/w, [0 for x in range(m-1)] + [1])
    n_fac = scipy.misc.factorial(n)
    m_fac = scipy.misc.factorial(m)
    # Create the field
    x_field = (2/np.pi)**(1/4) * (1/(2**n * n_fac * w0))**(1/2) * \
              (q0/np.conjugate(q0)*np.conjugate(q)/q)**(n/2) * x_herm * \
              np.exp(-1j*k/2 * x_list**2/q)
    y_field = (2/np.pi)**(1/4) * (1/(2**m * m_fac * w0))**(1/2) * \
              (q0/np.conjugate(q0) * np.conjugate(q)/q)**(m/2) * y_herm * \
              np.exp(-1j*k/2 * y_list**2/q)
    e0 = np.outer(y_field, x_field)
    return e0, x, y

def phase_mirror(radius, x, y, wvlnt):
    """ Returns the phase transformation picked up at a mirror.

    If the beam incident on the mirror is given by e0, and the phase returned from this function
    is :code:`phase = phase_mirror(radius, size)`, then the beam reflected from the mirror is
    :code:`ef = e0 * phs`.

    :param radius: The radius of curvature of the mirror
    :param x: The x coordinate matrix which describes the x coordinates of the beam
    :param y: The y-coordinate matrix which describes the y coordinates of the beam.
    :type radius: float
    :type x: 2d ndarray
    :type y: 2d ndarray
    :return: The phase picked up by the beam in reflection from the mirror.
    :rtype: 2d ndarray (same size as x and y)
    """
    # Check things
    x, y = np.array(x), np.array(y)
    if not ((type(radius) is int) or (type(radius) is float)):
        raise TypeError('radius should be a scalar')
    if (len(x.shape) is not 2) or (len(y.shape) is not 2):
        raise TypeError('x and y should be 2d arrays')
    if not x.shape == y.shape:
        raise TypeError('x and y should be the same size')
    # Calculate r
    r = (x**2 + y**2)**(1/2)
    if np.any(r > abs(radius)):
        warnings.warn('Maximum grid coordinate is larger than the mirror radius of curvature, ' +
                      'these will be set to zero.')
        r[r > abs(radius)   ] = 0
    # Calculate height and phase
    height = 2 * (radius - (radius**2 - r**2)**(1/2))
    phs = np.exp(1j * 2*np.pi/wvlnt * height)
    return phs







