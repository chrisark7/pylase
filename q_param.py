""" A module for working with q parameters which describe Gaussian laser beams

The q parameter is a complex number which specifies the properties of a Gaussian laser beam at a
particular point.  For an ideal Gaussian laser beam in the fundamental mode, the q parameter
contains enough information to characterize the beam at every point in space.
"""

import warnings
import numpy as np

__author__ = "Chris Mueller"
__email__ = "chrisark7@gmail.com"
__status__ = "Development"


###################################################################################################
# qParameter Class
###################################################################################################
class qParameter(object):
    """ Used to define the complex q parameter of Gaussian beams.

    This class is designed for working with the complex-valued q parameter of Gaussian beams.
    The root object contains the complex-valued q parameter and the wavelength of the radiation
    described by that q parameter.

    :param q: The complex-valued q parameter
    :param wvlnt: The wavelength of the radiation
    :type q: complex
    :type wvlnt: float
    """
    def __init__(self, q=None, wvlnt=None):
        """ Returns an object in the qParameter class

        It is possible to initialie an empty object so that the set routines (defined below) can
        be used to more easily define the q parameter.  It is important that the units of the q
        parameter and the wavlength match, and it is generally best if both are expressed in
        meters.

        :param q: The complex-valued q parameter
        :param wvlnt: The wavelength of the radiation
        :type q: complex
        :type wvlnt: float
        :return: An instance of the qParameter class
        :rtype: qParameter
        """
        # Check types
        if q is not None:
            if not np.iscomplex(q):
                raise TypeError('q parameter should be complex')
            elif q.imag < 0:
                raise TypeError('q parameter should have a positive imaginary value')
        if wvlnt is not None:
            if wvlnt < 0:
                raise TypeError('wvlnt should be positive')
            elif wvlnt > 10e-6:
                warnings.warn('wvlnt is greater than 10e-6, double check the units')
        self.q = q
        self.wvlnt = wvlnt

    def copy(self):
        """ Returns a copy of the qParameter instance

        :return: fresh copy of qParameter
        :rtype: qParameter
        """
        other = qParameter(q=self.q, wvlnt=self.wvlnt)
        return other

    def __str__(self):
        """ Defines the string representation of the q parameter

        :return: string representation of the q parameter
        :rtype: str
        """
        q = self.get_q()
        if abs(q.real) < 1e-10:
            real_str = '{0:6.3f}'.format(q.real)
        elif np.log10(abs(q.real)) < -3:
            real_str = '{0:4.3e}'.format(q.real)
        else:
            real_str = '{0:6.3f}'.format(q.real)
        if abs(q.imag) < 1e-10:
            imag_str = '{0:6.3f}'.format(q.imag)
        elif np.log10(abs(q.imag)) < -3:
            imag_str = '{0:4.3e}'.format(q.imag)
        else:
            imag_str = '{0:6.3f}'.format(q.imag)
        return '{0} + {1}i'.format(real_str, imag_str)

    def __repr__(self):
        """ Defines the representation of the q parameter when called at the command line

        :return: string representation of the q parameter
        :rtype: str
        """
        return self.__str__()

    def print_q(self):
        """ Prints the q parameter to the screen
        """
        print(self.__str__())

    ###############################################################################################
    # get/set methods
    ###############################################################################################
    def get_q(self):
        """ Returns the q value

        :return: q parameter
        :rtype: complex
        """
        if self.q is None:
            raise AttributeError('q value has not yet been defined')
        return self.q

    def get_wvlnt(self):
        """ Returns the wvlnt

        :return: wavelength of q parameter
        :rtype: float
        """
        if self.wvlnt is None:
            raise AttributeError('wvlnt value has not yet been defined')
        return self.wvlnt

    def set_q(self, q=None, position=None, beamsize=None, rayleigh=None):
        """ Sets the q parameter

        This method is designed to be a flexible way to set the q parameter from more common
        variables.  It is important that the keword arguments be used otherwise the function will
        always assume that the first value is the q parameter.

        The function is hierarchical in the following ways
          - **q:** If q is not None, then all other arguments are ignored.  Note also that python
            will assume that the first argument is :code:`q` if keyword arguments are not used.
          - **position:** This is the position with respect to the waist.  If it is left as None,
            then it is assumed to be zero.
          - **beamsize:** This is the beam size expressed as the 1/e**2 *radius* (i.e. :math:`1w`,
            not :math:`2w`). If this is not None, then the rayleigh length is ignroed.
          - **rayleigh:** This is the rayleight range of the beam and is only used if
            :code:`beamsize` is None.

        :param q: The q parameter
        :param position: The postion with respect to the waist
        :param beamsize: The 1/e**2 beam radius
        :param rayleigh: The rayleight range of the beam
        :type q: complex
        :type position: float
        :type beamsize: float
        :type rayleigh: float
        :return: A modified qParameter instance with q redefined
        :rtype: qParameter
        """
        if q is None:
            if position is None:
                z = 0
            elif not (type(position) is float or type(position) is int):
                raise TypeError('position should be a float or int')
            else:
                z = position
            if beamsize is None:
                if rayleigh is None:
                    raise TypeError('either beamsize or rayleigh must be specified if q is None')
                elif rayleigh <= 0:
                    raise TypeError('rayleigh should be positive')
                else:
                    zr = rayleigh
            elif beamsize <= 0:
                raise TypeError('beamsize should be positive')
            else:
                wvlnt = self.get_wvlnt()
                zr = np.pi * beamsize**2/wvlnt
            q = z + 1j*zr
        if not np.iscomplex(q):
            raise TypeError('q parameter should be complex')
        elif q.imag < 0:
            raise TypeError('q parameter should have a positive imaginary value')
        self.q = q
        return self

    def set_wvlnt(self, wvlnt):
        """ Sets the wavelength of the radiation described by the q parameter.

        :param wvlnt: The wavelength of the radiation
        :type wvlnt: float
        :return: modified instance of the qParameter class
        :rtype: qParameter
        """
        if wvlnt < 0:
            raise TypeError('wvlnt should be positive')
        elif wvlnt > 10e-6:
            warnings.warn('wvlnt is greater than 10e-6, double check the units')
        self.wvlnt = wvlnt

    ###############################################################################################
    # Add/Subtract Methods
    ###############################################################################################
    def __add__(self, q2):
        copy = self.copy()
        if type(q2) is qParameter:
            if not self.get_wvlnt() == q2.get_wvlnt():
                warnings.warn('wvlnt of two qParameter instances is not equal')
            copy.q = self.get_q() + q2.get_q()
        else:
            copy.q = self.get_q() + q2
        return copy

    def __radd__(self, q2):
        copy = self.copy()
        copy.q = q2 + self.get_q()
        return copy

    def __iadd__(self, q2):
        if type(q2) is qParameter:
            if not self.get_wvlnt() == q2.get_wvlnt():
                warnings.warn('wvlnt of two qParameter instances is not equal')
            self.q = self.get_q() + q2.get_q()
        else:
            self.q = self.get_q() + q2
        return self

    def __sub__(self, q2):
        copy = self.copy()
        if type(q2) is qParameter:
            if not self.get_wvlnt() == q2.get_wvlnt():
                warnings.warn('wvlnt of two qParameter instances is not equal')
            copy.q = self.get_q() - q2.get_q()
        else:
            copy.q = self.get_q() - q2
        return copy

    def __rsub__(self, q2):
        copy = self.copy()
        copy.q = q2 - self.get_q()
        return copy

    def __isub__(self, q2):
        if type(q2) is qParameter:
            if not self.get_wvlnt() == q2.get_wvlnt():
                warnings.warn('wvlnt of two qParameter instances is not equal')
            self.q = self.get_q() - q2.get_q()
        else:
            self.q = self.get_q() - q2
        return self

    ###############################################################################################
    # Beam Property Calculations
    ###############################################################################################
    def w(self, m2=1):
        """ Calculates the beam size (not necessarily at the waist) of the q parameter.

        An M**2 value other than 1 can be taken into account.  The beam size returned is the
        1/e**2 beam size usually denoted as w.  Note that M**2 is used, not M.

        :param m2: The M**2 value of the beam
        :type m2: float > 1
        :return: The 1/e**2 beam size
        :rtype: float
        """
        # Check m2
        if m2 < 1:
            raise ValueError('m2 should be greater than or equal to 1')
        wvlnt = self.get_wvlnt()
        q = self.get_q()
        return (wvlnt/np.pi * q.imag * (1 + m2**2 * (q.real/q.imag)**2))**(1/2)

    def r(self):
        """ Calculates the radius of curvature of the field's phasefronts.
        """
        q = self.get_q()
        return q.imag * (q.real/q.imag + q.imag/q.real)

    def w0(self):
        """ Calculates the *waist size* of the beam.
        """
        wvlnt = self.get_wvlnt()
        q = self.get_q()
        return (wvlnt/np.pi * q.imag)**(1/2)

    def z(self):
        """ Calculates the position of the beam relative to the waist

         I.E. the real part of the q parameter.

        :return: position relative to waist
        :rtype: float
        """
        q = self.get_q()
        return q.real

    def zr(self):
        """ Calculates the Rayleigh range of the beam

        I.E. the imaginary part of the q parameter.

        :return: Rayleigh length of the beam
        :rtype: float
        """
        q = self.get_q()
        return q.imag

    def gouy(self):
        """ Calculates the Gouy phase, relative to the waist.
        """
        q = self.get_q()
        return np.arctan(q.real/q.imag)

    def divang(self):
        """ Calculates the divergence angle of the beam given the q parameter.

        This is the diffraction limited (M^2=1) beam divergence.
        """
        wvlnt = self.get_wvlnt()
        return wvlnt/(np.pi * self.w0())

    def reverse(self):
        """ Returns a q parameter which is identical but facing the other direction.

        This is accomplished by switching the sign of the real component.

        :return: reversed q parameter
        :rtype: qParameter
        """
        q = self.get_q()
        self.q = q - 2 * q.real
        return self

    def ovlp_field(self, q2):
        """ Calculates the field overlap coefficient with q2.

        This method calculates the overlap coefficient of two Gaussian beams defined by the two
        q parameters.  This is a complex coefficient which describes how much of one field
        overlaps with the other in the orthogonal basis.

        :param q2: The q parameter against which the overlap will be calculated
        :type q2: qParameter
        :return: complex-valued overlap coefficient
        :rtype: complex
        """
        # Check type of q2
        if type(q2) is not qParameter:
            raise TypeError('q2 should be an instance of the qParameter class')
        q1_val = self.get_q()
        q2_val = q2.get_q()
        return (4 * q1_val.imag * q2_val.imag)/abs(q1_val.conjugate() - q2_val)**2

    def ovlp_pow(self, q2):
        """ Calculates the power overlap coefficient with q2.

        This method calculates the overlap coefficient of two Gaussian beams defined by the two
        q parameters.  This is a real-valued coefficient which describes how much of the power in
        one field overlaps with the other.

        :param q2: The q parameter against which the overlap will be calculated
        :type q2: qParameter
        :return: real-valued overlap coefficient
        :rtype: float
        """
        return abs(self.ovlp_field(q2))**2

