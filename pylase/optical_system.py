""" Work with ray matrices and q parameters to calculate properties of an optical system

The Gaussian beam q parameter, encapsulated in the qParameter class, completely defines a Gaussian
laser beam in the paraxial limit.  Similiarly, a system of geometrical optical ray matrices,
encapsulated in the RayMatrix class, completely defines a series of optical system in the paraxial
limit.  This module brings the two of those togeter to allow the user to calculate properties of
an optical system consisting of a number of optical elements and a Gaussian laser beam.
"""

import warnings
from difflib import SequenceMatcher
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pylase import q_param, ray_matrix
from scipy.optimize import minimize

__author__ = "Chris Mueller"
__email__ = "chrisark7@gmail.com"
__status__ = "Development"


###################################################################################################
# opticalSystem Class
###################################################################################################
class OpticalSystem:
    """ A class for working with optical systems consisting of ray matrices and q parameters

    Internally, the class contains a RayMatrix object, and a qParameter object as well as a
    specification of the location of the qParameter.
    """
    def __init__(self):
        """ The constructor for the OpticalSystem class.

        The three inputs necessary to define an OpticalSystem object are an object of the
        RayMatrix class, an object of the qParameter class, and an location within the system
        specifying where the qParameter is defined.  These are built up in the OpticalSytem using
        the class methods rather than being passed to the constructor.  This method simply
        initializes an empty object.

        The elements component of the class contains the information needed to build the
        RayMatrix object.  It consists of a list of tuples where each tuple contains three
        elements:
          elements[0][0]: string specifying type of element
          elements[0][1]: tuple containing the parameters required to describe the element
          elements[0][2]: position of each element along the optical axis
          elements[0][3]: string label for each element

        The beams component of the class contains a list of tuples.  Each tuple contains three
        entries:
          beams[0][0]: q_parameter.qParameter object specifying beam
          beams[0][1]: location of beam along the optical axis
          beams[0][2]: string label for each beam

        :return: An instance of the OpticalSystem class
        :rtype: OpticalSystem
        """
        # Initialize RayMatrix to None
        self.rm = None
        # Initialize the beams component to an empty list and initialize its hash
        self.beams = []
        self._beam_hash = hash(tuple(self.beams))
        # Initialize the elements list and the elements hash
        self.elements = []
        self._el_hash = hash(tuple(self.elements))
        # Initialize a list for the q parameters
        self.all_qs = []
        # Add a dummy element until another element is added
        self._add_element(('interface_flat', (1, 1), 0, 'empty_system'), initial=True)

    ###############################################################################################
    # Overloading
    ###############################################################################################
    def __bool__(self):
        """ Returns a boolean value

        :return: False if no elements have been added yet or True otherwise
        :rtype: bool
        """
        return bool(self.elements)

    ###############################################################################################
    # Internal get/set/add/remove methods
    ###############################################################################################
    def _get_beam(self, beam_label):
        """ Returns the beam specified by beam_label

        beam_label should be the string specifying the beam, but an integer index is also accepted
        (though discouraged).

        :param beam_label: The string label associated with the beam (an index is also accepted)
        :type beam_label: str or int
        :return: The internal beam definition: (qParameter, z, label)
        :rtype: (q_param.qParameter, float, str)
        """
        # Check if any beams are defined
        if not self.beams:
            raise LookupError('no beams defined yet')
        # Get labels
        labels = [bm[2] for bm in self.beams]
        # Try by index if beam_label is integer
        if type(beam_label) is int:
            try:
                beam = self.beams[beam_label]
            except IndexError:
                warnings.warn('Unable to lookup beam by index, trying by beam_label')
                beam_label = str(beam_label)
                if beam_label not in labels:
                    raise ValueError('beam_label does not specify a beam')
                else:
                    beam = self.beams[labels.index(beam_label)]
        # Otherwise search by beam_label
        else:
            # If beam_label is any other type, convert it to str
            if type(beam_label) is not str:
                try:
                    beam_label = str(beam_label)
                except:
                    raise TypeError('beam_label should be a string')
            if beam_label not in labels:
                raise ValueError('beam_label does not specify a beam')
            else:
                beam = self.beams[labels.index(beam_label)]
        # Return beam
        return beam

    def _add_beam(self, beam):
        """ Adds a beam to the beams attribute

        :param beam: (qParameter object describing the beam, position of the beam along the axis, label)
        :type beam: (q_param.qParameter, float, str)
        """
        # Check types
        if type(beam) is not tuple:
            try:
                beam = tuple(beam)
            except:
                raise TypeError('beam should be a tuple with a qParameter and a float')
        if not len(beam) == 3:
            raise ValueError('beam should be a three-element tuple')
        if type(beam[0]) is not q_param.qParameter:
            raise TypeError('first element of beam should be of type qParameter')
        if type(beam[1]) is not float:
            try:
                beam = (beam[0], float(beam[1]), beam[2])
            except:
                raise TypeError('second element of beam should be a float')
        if type(beam[2]) is not str:
            try:
                beam = (beam[0], beam[1], str(beam[2]))
            except:
                raise TypeError('third element of beam should be a string')
        # Check that label is not repeated
        if beam[2] in (bm[2] for bm in self.beams):
            raise ValueError('label is already used for another beam')
        # Add beam to self.beams
        self.beams.append(beam)
        # Update
        self._update()

    def _remove_beam(self, beam_label):
        """ Removes a beam from the beams attribute

        :param beam_label: The label associated with the beam
        :type beam_label: str
        """
        # Find the index from the label
        ind = self._get_beam_index(beam_label)
        # Remove the element
        del self.beams[ind]
        # Update
        self._update()

    def _get_q(self, beam_label, pos_num):
        """ Returns the q parameter of the specified beam at the specified position

        :param beam_label: The string label associated with the beam (an index is also accepted)
        :param pos_num:
        :type beam_label: str or int
        :type pos_num: int
        :return: (q parameter, index of refraction)
        :rtype: (q_param.qParameter, float)
        """
        # Check if any beams are defined
        if not self.beams:
            raise LookupError('no beams defined yet')
        # Get labels
        labels = [bm[2] for bm in self.beams]
        # Try by index if beam_label is integer
        if type(beam_label) is int:
            try:
                all_qs = self.all_qs[beam_label]
            except IndexError:
                warnings.warn('Unable to lookup beam by index, trying by beam_label')
                beam_label = str(beam_label)
                if beam_label not in labels:
                    raise ValueError('beam_label does not specify a beam')
                else:
                    all_qs = self.all_qs[labels.index(beam_label)]
        # Otherwise search by beam_label
        else:
            # If beam_label is any other type, convert it to str
            if type(beam_label) is not str:
                try:
                    beam_label = str(beam_label)
                except:
                    raise TypeError('beam_label should be a string')
            if beam_label not in labels:
                raise ValueError('beam_label does not specify a beam')
            else:
                all_qs = self.all_qs[labels.index(beam_label)]
        # If pos num is not an int, try to convert it
        if type(pos_num) is not int:
            try:
                pos_num = int(pos_num)
            except ValueError:
                raise TypeError('pos_num should be an integer')
        # Try to lookup the q parameter
        try:
            q_out = all_qs[pos_num]
        except IndexError:
            raise ValueError('pos_num is out of range')
        # Return
        return (q_out[0].copy(), q_out[1])

    def _set_rm(self, rm):
        """ Sets the internal RayMatrix object

        :param rm: An instance of the RayMatrix class which defines the optical system
        :type rm: ray_matrix.RayMatrix
        """
        if type(rm) is not ray_matrix.RayMatrix:
            raise TypeError('rm should be an instance of the RayMatrix class')
        self.rm = rm

    def _get_rm(self):
        """ Returns the internal RayMatrix object

        :return: The internal RayMatrix object
        :rtype: ray_matrix.RayMatrix
        """
        if self.rm is None:
            raise LookupError('rm is not yet defined')
        return self.rm

    def _set_elements(self, elements):
        """ Sets the elements parameter

        :param elements: list of tuples containing information about the individual optical elements
        :type elements: list of tuple
        """
        # Check that none of the labels are the same
        seen = set()
        if any(el[3] in seen or seen.add(el[3]) for el in elements):
            raise ValueError('multiple labels are identical')
        # Check that elements is a list
        if type(elements) is not list:
            raise TypeError('elements should be a list of tuples')
        # Sort elements by position
        elements = sorted(elements, key=lambda x: x[2])
        self.elements = elements
        # Update
        self._update()

    def _get_elements(self):
        """ Returns the elements parameter of the OpticalSystem instance

        :return: the elements list from the object
        :rtype: list of tuple
        """
        return self.elements

    def _add_element(self, element, initial=False):
        """ Adds an element tuple to the elements list

        :param element: an element tuple specifying the optical element
        :type element: tuple
        """
        # If not initial, remove the empty system element
        if not initial:
            self._remove_element('empty_system')
        # Check that none of the labels are the same
        if element[3] in (el[3] for el in self.elements):
            raise ValueError('label is already used for another optical element')
        # Append and sort elements by position
        self.elements.append(element)
        self.elements = sorted(self.elements, key=lambda x: x[2])
        # Update
        self._update()

    def _remove_element(self, label):
        """ Remove an element from the optical system

        :param label: the string identifier for the element
        :type label: str
        """
        # Find the index from the label
        ind = self._get_element_index(label)
        # Remove the elemnt
        del self.elements[ind]
        # If there aren't any elements in the system, add the empty system element
        if len(self.elements) == 0:
            self._add_element(('interface_flat', (1, 1), 0, 'empty_system'), initial=True)
        # Update
        self._update()

    def _adjust_element(self, element):
        """ Overwrites an element in the optical system with a new element of the same label

        :param element: an element tuple specifying the optical element
        :type element: tuple
        """
        # Find element index
        ind = self._get_element_index(element[3])
        # Remove element
        del self.elements[ind]
        # Re-add element
        self._add_element(element)

    def _get_element_index(self, label):
        """ Returns the element index given the label

        :param label: The label assigned to the element
        :type label: str or int
        :return: The index of the element with label
        :rtype: int
        """
        if type(label) is int:
            try:
                self.elements[label]
                ind = label
            except IndexError as exc:
                raise ValueError('label is not a recognized element label') from exc
        else:
            labels = [el[3] for el in self.elements]
            try:
                ind = labels.index(label)
            except ValueError as exc:
                raise ValueError('label is not a recognized element label') from exc
        return ind

    def _get_beam_index(self, beam_label):
        """ Returns the beam index given the beam_label"""
        if type(beam_label) is int:
            try:
                self.beams[beam_label]
                ind = beam_label
            except IndexError as exc:
                raise ValueError('beam_label is not a recognized beam label') from exc
        else:
            labels = [bm[2] for bm in self.beams]
            try:
                ind = labels.index(beam_label)
            except ValueError as exc:
                raise ValueError('beam_label is not a recognized beam label') from exc
        return ind

    ###############################################################################################
    # Internal System Building and Propagation Methods
    ###############################################################################################
    def _update(self):
        """ Checks the _beam_hash and _el_hash and updates rm and qll_qs if necessary
        """
        if not hash(tuple(self.elements)) == self._el_hash:
            # Rebuild ray matrix and qs
            self._build_raymatrix()
            self._calculate_all_qs()
        elif not hash(tuple(self.beams)) == self._beam_hash:
            # Rebuild qs
            self._calculate_all_qs()

    def _build_raymatrix(self):
        """

        """
        # Define the key of possible element names
        key = {
            'mc': 'mirror_curved',
            'mf': 'mirror_flat',
            'l': 'lens_thin',
            'if': 'interface_flat',
            'ic': 'interface_curved',
            'p': 'prism',
            'mtt': 'mirror_tilted_tangential',
            'mts': 'mirror_tilted_sagittal',
            'itt': 'interface_tilted_tangential',
            'its': 'interface_tilted_sagittal'
        }
        # Raise a LookupError if no elements have been defined yet
        elements = self._get_elements()
        if not elements:
            raise LookupError('no elements have been defined yet')
        # Build descriptor
        descriptor = []
        t_now = min(0.0, elements[0][2])
        for el in elements:
            # Append the proper translation
            t_delta = el[2] - t_now
            t_now = el[2]
            if t_now > 1e-8:
                descriptor.append(('translation', t_delta))
            # Append the element
            if el[1] is None:
                descriptor.append((el[0],))
            else:
                descriptor.append((el[0], el[1]))
        # Create RayMatrix object
        self._set_rm(ray_matrix.RayMatrix(descriptor=descriptor))
        # Update hash
        self._el_hash = hash(tuple(self.elements))

    def _calculate_all_qs(self):
        """ Calculates the q parameters for all beams at all points in the optical system
        """
        all_qs = []
        for beam in self.beams:
            # Propagate q to nearest pos_num
            now_ind, now_dist = self.rm.get_distance_to_nearest_pos_num(beam[1])
            now_q = beam[0] + now_dist
            now_all_qs = []
            for jj in range(len(self.rm.sys)+1):
                ior = self.rm.get_index_of_refraction(jj)
                if jj < now_ind:
                    mat = self.rm.get_matrix_backward(el_range=[jj, now_ind])
                    now_all_qs.append((self._prop_root(now_q, mat), ior))
                elif jj == now_ind:
                    now_all_qs.append((now_q, ior))
                else:
                    mat = self.rm.get_matrix_forward(el_range=[now_ind, jj])
                    now_all_qs.append((self._prop_root(now_q, mat), ior))
            all_qs.append(tuple(now_all_qs))
        # Assign
        self.all_qs = all_qs
        # Update hash
        self._beam_hash = hash(tuple(self.beams))

    def _prop_root(self, q, mat):
        """ Propagates a q parameter with an ABCD matrix

        :param q: q parameter
        :param mat: abcd matrix
        :type q: q_param.qParameter
        :type mat: np.matrix
        :return: new q parameter
        :rtype: q_param.qParameter
        """
        q_old = q.get_q()
        q_new = (mat[0, 0] * q_old + mat[0, 1])/(mat[1, 0] * q_old + mat[1, 1])
        q_new = q_param.qParameter(q=q_new, wvlnt=q.get_wvlnt())
        return q_new

    ###############################################################################################
    # Add and Remove Optical Elements and Beams
    ###############################################################################################
    def add_element(self, element_type, parameters, z, label):
        """ Adds an optical element to the system

        Possible element types and their parameters are:
          - 'mc' or 'mirror_curved':                ROC
          - 'mf' or 'mirror_flat':                  None
          - 'l' or 'lens_thin':                     focal length
          - 'if' or 'interface_flat':               (n_ini, n_fin)
          - 'ic' or 'interface_curved':             (n_ini, n_fin, ROC)
          - 'p' or 'prism':                         (n_air, n_mat, theta_1, alpha, s)
          - 'mtt' or 'mirror_tilted_tangential':    (ROC, AOI)
          - 'mts' or 'mirror_tilted_sagittal':      (ROC, AOI)
          - 'itt' or 'interface_tilted_tangential': (n_ini, n_fin, ROC, AOI)
          - 'its' or 'interface_tilted_sagittal':   (n_ini, n_fin, ROC, AOI)

        :param element_type: A string specifying one of the designated element types
        :param parameters: A tuple or float designating the parameters to describe the element
        :param z: The position along the optical axis
        :param label: A string identifier for the element
        :type element_type: str
        :type parameters: tuple of float or float or None
        :type z: float
        :type label: str
        """
        # Define the key
        key = {
            'mc': 'mirror_curved',
            'mf': 'mirror_flat',
            'l': 'lens_thin',
            'if': 'interface_flat',
            'ic': 'interface_curved',
            'p': 'prism',
            'mtt': 'mirror_tilted_tangential',
            'mts': 'mirror_tilted_sagittal',
            'itt': 'interface_tilted_tangential',
            'its': 'interface_tilted_sagittal'
        }
        # Check some types
        if type(label) is not str:
            try:
                label = str(label)
            except:
                raise TypeError('label should be a string')
        if type(z) is not float:
            try:
                z = float(z)
            except:
                raise TypeError('z should be a float')
        # If the key is used, replace it with the long form name
        if element_type in key:
            element_type = key[element_type]
        # Check that the element_type corresponds to a known type
        if not hasattr(ray_matrix.RayMatrix, element_type):
            # Try to guess what element_type was meant
            if len(element_type) < 4:
                possibles = [x for x in key if SequenceMatcher(None, x, element_type).ratio() > 0.2]
            else:
                possibles = [key[x] for x in key if SequenceMatcher(None, key[x], element_type).ratio() > 0.6]
            print('{0} is not a known element_type.  Did you mean to use one of: ['.format(element_type) + \
                  ', '.join(possibles) + ']?')
            raise ValueError('element_type is not a known type')
        # Check that the element_type and parameters properly specify a ray matrix
        try:
            # If parameters is None, try without argument
            if parameters is None:
                getattr(ray_matrix.RayMatrix, element_type)()
            # If parameters is not a tuple, try to pass it as a single argument
            elif type(parameters) is not tuple:
                getattr(ray_matrix.RayMatrix, element_type)(parameters)
                parameters = (parameters, )
            # If it is a tuple, passing the arguments depends on the length
            elif len(parameters) == 0:
                getattr(ray_matrix.RayMatrix, element_type)()
                parameters = None
            elif len(parameters) == 1:
                getattr(ray_matrix.RayMatrix, element_type)(parameters[0])
            else:
                getattr(ray_matrix.RayMatrix, element_type)(*parameters)
            element = (element_type, parameters, z, label)
        except:
            print('parameters do not match up with element type')
            raise
        # Add the element to elements
        self._add_element(element)

    def remove_element(self, label):
        """ Removes an optical element from the system

        :param label: The string label for the element or an integer index
        :type label: str or int
        """
        # Parse label
        if type(label) not in [str, int]:
            try:
                label = str(label)
            except:
                raise TypeError('label should be a string')
        # Remove element
        self._remove_element(label=label)

    def adjust_element_z(self, label, new_z):
        """ Changes the position of an element to `new_z`

        :param label: The string label for the element or an integer index
        :param new_z: The new position of the element
        :type label: str or int
        :type new_z: float
        """
        # Get index from label
        ind = self._get_element_index(label)
        # Check that new_z can be a float
        if type(new_z) is not float:
            try:
                new_z = float(new_z)
            except ValueError as exc:
                raise TypeError('new_z should be a float')
        # Create new element
        new_el = list(self.elements[ind])
        new_el[2] = new_z
        new_el = tuple(new_el)
        # Update
        self._adjust_element(new_el)

    def add_thin_lens(self, f, z, label):
        """ Add a thin lens to the system with focal length f at position z

        :param f: focal length
        :param z: position along optical axis
        :param label: string label to identify the element
        :type f: float
        :type z: float
        :type label: str
        """
        self.add_element('lens_thin', (f,), z, label)

    def add_curved_mirror(self, roc, z, label):
        """ Add a curved mirror to the system with radius of curvature roc at position z

        :param roc: radius of curvature of mirror
        :param z: position along optical axis
        :param label: string label to identify the element
        :type roc: float
        :type z: float
        :type label: str
        """
        self.add_element('mirror_curved', (roc,), z, label)

    ###############################################################################################
    # Add and Remove  Beams
    ###############################################################################################
    def add_beam(self, waist_size, distance_to_waist, wvlnt, z, beam_label, q=None):
        """ Adds a beam to the optical system instance

        A beam in the OpticalSystem class contains the following pieces of information:
          - q_param.qParameter object specifying beam
          - location of beam along the optical axis
          - string label for each beam

        This method allows the user to add a beam with more real-world parameters, namely the
        waist size and the distance to the waist.  Note that **all beam sizes are specified by the
        1/e^2 radius**.  The z parameter specifies the location of the beam along the
        optical axis, and the label parameter is a string label used to access the beam later on.

        :param waist_size: 1/e^2 radius of beam
        :param distance_to_waist: distance between waist and location at which beam is defined. A
                                  negative number means the waist is farther along the opical axis
        :param wvlnt: the wavelength of the light
        :param z: position along optical axis at which beam is defined
        :param beam_label: a string label to access the beam later
        :param q: the q parameter can be specified directly in which case waist_size,
                  distance_to_waist, and wvlnt parameters are ignored
        :type waist_size: float
        :type distance_to_waist: float
        :type wvlnt: float
        :type z: float
        :type beam_label: str
        :type q: q_param.qParameter or None
        """
        # Build qParameter if q is None
        if q is not None:
            if type(q) is not q_param.qParameter:
                raise TypeError('q should be an instance of the qParameter class')
        else:
            q = q_param.qParameter(wvlnt=wvlnt)
            q.set_q(beamsize=waist_size, position=distance_to_waist)
        # Add beam
        self._add_beam((q, z, beam_label))

    def remove_beam(self, beam_label):
        """ Removes a beam from the optical system given the beam_label

        :param beam_label: The identifier associated with the beam to be removed
        :type beam_label: str
        """
        self._remove_beam(beam_label)

    ###############################################################################################
    # Calculations
    ###############################################################################################
    def get_q(self, beam_label, z):
        """ Returns the q parameter at the specified position within the optical system

        :param z: a position along the optical axis
        :param beam_label: the label or index of the beam
        :type z: float
        :type beam_label: str or float
        :return: (q parameter, index of refraction)
        :rtype: (q_param.qParameter, float)
        """
        # Get the pos_num and distance to the position
        pos_num, dist = self.rm.get_el_num_from_position(z)
        # Get the q parameter
        q_out = self._get_q(beam_label, pos_num)
        # Add the extra distance
        q_out = (q_out[0] + dist, q_out[1])
        return q_out

    def w(self, beam_label, z):
        """ Returns the beam size w for the specified beam at the specified position

        :param beam_label: the label or index of the beam
        :param z: the position along the optical axis
        :type beam_label: str or int
        :type z: float
        :return: 1/e^2 beam radius at the position specified
        :rtype: float
        """
        # Get the q parameter and ior at the specified location
        q, ior = self.get_q(beam_label=beam_label, z=z)
        # Scale q by the index of refraction
        q /= ior
        # Return
        return q.w(m2=1)

    def fit_q(self, guess_beam_label, data):
        """ Uses measured data to fit a beam starting with an initial guess

        This method fits a q parameter to measured data using the `scipy.optimize.minimize`
        routine.  The optical system needs to have a beam defined which this method will use as
        its initial guess.  The data should be passed in list of list format with the first
        element being the position along the optical axis, and the second being the beam size at
        that position, i.e.
           - data = [[z_1, w_z], [z_2, w_2], [z_3, w_3], ....]

        :param guess_beam_label: The beam_label of the beam which will be used as a guess
        :param data: The data to which the beam will be fit
        :type guess_beam_label: str
        :type data: list of list of float
        :return:
        """
        # Check format of data
        try:
            zs = [x[0] for x in data]
            ws = [x[1] for x in data]
        except:
            raise TypeError('data should be a list of 2-element lists')
        # Scale beam
        guess_q = self._get_beam(beam_label=guess_beam_label)
        scale_w0 = guess_q[0].w0
        # Define fitting function
        def fit_fun(params):
            """ params = (w0, z)
            """
            # Add the beam
            self.add_beam(waist_size=params[0]*scale_w0,
                          distance_to_waist=params[1],
                          wvlnt=guess_q[0].get_wvlnt(),
                          z=guess_q[1],
                          beam_label='fitting_beam')
            # Calculate sum of squares of differences
            ssq = sum(((self.w(beam_label='fitting_beam', z=z) - w)**2 for z, w in zip(zs, ws)))
            # Remove beam
            self.remove_beam(beam_label='fitting_beam')
            return ssq
        # Minimize the sum of squares
        res = minimize(fit_fun, (1, guess_q[0].z()), tol=1e-5, method='Nelder-Mead')
        if res.success:
            print(res.message)
        else:
            warnings.warn('Optimization failed with message: {0}'.format(res.message))
        # Return a q parameter
        out_q = q_param.qParameter(wvlnt=guess_q[0].get_wvlnt())
        out_q.set_q(beamsize=res.x[0], position=res.x[1])
        return out_q

    ###############################################################################################
    # Graphics
    ###############################################################################################
    def plot_w(self, zs, fig_num=None, other_sys=None):
        """ Plots the beam size for all beams at the points given in `zs`

        :param zs: the positions along the optical axis at which to plot the beams size
        :param fig_num: the figure number to use (next available if None)
        :param other_sys: another optical system or a list of other optical systems
        :type zs: list or np.ndarray
        :type fig_num: int or None
        :type other_sys: OpticalSystem or list of OpticalSystem
        :return: (figure_handle, axis_handle)
        :rtype: (plt.
        """
        # Combine systems
        systems = [self]
        if other_sys is not None:
            if type(other_sys) is OpticalSystem:
                systems.append(other_sys)
            else:
                try:
                    for sys in other_sys:
                        systems.append(sys)
                except:
                    raise TypeError('other_sys should be an OpticalSystem instance or a list of OpticalSystem instances')
        # Get beam sizes
        beam_labels, ws, elements = [], [], []
        for sys in systems:
            beam_label_n = [beam[2] for beam in sys.beams]
            beam_labels.append(beam_label_n)
            elements.append(sys.elements)
            ws_n = {}
            for beam_label in beam_label_n:
                ws_n[beam_label] = [sys.w(beam_label=beam_label, z=z) for z in zs]
            ws.append(ws_n)
        # Calculate units
        mx = max((max(max(x[key]) for key in x) for x in ws))
        scale, unit = 1, 'm'
        if np.log10(mx) < -4:
            scale, unit = 1e6, 'um'
        elif np.log10(mx) < -1:
            scale, unit = 1e3, 'mm'
        # Initialize figure
        fig = plt.figure(num=fig_num, figsize=[11, 7])
        # Grid
        grd0 = GridSpec(1, 1)
        grd0.update(left=0.1, right=0.90, bottom=0.1, top=0.93, hspace=0.30, wspace=0.28)
        # Initialize axes
        ax0 = fig.add_subplot(grd0[:, :])
        xlims = [np.min(zs), np.max(zs)]
        # Plot
        for ws_n, beam_label_n in zip(ws, beam_labels):
            for beam_label in beam_label_n:
                ln = ax0.plot(zs, [scale*w for w in ws_n[beam_label]], lw=2, label=beam_label)
                ax0.hold(True)
                ax0.plot(zs, [-1*scale*w for w in ws_n[beam_label]], lw=2, color=ln[0].get_color())
        ax0.grid(True)
        ax0.set_xlabel('Position [m]')
        ax0.set_ylabel('Beam Radius [{0}]'.format(unit))
        ax0.set_xlim(xlims)
        ax0.legend(loc='best', fontsize=12)
        # Get scales for
        ylims = ax0.get_ylim()
        dx, dy = xlims[1] - xlims[0], ylims[1] - ylims[0]
        for elements_n in elements:
            for el in elements_n:
                pos, label = el[2], el[3]
                ax0.plot([pos, pos], ylims, color='black', lw=1, ls='--')
                ax0.text(pos, ylims[1]-0.03*dy, label, rotation=-90)
        ax0.hold(False)
        ax0.set_xlim(xlims)
        ax0.set_ylim(ylims)
        # Return
        return fig, ax0
