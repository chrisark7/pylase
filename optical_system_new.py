""" Work with ray matrices and q parameters to calculate properties of an optical system

The Gaussian beam q parameter, encapsulated in the qParameter class, completely defines a Gaussian
laser beam in the paraxial limit.  Similiarly, a system of geometrical optical ray matrices,
encapsulated in the RayMatrix class, completely defines a series of optical system in the paraxial
limit.  This module brings the two of those togeter to allow the user to calculate properties of
an optical system consisting of a number of optical elements and a Gaussian laser beam.
"""

import warnings
import numpy as np
import ray_matrix
import q_param
from difflib import SequenceMatcher

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
        # Initialize the beams component to an empty list
        self.beams = []
        # Initialize the elements list
        self.elements = []
        # Initialize a list for the q parameters
        self.all_qs = []

    ###############################################################################################
    # Internal get/set/add/remove methods
    ###############################################################################################
    def _get_beam(self, beam_label):
        """ Returns the beam specified by beam_label

        beam_label should be the string specifying the beam, but an integer index is also accepted
        (though discouraged).

        :param beam_label: The string label associated with the beam (an index is also accepted)
        :type beam_label: str or int
        :return: The internal beam definition: (qParameter, position, label)
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
        # Recalculate qs
        self._calculate_all_qs()

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
        return q_out

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
        # Rebuild the ray matrix
        self._build_raymatrix()
        # Recalculate the q parameters
        self._calculate_all_qs()

    def _get_elements(self):
        """ Returns the elements parameter of the OpticalSystem instance

        :return: the elements list from the object
        :rtype: list of tuple
        """
        return self.elements

    def _add_element(self, element):
        """ Adds an element tuple to the elements list

        :param element: an element tuple specifying the optical element
        :type element: tuple
        """
        # Check that none of the labels are the same
        if element[3] in (el[3] for el in self.elements):
            raise ValueError('label is already used for another optical element')
        # Append and sort elements by position
        self.elements.append(element)
        self.elements = sorted(self.elements, key=lambda x: x[2])
        # Rebuild the ray matrix
        self._build_raymatrix()
        # Recalculate all of the q parameters
        self._calculate_all_qs()

    ###############################################################################################
    # Internal System Building and Propagation Methods
    ###############################################################################################
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
    def add_element(self, element_type, parameters, position, label):
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
        :param position: The position along the optical axis
        :param label: A string identifier for the element
        :type element_type: str
        :type parameters: tuple of float
        :type position: float
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
        if type(position) is not float:
            try:
                position = float(position)
            except:
                raise TypeError('position should be a float')
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
            element = (element_type, parameters, position, label)
        except:
            print('parameters do not match up with element type')
            raise
        # Add the element to elements
        self._add_element(element)

    def add_beam(self, waist_size, distance_to_waist, wvlnt, location, label, q=None):
        """ Adds a beam to the optical system instance

        A beam in the OpticalSystem class contains the following pieces of information:
          - q_param.qParameter object specifying beam
          - location of beam along the optical axis
          - string label for each beam

        This method allows the user to add a beam with more real-world parameters, namely the
        waist size and the distance to the waist.  Note that **all beam sizes are specified by the
        1/e^2 radius**.  The location parameter specifies the location of the beam along the
        optical axis, and the label parameter is a string label used to access the beam later on.

        :param waist_size: 1/e^2 radius of beam
        :param distance_to_waist: distance between waist and location at which beam is defined. A
                                  negative number means the waist is farther along the opical axis
        :param wvlnt: the wavelength of the light
        :param location: position along optical axis at which beam is defined
        :param label: a string label to access the beam later
        :param q: the q parameter can be specified directly in which case waist_size,
                  distance_to_waist, and wvlnt parameters are ignored
        :type waist_size: float
        :type distance_to_waist: float
        :type wvlnt: float
        :type location: float
        :type label: str
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
        self._add_beam((q, location, label))

    ###############################################################################################
    # Calculations
    ###############################################################################################
    def get_q(self, beam_label, position):
        """ Returns the q parameter at the specified position within the optical system

        :param position: a position along the optical axis
        :param beam_label: the label or index of the beam
        :type position: float
        :type beam_label: str or float
        :return: (q parameter, index of refraction)
        :rtype: (q_param.qParameter, float)
        """
        # Get the pos_num and distance to the position
        pos_num, dist = self.rm.get_el_num_from_position(position)
        # Get the q parameter
        q_out = self._get_q(beam_label, pos_num)
        # Add the extra distance
        q_out = (q_out[0] + dist, q_out[1])
        return q_out














