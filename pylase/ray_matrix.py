""" A module for working with ray matrices

Ray matrices, often referred to as ABCD matrices, are useful tools for characterizing optical
systems in the paraxial limit.  The matrices describe how a ray of light with a specific input
angle and position propagates through a optical element.  The idea of an optical ray comes from
geometric optics, but ray matrices can also be used to describe optical systems governed by
diffraction such as laser systems.

Combining ray matrices with the q parameter description of Gaussian laser beams gives a simple yet
powerful framework for calculating how laser beams will transform through an optical system.
"""

import warnings
import numpy as np

__author__ = "Chris Mueller"
__email__ = "chrisark7@gmail.com"
__status__ = "Development"

###################################################################################################
# RayMatrix Class
###################################################################################################
class RayMatrix(object):
    """ A class for working with ray matrices.

    This class is designed for working with optical systems in the paraxial limit which can be
    described by the ray matrices of geometrical optics.

    Instances of the class contain two core attributes: the descriptor and the system.
      - The descriptor (:code:`desc`) is a list of tuples.  Each tuple contains a string
        identifying the type of element and a tuple with the parameters needed to specify the
        element.  Each optical element/interface needs a different number of parameters so the
        tuples associated with each element differ in length.
      - The system (:code:`sys`) is also a list of tuples.  The first element of each tuple is
        the 2x2 ray matrix which describes that element, and the second is the distance translated
        inside of that element.  Most elements other than the :code:`translation` element have
        a distance of 0. The third parameter is the index of refraction in the intervening space
        between the two matrices.

    The descriptor is what the user manipulates to define the optical system, while the system
    attribute is generated internally and is used for the calculations.

    There are two numbering systems which are important to keep in mind when working with the
    RayMatrix class: elements (:code:`el_num` or :code:`el_range`) and positions (:code:`pos_num`
    and :code:`pos_range`).
      - Elements: These are the optical elements of the system (including translations) which are
        defined by ray matrices.  If a RayMatrix instance contains N elements, then the element
        numbers range from 0 to N-1 (standard Python numbering).
      - Positions: The positions are the spaces between the elements where the q parameters live
        in an OpticalSystem instance.  If a system has N elements, then the positions numbers
        range from 0 to N.  This is because there is one more position than there are elements.
        The first position, which is prior to the first element, has :code:`pos_num=0`.  The last
        position, which is after the last element, has :code:`pos_num=N`.
    """

    def __init__(self, descriptor=None):
        """ Initializes the RayMatrix class structure

        The descriptor (:code:`desc`) is a list of tuples.  Each tuple contains a string
        identifying the type of element and a tuple with the parameters needed to specify the
        element.  Each optical element/interface needs a different number of parameters so the
        tuples associated with each element differ in length.

        It is possible, and sometimes easier, to initialize a empty instance of the class and use
        the 'build' commands to build the system up one piece at a time.  The build commands are:
          - :code:`build_prepend()`: which allows one to attach a new element at the front of the
            system.
          - :code:`build_append()`: which allows one to attach a new element at the rear of the
            system.
          - :code:`build_insert()`: which allows one to insert a new element at an arbitrary
            point in the system.

        :param descriptor: list of tuples
        :type descriptor: list
        :return: instance of the RayMatrix class
        :rtype: RayMatrix
        """
        if descriptor is None:
            self.desc = None
            self.sys = None
        elif type(descriptor[0]) is str:
            self.desc = (descriptor, )
            self.generate_system()
        else:
            self.desc = descriptor
            self.generate_system()

    def __str__(self):
        """ Builds a string which gives a visual representation of the system

        This function is useful for visualizing the setup of the current system as well as
        understanding the position number and element number system employed throughout the
        package.
        """
        heads = ['#', 'Element', 'Params', 'Dist', 'Cum. Dist']
        if self.desc is None:
            return 'system is empty'
        else:
            desc = self.desc
            out = ''
            # Determine column widths
            col_width_1 = max(len(desc).__str__().__len__() + 3, len(heads[0]))
            col_width_2 = max(max([len(x[0]) for x in desc]), len(heads[1]))
            col_width_3 = max(max([x[1].__str__().__len__() for x in desc]), len(heads[2]))
            col_width_4 = max(max([x[1].__str__().__len__() for x in self.sys]), len(heads[3]))
            col_width_5 = max(np.ceil(np.log10(self.get_distance([0, len(self.sys)]))) + 4,
                              len(heads[4]))
            tot_width = (col_width_1 + col_width_2 + col_width_3 + col_width_4 +
                         col_width_5 + 18)
            # Print header
            out += heads[0].ljust(col_width_1+2) + '| ' + \
                   heads[1].ljust(col_width_2+2) + '| ' + \
                   heads[2].ljust(col_width_3+2) + '| ' + \
                   heads[3].ljust(col_width_4+2) + '| ' + \
                   heads[4].ljust(col_width_5+2) + '\n'
            # Print horizontal rule
            out += '=' * tot_width + '\n'
            # Loop through elements
            for jj in range(len(desc)):
                # Print positions
                out += 'p# {0}'.format(jj).ljust(col_width_1+2) + '| ' + \
                       ' '.ljust(col_width_2+2) + '| ' + \
                       ' '.ljust(col_width_3+2) + '| ' + \
                       ' '.ljust(col_width_4+2) + '| ' + \
                       '{0:0.3f}'.format(self.get_distance([0, jj])).rjust(col_width_5+2) + '\n'
                # Print elements
                out += 'e# {0}'.format(jj).ljust(col_width_1+2) + '| ' + \
                       desc[jj][0].ljust(col_width_2+2) + '| ' + \
                       desc[jj][1].__str__().ljust(col_width_3+2) + '| ' + \
                       '{0:0.3f}'.format(self.get_distance(jj)).rjust(col_width_4+2) + '| ' + '\n'
            #Print final position
            jj += 1
            out += 'p# {0}'.format(jj).ljust(col_width_1+2) + '| ' + \
                   ' '.ljust(col_width_2+2) + '| ' + \
                   ' '.ljust(col_width_3+2) + '| ' + \
                   ' '.ljust(col_width_4+2) + '| ' + \
                   '{0:0.3f}'.format(self.get_distance([0, jj])).rjust(col_width_5+2)
            return out

    def __repr__(self):
        """ Defines the representation of the system when called at the command line
        """
        return self.__str__()

    def print_summary(self):
        """ Prints a summary of the current system description.

        This function is useful for visualizing the setup of the current system as well as
        understanding the position number and element number system employed throughout the
        package.
        """
        print(self.__str__())

    def generate_system(self):
        """ Uses the descriptor to build the optical system.

        This function is mostly intended for internal use.  It updates the system attribute
        (:code:`sys`) of the RayMatrix instance whenever the descriptor is changed.  It also
        updates the descriptor (:code:`desc`) to use the full terms instead of the keys.

        :return: updated instance
        :rtype: RayMatrix
        """
        # Define the key
        key = {
            't': 'translation',
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
        sys = list()
        desc = self.desc
        if desc is None:
            raise ValueError('desc is None')
        for jj in range(len(desc)):
            # If a key is used, replace it with the proper descriptor
            if desc[jj][0] in key:
                desc[jj] = (key[desc[jj][0]], desc[jj][1])
            # Check that the descriptor is properly specified and append to sys
            if hasattr(RayMatrix, desc[jj][0]):
                # If second term wasn't included, assume it is an empty
                if len(desc[jj]) == 1:
                    sys.append(getattr(RayMatrix, desc[jj][0])())
                    desc[jj] = (desc[jj][0], ())
                # If the second term isn't a tuple try to pass it as a single argument
                elif type(desc[jj][1]) is not tuple:
                    sys.append(getattr(RayMatrix, desc[jj][0])(desc[jj][1]))
                    desc[jj] = (desc[jj][0], (desc[jj][1],))
                # An empty tuple yields no arguments
                elif len(desc[jj][1]) == 0:
                    sys.append(getattr(RayMatrix, desc[jj][0])())
                elif len(desc[jj][1]) == 1:
                    sys.append(getattr(RayMatrix, desc[jj][0])(desc[jj][1][0]))
                else:
                    sys.append(getattr(RayMatrix, desc[jj][0])(*desc[jj][1]))
            else:
                raise ValueError('{0} at position {1} is not a recognized descriptor'.format(
                    desc[jj][0], jj))
        # Update indices of refraction
        first_ior_flag = False
        current_ior = None
        for jj in range(len(sys)):
            # First element
            if sys[jj][2][0] is not None:
                current_ior = sys[jj][2][0]
                if not first_ior_flag:
                    first_ior_flag = True
                    first_ior = current_ior
            else:
                sys[jj] = (sys[jj][0], sys[jj][1], (current_ior, sys[jj][2][1]))
            # Second element
            if sys[jj][2][1] is not None:
                current_ior = sys[jj][2][1]
                if not first_ior_flag:
                    first_ior_flag = True
                    first_ior = current_ior
            else:
                sys[jj] = (sys[jj][0], sys[jj][1], (sys[jj][2][0], current_ior))
        # Set the Nones which were skipped at the beginning
        if first_ior_flag:
            for jj in range(len(sys)):
                # First element
                if sys[jj][2][0] is None:
                    sys[jj] = (sys[jj][0], sys[jj][1], (first_ior, sys[jj][2][1]))
                # Second element
                if sys[jj][2][1] is None:
                    sys[jj] = (sys[jj][0], sys[jj][1], (sys[jj][2][0], first_ior))
        # If no iors were found set everything to 1
        else:
            for jj in range(len(sys)):
                sys[jj] = (sys[jj][0], sys[jj][1], (1, 1))
        self.desc = desc
        self.sys = sys

    ###############################################################################################
    # get methods
    ###############################################################################################
    def get_distance(self, pos_range='all'):
        """ Returns the distance of a group of element or of a particular element

        The primary purpose of this method is to return the cumulative distance over a group of
        elements.  However, if only an integer is specified, the distance of a single element is
        returned (in this case :code:`pos_range` is actually acting like :code:`el_num`).

        A single element is specified by its element number (an integer) while a group of elements
        is specified with a position range (a two element list of integers).  When specifying a
        group of elements, the same slicing protocol used in numpy arrays is used.  So, for example
        :code:`pos_range=[0, 3]` will return the cumulative distance of elements 0, 1, and 2.  Note
        that :code:`pos_range=[0, 0]` will always return 0.

        Use :code:`print_summary()` to see the element and position numbers.

        :param pos_range either the element number or a two element list of integers
        :type pos_range: int or list or str
        :return: total distance of specified elements
        :rtype: float
        """
        if pos_range == 'all':
            pos_range = [0, len(self.sys)]
        if type(pos_range) is int:
            d = self.sys[pos_range][1]
        elif len(pos_range) == 2:
            d = 0.0
            for jj in range(pos_range[0], pos_range[1]):
                d += self.sys[jj][1]
        else:
            raise TypeError('pos_range is improperly specified')
        return d

    def get_distance_to_nearest_pos_num(self, z):
        """ Returns the distance to the nearest pos_num given an arbitrary location in the system

        :param z: position along the optical axis
        :type z: float
        :return: (pos_num, distance)
        :rtype: (int, float)
        """
        # Calculate position of all optical elements
        positions = []
        for jj in range(len(self.sys)):
            positions.append([jj, self.get_distance(pos_range=[0, jj]) - z])
        # Find the closest index
        min_val = min(positions, key=lambda x: abs(x[1]))
        return min_val

    def get_el_num_from_position(self, z, start_pos_num=0):
        """ Returns the pos_num and forward distance to a given position from the start position.

        This method takes in a starting pos_num number and a distance from that starting pos_num
        (which could be negative) and returns the pos_num of the closest position with the
        additional distance needed to get to the specified position.  It can be thought of as
        providing a road-map to the specified location from the starting position.

        For instance consider a system defined by three translation matrices:
        desc=[('t', 1), ('t', 1), ('t', 1)]. When asked for :code:`z=-0.75` from
        :code:`start_pos_num=2` will return (1, 0.25).  Because getting the q parameter to that
        position requires backwards propagation to :code:`pos_num=1` then translating forward by
        0.25

        The returned distance will only be negative if the specified distance is less than
        position of the first element.

        :param z: The distance (which can be negative) from the start position
        :param start_pos_num: The start position index
        :type z: float
        :type start_pos_num: int
        :return: (closest el_num before position, remainder distance)
        :rtype: (int, float)
        """
        # Get total distance to front and back
        tot_bkd = self.get_distance(pos_range=[0, start_pos_num])
        tot_fwd = self.get_distance(pos_range='all') - tot_bkd
        # Cases on z: 0, >0, <0
        if z == 0:
            ret_val = (start_pos_num, 0.0)
        elif z > 0:
            if z > tot_fwd:
                ret_val = (len(self.desc), z - tot_fwd)
            else:
                now_dist = 0
                now_ind = start_pos_num
                while z > now_dist:
                    now_ind += 1
                    now_dist = self.get_distance(pos_range=[start_pos_num, now_ind])
                else:
                    if now_ind == 0:
                        now_ind += 1
                ret_val = (now_ind - 1, z - self.get_distance(pos_range=[start_pos_num, now_ind-1]))
        else:
            if abs(z) > tot_bkd:
                ret_val = (0, tot_bkd + z)
            else:
                z_fwd = tot_bkd + z
                now_dist = 0
                now_ind = 0
                while z_fwd > now_dist:
                    now_ind += 1
                    now_dist = self.get_distance(pos_range=[0, now_ind])
                else:
                    if now_ind == 0:
                        now_ind += 1
                ret_val = (now_ind - 1, z_fwd - self.get_distance(pos_range=[0, now_ind - 1]))
        return ret_val

    def get_matrix_forward(self, el_range='all'):
        """ Returns the forward matrix of a particular element or a group of elements

        This method returns either the matrix of a particular element or the total matrix of
        a group of elements.  A single element is specified by its element number (an integer)
        while a group of elements is specified with a two element list of integers.

        When specifying a group of elements, the same slicing protocol used in numpy arrays is
        used.  So, for example :code:`el_range=[0,3]` will return the total distance of elements 0,
        1, and 2.

        Use :code:`print_summary()` to see the element numbers.

        :param el_range: either the element number or a two element list of integers
        :type el_range: int or list
        :return: total matrix of specified elements
        :rtype: np.matrix
        """
        if el_range == 'all':
            el_range = [0, len(self.sys)]
        if type(el_range) is int:
            mat = self.sys[el_range][0]
        elif len(el_range) == 2:
            mat = np.matrix([[1.0, 0], [0, 1.0]])
            for jj in range(el_range[0], el_range[1]):
                mat = self.sys[jj][0] * mat
        else:
            raise TypeError('el_range is improperly specified')
        return mat

    def get_matrix_backward(self, el_range='all'):
        """ Returns the backward matrix for an element or group of elements.

        This method simple calls :code:`get_matrix_forward()` and inverts it.  The docstring of
        that method has more details about the :code:`el_range` parameter.

        :param el_range: either the element number or a two element list of integers
        :type el_range: int or list
        :return: total matrix of specified elements
        :rtype: np.matrix
        """
        # Get forward matrix
        mat = self.get_matrix_forward(el_range=el_range)
        # Invert
        a = mat[0, 0]
        b = mat[0, 1]
        c = mat[1, 0]
        d = mat[1, 1]
        inv_mat = 1/(a*d - b*c) * np.matrix([[d, -b], [-c, a]])
        return inv_mat

    def get_index_of_refraction(self, pos_num):
        """ Returns the index of refraction associated with a particular position number

        :param pos_num: The index of the position
        :type pos_num: int
        """
        if type(pos_num) is not int:
            warnings.warn('pos_num is being converted to an int')
            pos_num = int(pos_num)
        if pos_num == len(self.sys):
            ior = self.sys[pos_num - 1][2][1]
        elif pos_num >=0 and pos_num < len(self.sys):
            ior = self.sys[pos_num][2][0]
        else:
            raise ValueError('pos_num is not an index of the system')
        return ior

    ###############################################################################################
    # insert and info methods
    ###############################################################################################
    def build_prepend(self, new_desc):
        """ Adds an element at the beginning of the current optical system.

        :param new_desc: A tuple containing the element identifier and its parameters
        :type new_desc: tuple
        :return: modified instance of RayMatrix object
        :rtype: RayMatrix
        """
        desc = self.desc
        if desc is None:
            desc = [new_desc]
        else:
            desc.insert(0, new_desc)
        self.desc = desc
        self.generate_system()
        return self

    def build_append(self, new_desc):
        """ Adds an element at the end of the current optical system

        :param new_desc: A tuple containing the element identifier and its parameters
        :type new_desc: tuple
        :return: modified instance of RayMatrix object
        :rtype: RayMatrix
        """
        desc = self.desc
        if desc is None:
            desc = [new_desc]
        else:
            desc.append(new_desc)
        self.desc = desc
        self.generate_system()
        return self

    def build_insert(self, new_desc, pos_num=0):
        """ Inserts an element at the specified position.

        Use :code:`print_summary()` in order to identify the proper position number.  The position
        number is read such that the new element will be inserted before that position.  E.g.
        :code:`build_insert(new_desc, 0)` is equivalent to :code:`build_prepend(new_desc)`.
        This method uses the :code:`list.insert()` function so the position index is specified in
        the same way.

        One other way to think of it is that the inserted element will have :code:`position` as its
        element number.

        :param new_desc: A tuple containing the element identifier and its parameters
        :param pos_num: An index describing where to insert the new element
        :type new_desc: tuple
        :type pos_num: int
        :return: modified instance of RayMatrix object
        :rtype: RayMatrix
        """
        desc = self.desc
        if desc is None:
            desc = [new_desc]
        else:
            desc.insert(pos_num, new_desc)
        self.desc = desc
        self.generate_system()
        return self

    ###############################################################################################
    # Ray Matrices
    ###############################################################################################
    @classmethod
    def translation(cls, dist):
        """ Ray matrix for translation by a distance :code:`dist`.

        :param dist: distance of the translation
        :type dist: float
        :return: (ray matrix, total distance)
        :rtype: (np.matrix, float)
        """
        return np.matrix([[1, dist], [0, 1]]), dist, (None, None)

    @classmethod
    def mirror_curved(cls, roc):
        """ Ray matrix for reflection from a curved mirror with ROC of :code:`roc`.

        :param roc: radius of curvature of the mirror
        :type roc: float
        :return: (ray matrix, total distance)
        :rtype: (np.matrix, float)
        """
        return np.matrix([[1, 0], [-2/roc, 1]]), 0, (None, None)

    @classmethod
    def mirror_flat(cls):
        """ Ray matrix for reflection from a flat mirror.

        :return: (ray matrix, total distance)
        :rtype: (np.matrix, float)
        """
        return np.matrix([[1, 0], [0, 1]]), 0, (None, None)

    @classmethod
    def lens_thin(cls, foc):
        """ Ray matrix for a thin lens of focal length :code:`foc`.

        :param foc: the focal length of the thin lens
        :type foc: float
        :return: (ray matrix, total distance)
        :rtype: (np.matrix, float)
        """
        return np.matrix([[1, 0], [-1/foc, 1]]), 0, (None, None)

    @classmethod
    def interface_flat(cls, n_ini, n_fin):
        """ The ray matrix for transitioning from one medium to another through a flat interface.

        :param n_ini: initial index of refraction
        :param n_fin: final index of refraction
        :type n_ini: float
        :type n_fin: float
        :return: (ray matrix, total distance)
        :rtype: (np.matrix, float)
        """
        return np.matrix([[1, 0], [0, n_ini/n_fin]]), 0, (n_ini, n_fin)

    @classmethod
    def interface_curved(cls, n_ini, n_fin, roc):
        """ The ray matrix for transitioning through a curved interface.

        The ROC is defined as being negative if the surface is concave from the point of view of
        the beam and positive otherwise.

        :param n_ini: initial index of refraction
        :param n_fin: final index of refraction
        :param roc: the radius of curvature of the interface
        :type n_ini: float
        :type n_fin: float
        :type roc: float
        :return: (ray matrix, total distance)
        :rtype: (np.matrix, float)
        """
        return np.matrix([[1, 0], [(n_ini - n_fin)/(roc * n_fin), n_ini/n_fin]]), 0, (n_ini, n_fin)

    @classmethod
    def prism(cls, n_air, n_mat, theta1, alpha, s):
        """ The ray matrix for a prism

        Prisms have the ability to shape the beam, and can be used a beam shaping devices along
        one of the axes.  This method returns the ray matrix for a standard beam shaping prism.

        The index of refraction of the air, n_air (n_air=1 usually), and the material, n_mat, are
        the first two arguments.  The last three arguments are the input angle theta1, the opening
        angle of the prism, alpha, and the vertical distance from the apex of the prism, s.

        The matrix is taken from: Kasuya, T., Suzuki, T. & Shimoda, K. A prism anamorphic system
        for Gaussian beam expander. Appl. Phys. 17, 131–136 (1978). and figure 1 of that paper has
        a clear definition of the different parameters.

        :param n_air: index of refraction of the air (usually 1)
        :param n_mat: index of refraction of the prism material
        :param theta1: input angle to the prism wrt the prism face
        :param alpha: opening angle of the prism (alpha=0 is equivalent to a glass plate)
        :param s: vertical distance from the apex of the prism.
        :type n_air: float
        :type n_mat: float
        :type theta1: float
        :type alpha: float
        :type s: float
        :return: (ray matrix, total distance)
        :rtype: (np.matrix, float)
        """
        # Calculate the other angles
        th1 = theta1
        th2 = np.arcsin(n_air/n_mat * np.sin(th1))
        th3 = alpha - th2
        th4 = np.arcsin(n_mat/n_air * np.sin(th3))
        # Calculate magnification and b (slight error in paper on b)
        m = np.cos(th2) * np.cos(th4) / (np.cos(th1) * np.cos(th3))
        b = (s * n_air * np.sin(alpha) * np.cos(th1) * np.cos(th4) /
             (n_mat * np.cos(th3)**2 * np.cos(th2)))
        # Calculate translation distance
        d = s * np.sin(alpha)/np.sin(np.pi/2 - th3)
        # Return
        return np.matrix([[m, b], [0, 1/m]]), d, (None, None)

    @classmethod
    def mirror_tilted_tangential(cls, roc, aoi):
        """ The ray matrix for the tangential rays of a tilted curved mirror

        Tangential rays are those that lie in the plane containing the center of the beam and the
        normal to the center of the mirror.  I.E. this it the ray matrix for the plane in which
        the mirror is tilted.

        :param roc: radius of curvature of the mirror
        :param aoi: angle of incidence between mirror and beam
        :type roc: float
        :type aoi: float
        :return: (ray matrix, total distance)
        :rtype: (np.matrix, float)
        """
        return np.matrix([[1, 0], [-2/(roc * np.cos(aoi)), 1]]), 0, (None, None)

    @classmethod
    def mirror_tilted_sagittal(cls, roc, aoi):
        """ The ray matrix for the sagittal rays of a tilted curved mirror

        Sagittal rays are those which lie in the plane perpendicular to the tangential plane.  I.E.
        the one perpendicular to the plane in which the surface is tilted.

        :param roc: radius of curvature of the mirror
        :param aoi: angle of incidence between mirror and beam
        :type roc: float
        :type aoi: float
        :return: (ray matrix, total distance)
        :rtype: (np.matrix, float)
        """
        return np.matrix([[1, 0], [-2/roc * np.cos(aoi), 1]]), 0, (None, None)

    @classmethod
    def interface_tilted_tangential(cls, n_ini, n_fin, roc, aoi):
        """ The ray matrix for the tangential rays of a tilted curved interface

        Tangential rays are those that lie in the plane containing the center of the beam and the
        normal to the interface.  I.E. this it the ray matrix for the plane in which the mirror is
        tilted.

        Note that the angle is the input angle relative to the beam which may be different from
        the tilt angle if the beam is transitioning from glass to air, for instance.

        :param n_ini: initial index of refraction
        :param n_fin: final index of refraction
        :param roc: radius of curvature of the interface (set to very large for flat)
        :param aoi: angle of incidence between interface and beam in radians
        :type n_ini: float
        :type n_fin: float
        :type roc: float
        :type aoi: float
        :return: (ray matrix, total distance)
        :rtype: (np.matrix, float)
        """
        nr = n_fin/n_ini
        mat = np.matrix(
            [
                [np.sqrt(nr**2 - np.sin(aoi)**2)/(nr * np.cos(aoi)),
                 0],
                [(np.cos(aoi) - np.sqrt(nr**2 - np.sin(aoi)**2)) /
                    (roc * np.cos(aoi) * np.sqrt(nr**2 - np.sin(aoi)**2)),
                 np.cos(aoi)/np.sqrt(nr**2 - np.sin(aoi)**2)]
            ])
        return mat, 0, (n_ini, n_fin)

    @classmethod
    def interface_tilted_sagittal(cls, n_ini, n_fin, roc, aoi):
        """ The ray matrix for the sagittal rays of a tilted curved interface

        Sagittal rays are those which lie in the plane perpendicular to the tangential plane.  I.E.
        the one perpendicular to the plane in which the surface is tilted.

        Note that the angle is the input angle relative to the beam which may be different from
        the tilt angle if the beam is transitioning from glass to air, for instance.

        :param n_ini: initial index of refraction
        :param n_fin: final index of refraction
        :param roc: radius of curvature of the interface (set to very large for flat)
        :param aoi: angle of incidence between interface and beam in radians
        :type n_ini: float
        :type n_fin: float
        :type roc: float
        :type aoi: float
        :return: (ray matrix, total distance)
        :rtype: (np.matrix, float)
        """
        nr = n_fin/n_ini
        mat = np.matrix(
            [
                [1,
                 0],
                [(np.cos(aoi) - np.sqrt(nr**2 - np.sin(aoi)**2))/(roc * nr),
                 1/nr]
            ])
        return mat, 0, (n_ini, n_fin)