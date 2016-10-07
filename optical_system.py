""" Combines the ray_matrix and q_parameter modules to create an optical system


"""

import warnings
import numpy as np
import ray_matrix
import q_param

__author__ = "Chris Mueller"
__email__ = "chrisark7@gmail.com"
__status__ = "Development"

###################################################################################################
# opticalSystem Class
###################################################################################################
class OpticalSystem(ray_matrix.RayMatrix):
    """ A class for working with optical systems consisting of ray matrices and q parameters

    This class has the same core components as the RayMatrix class with the addition of a list of
    tuples describing q parameters at various points in the system.

    The descriptor and system properties of this class are the same as that of the RayMatrix class
    since it is a subclass.  The qs property is a list of 2 element tuples.  Each tuple contains
    an instance of the qParameter class and an index which describes where that qParameter exists
    within the optical system.  If q parameters are passed without an index, then they are assumed
    to exist at the beginning of the system.

    Because the class can handle multiple q parameters which can all exist at different positions
    in the optical system, it is important to keep the numbering clear.  Throughout the class the
    two indexes will be referred to as:
      - q_num: This is index of the q parameter in the :code:`qs` tuple
      - pos_num: This is the index of the position within the optical system
    """
    def __init__(self, descriptor=None, qs=None):
        """
        :param descriptor: The same descriptor used to define the RayMatrix class.
        :param qs: A list of 2-tuples containing qParameter instances and and index
        :type descriptor: list or tuple
        :type qs: list or qParameter
        """
        # Initalize RayMatrix
        ray_matrix.RayMatrix.__init__(self, descriptor)
        # Parse qs
        if qs is None:
            qs = None
        elif type(qs) is q_param.qParameter:
            qs = [(qs, 0)]
        elif type(qs) is list or type(qs) is tuple:
            if type(qs[0]) is q_param.qParameter:
                qs = [(x, 0) for x in qs]
            elif hasattr(qs[0], '__len__'):
                qs = list(qs)
            else:
                raise TypeError('qs should be a list of q parameters or tuples')
        else:
            raise TypeError('qs should be a list of q parameters or tuples')
        # Check that qs are complex and indices are integers
        if qs is None:
            qsnew = None
        else:
            qsnew = list()
            for jj in range(len(qs)):
                if type(qs[jj][0]) is not q_param.qParameter:
                    raise TypeError('qs should contain instances of the qParameter class')
                qsnew.append((qs[jj][0], int(qs[jj][1])))
        # Assign
        self.qs = qsnew

    def __str__(self):
        """ Prints a summary of the current system description.

        This function is useful for visualizing the setup of the current system.
        """
        heads = ['#', 'Element', 'Params', 'Dist', 'Cum. Dist']
        # Begin printout
        if self.desc is None:
            return 'system is empty'
        else:
            desc = self.desc
            out = ''
            # Get q parameters
            q0_vals = list()
            for jj in range(len(self.sys) + 1):
                q0_vals.append(self.prop_q_index(q_num=0, pos_num=jj).get_q())
            # Determine column widths
            col_width_1 = max(len(desc).__str__().__len__() + 3, len(heads[0]))
            col_width_2 = max(max([len(x[0]) for x in desc]), len(heads[1]))
            col_width_3 = max(max([x[1].__str__().__len__() for x in desc]), len(heads[2]))
            col_width_4 = max(max([x[1].__str__().__len__() for x in self.sys]), len(heads[3]))
            col_width_5 = max(np.ceil(np.log10(self.get_distance([0, len(self.sys)]))) + 4,
                              len(heads[4]))
            # Include q parameter space
            col_width_2 = max(col_width_2, max([len('q0 = {0:0.3f} + {1:0.3f}*1j'.format(
                x.real, x.imag)) for x in q0_vals]))
            col_width_3 = max(col_width_3, max([len('n = {0:0.3f}'.format(x[2][0])) for
                                                x in self.sys]))
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
                out += ' p# {0}'.format(jj).ljust(col_width_1+2) + '| ' + \
                       ' q0 = {0:0.3f} + {1:0.3f}*1j'.format(
                           q0_vals[jj].real, q0_vals[jj].imag).ljust(col_width_2+2) + '| ' + \
                       ' n = {0:0.3f}'.format(self.get_index_of_refraction(pos_num=jj)).ljust(
                           col_width_3+2) + '| ' + \
                       ' '.ljust(col_width_4+2) + '| ' + \
                       '{0:0.3f}'.format(self.get_distance([0, jj])).rjust(col_width_5+2) + '\n'
                # Print elements
                out += 'e# {0}'.format(jj).ljust(col_width_1+2) + '| ' + \
                       desc[jj][0].ljust(col_width_2+2) + '| ' + \
                       desc[jj][1].__str__().ljust(col_width_3+2) + '| ' + \
                       '{0:0.3f}'.format(self.get_distance(jj)).rjust(col_width_4+2) + '| ' + '\n'
            #Print final position
            jj += 1
            out += ' p# {0}'.format(jj).ljust(col_width_1+2) + '| ' + \
                   ' q0 = {0:0.3f} + {1:0.3f}*1j'.format(
                       q0_vals[jj].real, q0_vals[jj].imag).ljust(col_width_2+2) + '| ' + \
                   ' n = {0:0.3f}'.format(self.get_index_of_refraction(pos_num=jj)).ljust(
                       col_width_3+2) + '| ' + \
                   ' '.ljust(col_width_4+2) + '| ' + \
                   '{0:0.3f}'.format(self.get_distance([0, jj])).rjust(col_width_5+2)
            return out

    def __repr__(self):
        """ Defines the representation of the system when called at the command line
        """
        return self.__str__()

    def print_summary(self):
        """ Prints a summary of the current system description.

        This function is useful for visualizing the setup of the current system.
        """
        print(self.__str__())

    ###############################################################################################
    # set/get and print methods
    ###############################################################################################
    def add_q(self, q, pos_num=0):
        """ Adds a q parameter to the OpticalSystem instance

        Note that the q parameter can be either a complex number or an instance of the qParameter
        class.  Internally it will be converted to an instance of the qParameter class.

        :param q: The q parameter to add to the OpticalSystem instance
        :param pos_num: The index location of the q parameter in the RayMatrix system
        :type q: qParameter
        :type pos_num: int
        """
        # Check types
        if type(pos_num) is not int:
            warnings.warn('converting pos_num to an integer')
            pos_num = int(pos_num)
        if type(q) is q_param.qParameter:
            val = (q, pos_num)
        else:
            raise TypeError('q should an instance of the qParameter class')
        # Append
        if self.qs is None:
            self.qs = [val]
        else:
            self.qs.append(val)

    def get_q_and_ind(self, q_num=0):
        """ Returns the q parameter specified by q_num

        :param q_num: The index number of the q parameter in qs
        :type q_num: int
        :return: q parameter
        :rtype: qParameter
        """
        # Check q_num
        if type(q_num) is not int:
            raise TypeError('q_num should be an integer')
        elif q_num < 0:
            raise ValueError('q_num should be at least 0')
        elif q_num > len(self.qs) - 1:
            raise ValueError('q_num is larger than the total number of qs')
        else:
            q = self.qs[q_num]
        return q

    ###############################################################################################
    # Propagate q Parameters
    ###############################################################################################
    @staticmethod
    def prop_root(q, mat):
        """ Propagates a q parameter with an ABCD matrix

        :param q: q parameter
        :param mat: abcd matrix
        :type q: complex
        :type mat: np.matrix
        :return: new q parameter
        :rtype: complex
        """
        q_new = (mat[0, 0] * q + mat[0, 1])/(mat[1, 0] * q + mat[1, 1])
        return q_new

    def prop_q_index(self, q_num=0, pos_num=0):
        """ Propagates the q parameter to any postion in the optical system

        This method calculates the q parameter at any position in the optical system.  The
        position is specified (as elsewhere in the OpticalSystem class) by the index number in
        the RayMatrix system, pos_num.

        :param q_num: specifies which q parameter to propagate
        :param pos_num: specifies where the q parameter should be propagated to
        :type q_num: int
        :type pos_num: int
        :return: q parameter
        :rtype: qParameter
        """
        # Get q and check pos_num
        q_val, q_ind = self.get_q_and_ind(q_num=q_num)
        q_out = q_val.copy()
        if type(pos_num) is not int:
            raise TypeError('pos_num should be an integer')
        elif pos_num < 0:
            raise ValueError('pos_num should be greater than or equal to 0')
        elif pos_num > len(self.desc):
            raise ValueError('pos_num should be an index of the RayMatrix instance')
        # Compare pos_num to q index
        if pos_num > q_ind:
            mat = self.get_matrix_forward(el_range=[q_ind, pos_num])
            q_out.set_q(q=self.prop_root(q_val.get_q(), mat))
        elif pos_num < q_ind:
            mat = self.get_matrix_backward(el_range=[pos_num, q_ind])
            q_out.set_q(q=self.prop_root(q_val.get_q(), mat))
        # Return
        return q_out

    ###############################################################################################
    # Beam Properties
    ###############################################################################################
    def w(self, z, q_num=0, m2=1):
        """ Calculates the beam size at an arbitrary point in the optical system.

        This is one of the most used methods of the OpticalSystem class.  It is used to calculate
        the diffraction limited beam size at an arbitrary location in the optical system.

        :param z: The position relative to the beginning of the system
        :param q_num: The q parameter to propagate through the system
        :param m2: The M**2 parameter, usually 1
        :type z: float
        :type q_num: int
        :type m2: float
        :return: beam size at position z
        :rtype: float
        """
        # Get qParameter information
        q, start_pos_num = self.get_q_and_ind(q_num=q_num)
        z_start = self.get_distance(pos_range=[0, start_pos_num])
        # Get closest position and distance
        pos_num, z_extra = self.get_el_num_from_position(z=z-z_start, start_pos_num=start_pos_num)
        # Propagate q to that point
        q_out = self.prop_q_index(q_num=q_num, pos_num=pos_num) + z_extra
        # Get index of refraction
        ior = self.get_index_of_refraction(pos_num=pos_num)
        q_out = q_out.set_q(q_out.get_q()/ior)
        # Return beam size
        return q_out.w(m2=m2)

    def gouy_cumulative(self, z, q_num=0):
        """ Calculates the cumulative Gouy phase at any point in the optical system

        :param z: The position relative to the beginning of the system
        :param q_num: The q parameter to propagate through the system
        :type z: float
        :type q_num: int
        :return: cumulative Gouy phase at position z
        :rtype: float
        """
        # Get q parameter
        q, start_pos_num = self.get_q_and_ind(q_num=q_num)
        z_start = self.get_distance(pos_range=[0, start_pos_num])
        # Calculate starting Gouy phase at all positions
        gouy_ini = list()
        gouy_current = 0
        for jj in range(len(self.sys) + 1):
            q_out = self.prop_q_index(q_num=q_num, pos_num=jj)
            gouy_ini.append(gouy_current)
            if not jj == len(self.sys):
                dist = self.get_distance(pos_range=jj)
                gouy_current += (q_out + dist).gouy() - q_out.gouy()
        # Get closest position and distance
        pos_num, z_extra = self.get_el_num_from_position(z=z-z_start, start_pos_num=start_pos_num)
        # Propagate q
        q_out = self.prop_q_index(q_num=q_num, pos_num=pos_num)
        gouy_out = (q_out + z_extra).gouy() - q_out.gouy() + gouy_ini[pos_num]
        return gouy_out

