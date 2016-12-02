""" A module for working with ray matrices

Ray matrices, often referred to as ABCD matrices, are useful tools for
characterizing optical systems in the paraxial limit.  The matrices describe
how a ray of light with a specific input angle and position propagates through
a optical element.  The idea of an optical ray comes from geometric optics, but
ray matrices can also be used to describe optical systems governed by
diffraction such as laser systems.

Combining ray matrices with the q parameter description of Gaussian laser beams
gives a simple yet powerful framework for calculating how laser beams will
transform through an optical system.
"""

import warnings
import numpy as np

__author__ = "Chris Mueller"
__status__ = "Development"

class RayMatrix:
    """ A class for ray matrices from geometrical optics
    """
    def __init__(self):
        """ Initializes an empty RayMatrix
        """
        self.matrix = None
        self.ior_init = None
        self.ior_fin = None
        self.dist_internal = None
        self.type = None
        self.parameters = None

class TranslationRM(RayMatrix):
    """ The RayMatrix for translation
    """
    def __init__(self, distance):
        """ Defines the translation RayMatrix

        :param distance: translation distance
        :type distance: float
        """
        super(TranslationRM, self).__init__()
        self.matrix = np.matrix([[1, distance], [0, 1]], dtype=np.float64)
        self.dist_internal = distance
        self.type = "Translation"
        self.parameters = ["d={0:0.3g}".format(distance)]

class ThinLensRM(RayMatrix):
    """ RayMatrix for a thin lens
    """
    def __init__(self, f):
        """ RayMatrix for a thin lens with focal length `f`

        Note that a thick lens can be created from multiple interfaces

        :param f: focal length of the lens
        :type f: float
        """
        super(ThinLensRM, self).__init__()
        self.matrix = np.matrix([[1, 0], [-1/f, 1]], dtype=np.float64)
        self.dist_internal = 0
        self.type = "Thin Lens"
        self.parameters = ["f={0:0.3g}".format(f)]

class PrismRM(RayMatrix):
    """ RayMatrix for a prism
    """
    def __init__(self, n_air, n_mat, theta1, alpha, s):
        """ RayMatrix for a prism

        Prisms have the ability to shape the beam, and can be used as beam
        shaping devices along one of the axes.  This method returns the ray
        matrix for a standard beam shaping prism.

        The index of refraction of the air, `n_air` (n_air=1 usually), and the
        material, `n_mat`, are the first two arguments.  The last three arguments
        are the input angle `theta1`, the opening angle of the prism, `alpha`, and
        the vertical distance from the apex of the prism, `s`.

        The matrix is taken from: Kasuya, T., Suzuki, T. & Shimoda, K. A prism
        anamorphic system for Gaussian beam expander. Appl. Phys. 17, 131–136
        (1978). and figure 1 of that paper has a clear definition of the
        different parameters.

        :param n_air: index of refraction of the air (usually 1)
        :param n_mat: index of refraction of the prism material
        :param theta1: input angle to the prism wrt the prism face
        :param alpha: opening angle of the prism (alpha=0 is equivalent to a
            glass plate)
        :param s: vertical distance from the apex of the prism.
        :type n_air: float
        :type n_mat: float
        :type theta1: float
        :type alpha: float
        :type s: float
        """
        super(PrismRM, self).__init__()
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
        self.matrix = np.matrix([[m, b], [0, 1/m]], dtype=np.float64)
        self.dist_internal = d
        # Set strings
        self.type = "Prism"
        self.parameters = [x + "{0:0.3g}".format(y) for x, y in
                           zip(("n_air=", "n_mat=", "theta1=", "alpha=", "s="),
                               (n_air, n_mat, theta1, alpha, s))]

class MirrorRM(RayMatrix):
    """ The RayMatrix for a curved or flat mirror
    """
    def __init__(self, roc=None, aoi=None, orientation='sagittal'):
        """ RayMatrix for a mirror which can optionally be curved and/or tilted

        This method creates the RayMatrix instance for a curved or flat mirror
        which can be at normal incidence or at an angle.  If the optical axis
        has a non-zero angle of incidence with the mirror, then it is important
        to specify if the ray matrix is for the sagittal or tangential rays.

        For a typical optical system where the optical axis stays in a plane
        parallel to the surface of the table, then sagittal rays are those in
        the vertical direction (y axis) and tangential rays are those in the
        horizontal direction (x axis).

        :param roc: radius of curvature of the mirror (None for flat)
        :param aoi: angle of incidence of the optical axis with the mirror in
            radians
        :param orientation: orientation of the tilted mirror, either
            `'sagittal'` or `'tangential'`
        :type roc: float or None
        :type aoi: float or None
        :type orientation: str
        """
        super(MirrorRM, self).__init__()
        if roc is None:
            self.matrix = np.matrix([[1, 0], [0, 1]], dtype=np.float64)
        elif aoi is None:
            self.matrix = np.matrix([[1, 0], [-2/roc, 1]], dtype=np.float64)
        else:
            if orientation == 'sagittal':
                self.matrix = np.matrix([[1, 0], [-2/roc * np.cos(aoi), 1]], dtype=np.float64)
            elif orientation == 'tangential':
                self.matrix = np.matrix([[1, 0], [-2/(roc * np.cos(aoi)), 1]], dtype=np.float64)
            else:
                raise ValueError("orientation should be either \'sagittal\' or \'tangential\'")
        self.dist_internal = 0
        # Strings
        self.type = "Mirror"
        self.parameters = [x + "{0:0.3g}".format(y) for x, y in
                           zip(("roc=", "aoi="), (roc, aoi)) if y is not None]
        if aoi is not None:
            self.parameters.append("orient=" + orientation)

class InterfaceRM(RayMatrix):
    """ RayMatrix for an interface which can optionally be curved and/or tilted
    """
    def __init__(self, ior_init, ior_fin, roc=None, aoi=None, orientation='sagittal'):
        """ RayMatrix for an interface which can optionally be curved and/or tilted

        This method creates the RayMatrix instance for a curved or flat
        interface which can be at normal incidence or at an angle.  If the
        optical axis has a non-zero angle of incidence with the mirror, then it
        is important to specify if the ray matrix is for the sagittal or
        tangential rays.

        For a typical optical system where the optical axis stays in a plane
        parallel to the surface of the table, then sagittal rays are those in
        the vertical direction (y axis) and tangential rays are those in the
        horizontal direction (x axis).

        :param ior_init: initial index of refraction
        :param ior_fin: final index of refraction
        :param roc: radius of curvature of the mirror (None for flat)
        :param aoi: angle of incidence of the optical axis with the mirror in
            radians
        :param orientation: orientation of the tilted mirror, either
            `'sagittal'` or `'tangential'`
        :type ior_init: float
        :type ior_fin: float
        :type roc: float or None
        :type aoi: float or None
        :type orientation: str
        """
        super(InterfaceRM, self).__init__()
        if roc is None:
            self.matrix = np.matrix([[1, 0], [0, ior_init/ior_fin]],
                                    dtype=np.float64)
        elif aoi is None:
            self.matrix = np.matrix([[1, 0], [(ior_init - ior_fin)/(roc * ior_fin), ior_init/ior_fin]],
                                    dtype=np.float64)
        else:
            nr = ior_fin/ior_init
            if orientation == 'sagittal':
                self.matrix = np.matrix([[1, 0],
                                         [(np.cos(aoi) - np.sqrt(nr**2 - np.sin(aoi)**2))/(roc * nr), 1/nr]],
                                        dtype=np.float64)
            elif orientation == 'tangential':
                self.matrix = np.matrix([[np.sqrt(nr**2 - np.sin(aoi)**2)/(nr * np.cos(aoi)), 0],
                                         [(np.cos(aoi) - np.sqrt(nr**2 - np.sin(aoi)**2)) /
                                          (roc * np.cos(aoi) * np.sqrt(nr**2 - np.sin(aoi)**2)),
                                           np.cos(aoi)/np.sqrt(nr**2 - np.sin(aoi)**2)]],
                                        dtype=np.float64)
            else:
                raise ValueError("orientation should be either \'sagittal\' or \'tangential\'")
        self.dist_internal = 0
        self.ior_init = ior_init
        self.ior_fin = ior_fin
        # Strings
        self.type = "Interface"
        self.parameters = [x + "{0:0.3g}".format(y) for x, y in
                           zip(("ior_init=", "ior_fin=", "roc=", "aoi="),
                               (ior_init, ior_fin, roc, aoi)) if y is not None]
        if aoi is not None:
            self.parameters.append("orient=" + orientation)

class RayMatrixSystem:
    """ A class for systems of ray matrices
    """
    def __init__(self, ray_matrices=None, positions=None):
        """ A class for systems of ray matrices

        Note that objects are added to the system and the intervening space is
        filled with translation ray matrices.  It is unnecessary to add the
        translation matrices by hand.

        Each argument is a list or tuple, and must be the same length.  The
        `ray_matrices` argument should be a list or tuple of RayMatrix
        instances, and the `positions` argument should be a list or tuple
        containing the position along the optical axis of each RayMatrix.

        :param ray_matrices: list of RayMatrix instances
        :param positions: list of positions for each RayMatrix instance
        :type ray_matrices: list of RayMatrix
        :type positions: list of float
        """
        # Check for Nones
        if ray_matrices is None:
            self.ray_matrices = None
            self.positions = None
        # Check if the objects are iterable
        elif not hasattr(ray_matrices, "__iter__"):
            if not issubclass(type(ray_matrices), RayMatrix):
                raise TypeError("ray_matrices argument should contain instances "
                                "of the RayMatrix class")
            try:
                positions = float(positions)
            except ValueError or TypeError:
                raise TypeError("positions should be a float if ray_matrices "
                                "is a RayMatrix")
            self.ray_matrices = [ray_matrices]
            self.positions = [positions]
        # If the objects are iterables, check that they are properly composed
        else:
            try:
                ray_matrices = list(ray_matrices)
                positions = list(positions)
            except TypeError:
                raise TypeError("ray_matrices and positions should be convertible to "
                                "lists")
            if not len(ray_matrices) == len(positions):
                raise TypeError("ray_matrices and positions should be the same length")
            for rm in ray_matrices:
                if not issubclass(type(rm), RayMatrix):
                    raise TypeError("ray_matrices should contain RayMatrix instances")
            try:
                positions = [float(x) for x in positions]
            except ValueError:
                raise TypeError("position elements should be convertible to floats")
            self.ray_matrices = ray_matrices
            self.positions = positions
        # Add empty ior attribute
        self.iors = None
        # Update
        self._update()

    def _update(self):
        """ Updates the instance once a new RayMatrix has been added

        Specifically, this method:
            1. Sorts the ray_matrices and positions attributes by position
            2. Calculates the index of refraction at all points in the system
        """
        if self.ray_matrices is not None:
            # Sort by position
            indices = sorted(range(len(self.positions)),
                             key=lambda x: self.positions[x])
            self.ray_matrices = [self.ray_matrices[i] for i in indices]
            self.positions = [self.positions[i] for i in indices]
            # Calculate indices of refraction
            first_found, iors = False, []
            current_ior, last_ior = None, None
            for rm in self.ray_matrices:
                # Initial ior
                if rm.ior_init is not None:
                    current_ior = rm.ior_init
                    if first_found:
                        if not last_ior == current_ior:
                            warnings.warn("output and input indices of refraction do "
                                          "not match for some elements")
                    first_found |= True
                else:
                    current_ior = last_ior
                iors.append(current_ior)
                # Final ior
                if rm.ior_fin is not None:
                    last_ior = rm.ior_fin
            iors.append(last_ior)
            if first_found:
                i, v = next((i, v) for i, v in enumerate(iors)
                            if v is not None)
                for jj in range(i):
                    iors[jj] = v
            else:
                iors = [1 for _ in iors]
            self.iors = iors

    ###########################################################################
    # System Representation
    ###########################################################################
    def print_summary(self, return_string=False):
        """ Prints a summary of the system to the command line

        If the `return_string` parameter is set to True, then the summary will
        be returned as a string instead of printed to the command line.

        :param return_string: if true, then the summary is returned as a string
        :type return_string: bool
        :return: if return_string is True, then a summary string is returned
        :rtype: str
        """
        # Labels
        col_labels = ["#", "Element", "Parameters", "z", "IOR"]
        # Determine lengths
        col_len = [len(str(len(self.positions)))]
        col_len.append(max(len(x.type) for x in self.ray_matrices))
        col_len.append(max(len(par) for rm in self.ray_matrices for par in rm.parameters))
        col_len.append(max(len("{0:0.3g}".format(x)) for x in self.positions))
        col_len.append(max(len("{0:0.3g}".format(x)) for x in self.iors))
        # Compare lengths to labels
        for i, v in enumerate(zip(col_len, col_labels)):
            col_len[i] = max((v[0], len(v[1])))
        # Build string
        out = "| " + " | ".join(clab.center(clen) for clab, clen in
                                zip(col_labels, col_len)) + " |\n"
        out += "=" * (sum(col_len) + 16) + "\n"
        for i, v in enumerate(zip(self.ray_matrices, self.positions)):
            rm, pos = v
            first_par = True
            out += "|-" + "-"*col_len[0] + "-|-" + "-"*col_len[1] +\
                   "-|-" + "-"*col_len[2] + "-|-" + "-"*col_len[3] + "-| " + \
                   "{0:0.3g}".format(self.iors[i]).ljust(col_len[4]) + " |\n"
            if rm.parameters:
                for par in rm.parameters:
                    if first_par:
                        first_par = False
                        out += "| " + "{0}".format(i).ljust(col_len[0]) + " | " + \
                               rm.type.ljust(col_len[1]) + " | " + \
                               par.ljust(col_len[2]) + " | " + \
                               "{0:0.3g}".format(pos).ljust(col_len[3]) + \
                               " | " + " " *col_len[4] + " |\n"
                    else:
                        out += "| " + " "*col_len[0] + " | " + " "*col_len[1] +\
                               " | " + par.ljust(col_len[2]) + " | " + \
                               " "*col_len[3] + " | " +  " " *col_len[4] + " |\n"
            else:
                out += "| " + "{0}".format(i).ljust(col_len[0]) + " | " + \
                       rm.type.ljust(col_len[1]) + " | " + \
                       " " * col_len[2] + " | " + \
                       "{0:0.3g}".format(pos).ljust(col_len[3]) + \
                       " | " + " " * col_len[4] + " |\n"
        out += "|-" + "-"*col_len[0] + "-|-" + "-"*col_len[1] +\
               "-|-" + "-"*col_len[2] + "-|-" + "-"*col_len[3] + "-| " + \
               "{0:0.3g}".format(self.iors[i+1]).ljust(col_len[4]) + " |\n"
        if return_string:
            return out
        else:
            print(out)


    ###########################################################################
    # Extract Matrices
    ###########################################################################
    @classmethod
    def _multiply_matrices(cls, ray_matrices, inverse=False):
        """ Multiplies ray matrices passed as a list of RayMatrix instances

        This method takes in a list or tuple of RayMatrix instances and returns
        a single matrix (type: np.matrix) which is the multiple of those
        matrices in the order specified.

        The inverse parameter can be used to produce the matrix for
        propagation in the opposite direction.  I.E.
        _multiply_matrices(rm) * _multiply_matrices(rm, inverse=1) gives the
        identity matrix

        :param ray_matrices: a list or tuple of RayMatrix instances to multiply
        :param inverse: inverts the matrix so that it can be used for
            propagation in the other direction
        :type ray_matrices: list of RayMatrix
        :return: the multiplied matrix
        :rtype: np.matrix
        """
        # Initialize identy matrix
        mat = np.matrix([[1, 0], [0, 1]], dtype=np.float64)
        # Loop through ray_matrices
        for rm in ray_matrices:
            mat = rm.matrix * mat
        # Backward
        if inverse:
            a = mat[0, 0]
            b = mat[0, 1]
            c = mat[1, 0]
            d = mat[1, 1]
            mat = 1 / (a * d - b * c) * np.matrix([[d, -b], [-c, a]])
        # Return
        return mat

    def _get_distances(self):
        """ Returns the important distances of the RayMatrix elements

        This method returns a list of lists with the internal distance and the
        distance to the next element for each element.

        :return: list with [internal distance, extra distance] entries
        :rtype: list of list
        """
        # Distance between elements
        dist = (a - b for a, b in zip(self.positions[1:], self.positions[:-1]))
        out = [[rm.dist_internal, d - rm.dist_internal] for rm, d in
               zip(self.ray_matrices[:-1], dist)]
        out += [[self.ray_matrices[-1].dist_internal, None]]
        return out

    def _get_deadzones(self):
        """ Returns the 'deadzones' where a ray matrix has internal distance

        :return: list with [begin dz, end dz] entries
        :rtype: list of list
        """
        dz = []
        for rm, pos in zip(self.ray_matrices, self.positions):
            if rm.dist_internal == 0:
                dz.append(None)
            else:
                dz.append([pos, pos + rm.dist_internal])
        return dz

    def get_matrix(self, z_from, z_to, inverse=False):
        """ Returns the matrix from `z_from` to `z_to`

        This method returns the matrix between any two points in the ray matrix
        system.  This matrix allows the user to propagate a ray between the two
        points.  Note that if `z_to` is less than `z_from`, then the matrix is
        such that the optical axis is reversed.  On the other hand, if the
        `inverse` parameter is set to True, then the matrix assumes that the
        optical axis still points in the same direction, but the propagation is
        in reverse.

        Note also that, for ray matrices which have an internal distance, the
        matrix will be the same for any z_from or z_to which falls within that
        range since the RayMatrixSystem does not have knowledge of how the ray
        transforms within the element.

        :param z_from: starting position along the optical axis
        :param z_to: ending position along the optical axis
        :param inverse: if True, then the inverse of the matrix is returned
        :type z_from: float
        :type z_to: float
        :type inverse: bool
        :return: ray matrix between the two points
        :rtype: np.matrix
        """
        # Is it backwards
        backwards = z_to < z_from
        if backwards:
            z_1, z_2 = z_to, z_from
        else:
            z_1, z_2 = z_from, z_to
        # Get included elements, positions, and internal distances
        els = [i for i, v in enumerate(self.positions) if z_1 < v < z_2]
        dists = self._get_distances()
        dzs = self._get_deadzones()
        # Check if z_1 or z_2 are in a deadzone
        dz_1, delt_1 = None, 0
        dz_2 = None
        ii = 0
        for dz in dzs:
            if dz is None:
                ii += 1
                pass
            elif dz[0] < z_1 < dz[1]:
                delt_1 = dz[1] - z_1
                dz_1 = ii
            elif dz[0] < z_2 < dz[1]:
                dz_2 = ii
            ii += 1
        # Gather matrices
        # If no elements were included, then just one translation matrix
        if not els:
            if (dz_1 is not None) and (dz_2 is not None):
                mats = [TranslationRM(0)]
            elif dz_1 is not None:
                mats = [TranslationRM(z_2 - z_1 - delt_1)]
            else:
                mats = [TranslationRM(z_2 - z_1)]
        else:
            # Add first translation
            mats = [TranslationRM(self.positions[els[0]] - z_1 - delt_1)]
            # Append middle section
            for i, el in enumerate(els):
                mats.append(self.ray_matrices[el])
                if not i == len(els) - 1:
                    mats.append(TranslationRM(dists[el][1]))
            # Append final translation
            if dz_2 is None:
                mats.append(TranslationRM(z_2 - self.positions[els[-1]] -
                                          self.ray_matrices[els[-1]].dist_internal))
        # Build total matrix
        if backwards:
            mat = self._multiply_matrices(reversed(mats), inverse=inverse)
        else:
            mat = self._multiply_matrices(mats, inverse=inverse)
        return mat

    ###########################################################################
    # Add and Remove Elements
    ###########################################################################
    def _add_element(self, z, rm):
        """ Adds an element to the RayMatrixSystem

        :param rm: RayMatrix instance
        :param z: position along the optical axis
        :type rm: RayMatrix
        :type z: float
        """
        # Check type
        if not issubclass(type(rm), RayMatrix):
            raise TypeError("rm should be a RayMatrix instance")
        # Append to the RayMatrixSystem instance
        if self.ray_matrices is None:
            self.ray_matrices = [rm]
            self.positions = [z]
        else:
            self.positions.append(z)
            self.ray_matrices.append(rm)
        self._update()

    def remove_element(self, el_num):
        """ Removes an element from the RayMatrixSystem

        The el_num is the index of the element to remove from the system

        :param el_num: index of the element to remove
        :type el_num: int
        """
        del self.ray_matrices[el_num]
        del self.positions[el_num]
        self._update()

    def add_thin_lens(self, z, f):
        """ Add a thin lens element to the RayMatrixSystem

        :param f: focal length
        :param z: position along the optical axis
        :type f: float
        :type z: float
        """
        rm = ThinLensRM(f)
        self._add_element(z, rm)

    def add_prism(self, z, n_air, n_mat, theta1, alpha, s):
        """ Add a prism element to the RayMatrixSystem

        Prisms have the ability to shape the beam, and can be used as beam
        shaping devices along one of the axes.  This method returns the ray
        matrix for a standard beam shaping prism.

        The index of refraction of the air, `n_air` (n_air=1 usually), and the
        material, `n_mat`, are the first two arguments.  The last three arguments
        are the input angle `theta1`, the opening angle of the prism, `alpha`, and
        the vertical distance from the apex of the prism, `s`.

        The matrix is taken from: Kasuya, T., Suzuki, T. & Shimoda, K. A prism
        anamorphic system for Gaussian beam expander. Appl. Phys. 17, 131–136
        (1978). and figure 1 of that paper has a clear definition of the
        different parameters.

        :param n_air: index of refraction of the air (usually 1)
        :param n_mat: index of refraction of the prism material
        :param theta1: input angle to the prism wrt the prism face
        :param alpha: opening angle of the prism (alpha=0 is equivalent to a
            glass plate)
        :param s: vertical distance from the apex of the prism.
        :param z: position along the optical axis of the prism input.
        :type n_air: float
        :type n_mat: float
        :type theta1: float
        :type alpha: float
        :type s: float
        :type z: float
        """
        rm = PrismRM(n_air, n_mat, theta1, alpha, s)
        self._add_element(z, rm)

    def add_mirror(self, z, roc=None, aoi=None, orientation='sagittal'):
        """ Adds a possibly curved/tilted mirror element to the RayMatrixSystem

        This method creates the RayMatrix instance for a curved or flat mirror
        which can be at normal incidence or at an angle.  If the optical axis
        has a non-zero angle of incidence with the mirror, then it is important
        to specify if the ray matrix is for the sagittal or tangential rays.

        For a typical optical system where the optical axis stays in a plane
        parallel to the surface of the table, then sagittal rays are those in
        the vertical direction (y axis) and tangential rays are those in the
        horizontal direction (x axis).

        :param z: position along optical axis
        :param roc: radius of curvature of the mirror (None for flat)
        :param aoi: angle of incidence of the optical axis with the mirror in
            radians
        :param orientation: orientation of the tilted mirror, either
            `'sagittal'` or `'tangential'`
        :type z: float
        :type roc: float or None
        :type aoi: float or None
        :type orientation: str
        """
        rm = MirrorRM(roc=roc, aoi=aoi, orientation=orientation)
        self._add_element(z, rm)

    def add_interface(self, z, ior_init, ior_fin, roc=None, aoi=None, orientation='sagittal'):
        """ RayMatrix for an interface which can optionally be curved and/or tilted

        This method creates the RayMatrix instance for a curved or flat
        interface which can be at normal incidence or at an angle.  If the
        optical axis has a non-zero angle of incidence with the mirror, then it
        is important to specify if the ray matrix is for the sagittal or
        tangential rays.

        For a typical optical system where the optical axis stays in a plane
        parallel to the surface of the table, then sagittal rays are those in
        the vertical direction (y axis) and tangential rays are those in the
        horizontal direction (x axis).

        :param z: position of the element along the optical axis
        :param ior_init: initial index of refraction
        :param ior_fin: final index of refraction
        :param roc: radius of curvature of the mirror (None for flat)
        :param aoi: angle of incidence of the optical axis with the mirror in
            radians
        :param orientation: orientation of the tilted mirror, either
            `'sagittal'` or `'tangential'`
        :type z: float
        :type ior_init: float
        :type ior_fin: float
        :type roc: float or None
        :type aoi: float or None
        :type orientation: str
        """
        rm = InterfaceRM(ior_init=ior_init, ior_fin=ior_fin,
                         aoi=aoi, orientation=orientation)
        self._add_element(z, rm)






