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

class RayMatrixSystem:
    """ A class for systems of ray matrices
    """
    def __init__(self, ray_matrices=None, positions=None):
        """ A class for systems of ray matrices

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
            self.ray_matrices = [rm for (pos, rm) in
                                 sorted(zip(self.positions, self.ray_matrices))]
            self.positions = list(sorted(self.positions))
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
                iors.append(current_ior)
                # Final ior
                if rm.ior_fin is not None:
                    last_ior = rm.ior_fin
            iors.append(last_ior)
            if first_found:
                i, v = next((i, v) for i, v in enumerate(iors) if v is not None)
                for jj in range(i):
                    iors[jj] = v
            else:
                iors = [1 for _ in iors]
            self.iors = iors