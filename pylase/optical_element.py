""" Creates a class structure for optical elements

The key difference between an optical element and a ray matrix is that an
optical element can consist of several ray matrices which together act
as a single element.  The most common example is that of a thick lens which
consists of two curved interfaces and an intervening translation.

The optical element class contains a position and label for the optical element
as well as a list of the ray matrices which make up that element and a list of
their positions relative to the element's overall position.
"""

from pylase import ray_matrix


class OpticalElement:
    """ A class for the elements of an OpticalSystem instance

    """
    def __init__(self, ray_matrices, relative_positions, position, label):
        """ The constructor

        An OpticalElement can consist of more than one RayMatrix, the most
        common example is a thick lens.  When defining the element, it is still
        unnecessary to explicitly call out the translation matrices.

          * ray_matrices: A list of instances from the RayMatrix class.  This
            should be a list even if there is only 1 RayMatrix
          * relative_positions: A list of floats which specify the position of
            the individual pieces relative to the `position` of the entire
            element.  Should be the same length as `ray_matrices`
          * position: The position along the optical axis of the entire element
          * label: A string label for the element

        :param ray_matrices: A list of instances of the RayMatrix class
        :param relative_positions: A list of floats
        :param position:
        :param label:
        """
        # Type assertions
        assert type(label) is str
        assert type(ray_matrices) is list
        for rm in ray_matrices:
            assert issubclass(type(rm), ray_matrix.RayMatrix)
        assert len(ray_matrices) == len(relative_positions)
        assert type(position) in [float, int]
        assert type(label) is str
        # Assignment
        self.ray_matrices = ray_matrices
        self.relative_positions = relative_positions
        self.position = position
        self.label = label

    def __repr__(self):
        return "\'{0}\' @ {1:0.2g}".format(self.label, self.position)


class ThinLensEL(OpticalElement):
    """ Creates an OpticalElement instance for a thin lens
    """
    def __init__(self, z, label, f):
        """ OpticalElement for a thin lens with focal length `f`

        Note that a thick lens can be created from multiple interfaces

        :param z: position
        :param label: string label for the element
        :param f: focal length
        :type z: float
        :type label: str
        :type f: float
        """
        ray_matrices = [ray_matrix.ThinLensRM(f)]
        relative_positions = [0]
        super(ThinLensEL, self).__init__(ray_matrices, relative_positions, z, label)


class MirrorEL(OpticalElement):
    """ Creates an OpticalElement instance for a mirror
    """
    def __init__(self, z, label, roc=None, aoi=None, orientation='sagittal'):
        """ OpticalElement for a mirror which can optionally be curved and/or tilted

        This method creates the OpticalElement instance for a curved or flat mirror
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
        ray_matrices = [ray_matrix.MirrorRM(roc, aoi, orientation)]
        relative_positions = [0]
        super(MirrorEL, self).__init__(ray_matrices, relative_positions, z, label)


class PrismEL(OpticalElement):
    """ Creates an OpticalElement instance for a prism
    """
    def __init__(self, z, label, n_air, n_mat, theta1, alpha, s):
        """ OpticalElement for a prism

        Prisms have the ability to shape the beam, and can be used as beam
        shaping devices along one of the axes.  This method returns the
        Optical Element for a standard beam shaping prism.

        The index of refraction of the air, `n_air` (n_air=1 usually), and the
        material, `n_mat`, are the first two arguments.  The last three arguments
        are the input angle `theta1`, the opening angle of the prism, `alpha`, and
        the vertical distance from the apex of the prism, `s`.

        The matrix is taken from: Kasuya, T., Suzuki, T. & Shimoda, K. A prism
        anamorphic system for Gaussian beam expander. Appl. Phys. 17, 131–136
        (1978). and figure 1 of that paper has a clear definition of the
        different parameters.

        :param z: position along the optical axis
        :param label: a string label for the prism
        :param n_air: index of refraction of the air (usually 1)
        :param n_mat: index of refraction of the prism material
        :param theta1: input angle to the prism wrt the prism face
        :param alpha: opening angle of the prism (alpha=0 is equivalent to a
            glass plate)
        :param s: vertical distance from the apex of the prism.
        :type z: float
        :type label: str
        :type n_air: float
        :type n_mat: float
        :type theta1: float
        :type alpha: float
        :type s: float
        """
        ray_matrices = [ray_matrix.PrismRM(n_air, n_mat, theta1, alpha, s)]
        relative_positions = [0]
        super(PrismEL, self).__init__(ray_matrices, relative_positions, z, label)


class InterfaceEL(OpticalElement):
    """ Creates an OpticalElement instance for an interface
    """
    def __init__(self, z, label, ior_init, ior_fin, roc=None, aoi=None,
                 orientation='sagittal'):
        """ OpticalElement for an interface which can optionally be curved and/or tilted

        This method creates the OpticalElement instance for a curved or flat
        interface which can be at normal incidence or at an angle.  If the
        optical axis has a non-zero angle of incidence with the mirror, then it
        is important to specify if the ray matrix is for the sagittal or
        tangential rays.

        For a typical optical system where the optical axis stays in a plane
        parallel to the surface of the table, then sagittal rays are those in
        the vertical direction (y axis) and tangential rays are those in the
        horizontal direction (x axis).

        :param z: position along the optical axis
        :param label: string label for the interface
        :param ior_init: initial index of refraction
        :param ior_fin: final index of refraction
        :param roc: radius of curvature of the mirror (None for flat)
        :param aoi: angle of incidence of the optical axis with the mirror in
            radians
        :param orientation: orientation of the tilted mirror, either
            `'sagittal'` or `'tangential'`
        :type z: float
        :type label: str
        :type ior_init: float
        :type ior_fin: float
        :type roc: float or None
        :type aoi: float or None
        :type orientation: str
        """
        ray_matrices = [ray_matrix.InterfaceRM(ior_init, ior_fin, roc=roc, aoi=aoi,
                 orientation=orientation)]
        relative_positions = [0]
        super(InterfaceEL, self).__init__(ray_matrices, relative_positions, z, label)

class NullEL(OpticalElement):
    """ Crates a null optical element
    """
    def __init__(self, z, label):
        """ Constructs a null optical element

        The null element acts as a placeholder, and does not alter the optical
        characteristics of the system.

        :param z: position along the optical axis
        :param label: string label for the interface
        :type z: float
        :type label: str
        """
        ray_matrices = [ray_matrix.NullRM()]
        relative_positions = [0]
        super(NullEL, self).__init__(ray_matrices, relative_positions, z, label)
