""" Uses ray matrices and q parameters to calculate properties of a laser system

The Gaussian beam q parameter, encapsulated in the qParameter class, completely
defines a Gaussian laser beam in the paraxial limit.  Similarly, a system of
geometrical optical ray matrices, encapsulated in the RayMatrix class,
completely defines a series of optical system in the paraxial limit.  This
package brings the two of those together to allow the user to calculate
the properties of an optical system consisting of a number of optical elements
and a Gaussian laser beam.
"""

from bisect import bisect_left
import numpy as np
from pylase import q_param, ray_matrix, optical_element

__author__ = "Chris Mueller"
__status__ = "Development"


class OpticalSystem:
    """ A class for calculating properties of optical systems

    Internally, the class instances contain instances of the OpticalElement
    together with instances of the qParameter class.  Each of these objects is
    associated with a location within the optical system.

    Each time a new element is added, the qParameters at every key point in
    the system are re-calculated.  This dramatically increases the speed of
    later calculations.
    """
    def __init__(self):
        """ The constructor for the OpticalSystem class

        An instance of the OpticalSystem class contains
          - 0 or more instances of the OpticalElement class
          - 1 or more instances of the Beam class

        This constructor initializes an instance of the class, and the
        OpticalElements and Beams are added with construction methods defined
        within the class.
        """
        # Initialize the beams component and its hash
        self.beams = []
        self._beam_hash = hash(tuple(self.beams))
        # Initialize the elements list and the elements hash
        self.elements = []
        self._el_hash = hash(tuple(self.elements))
        # Initialize the RayMatrixSystem
        self.rms = ray_matrix.RayMatrixSystem()

    ###########################################################################
    # Internal Add/Remove Methods
    ###########################################################################
    def _add_beam(self, beam):
        """ Adds a Beam instance to the list of beams

        :param beam: An instance of the Beam class
        :type beam: q_param.Beam
        """
        assert type(beam) is q_param.Beam
        # Check that beam label does not match a prior beam label
        if any(beam.label == x.label for x in self.beams):
            raise ValueError("beam label is already in use")
        # Append beam to list
        self.beams.append(beam)

    def _add_element(self, element):
        """ Adds an OpticalElement instance to the list of elements

        :param element: An instance of the OpticalElement class
        :type element: optical_element.OpticalElement
        """
        assert isinstance(element, optical_element.OpticalElement)
        # Check that beam label does not match a prior beam label
        if any(element.label == x.label for x in self.elements):
            raise ValueError("element label is already in use")
        # Append element to list
        self.elements.append(element)
        # Sort list to maintain order
        self.elements.sort(key=lambda x: x.position)

    ###########################################################################
    # System Calculation
    ###########################################################################
    @staticmethod
    def _calc_prop_q(q, mat):
        """ Propagates a q parameter with an ABCD matrix

        :param q: q parameter
        :param mat: abcd matrix
        :type q: q_param.qParameter
        :type mat: np.matrix
        :return: new q parameter
        :rtype: q_param.qParameter
        """
        assert issubclass(type(q), q_param.qParameter)
        assert type(mat) is np.matrix
        q_old = q.get_q()
        q_new = (mat[0, 0] * q_old + mat[0, 1])/(mat[1, 0] * q_old + mat[1, 1])
        q_new = q_param.qParameter(q=q_new, wvlnt=q.get_wvlnt())
        return q_new

    def _calc_all_matrices(self):
        """ Extracts the ray matrices and their positions from the elements

        This method returns a list of 2-element entries with the position and
        ray matrix of every matrix in the optical system.  This differs from
        the information stored in self.elements because the optical elements
        can contain multiple ray matrices while this is every ray matrix in
        succession.  The returned list is sorted by position.

        :return: every ray matrix with its position
        :rtype: list of tuple
        """
        # Get all ray matrices and their positions
        out = []
        for el in self.elements:
            for pos, rm in zip(el.relative_positions, el.ray_matrices):
                out.append([el.position + pos, rm])
        # Sort the output
        out.sort(key=lambda x: x[0])
        return out

    def _calc_ray_matrix_system(self):
        """ Builds the RayMatrixSystem from the OpticalElements

        This method takes the individual ray matrices contained in
        `self.elements` (extracted using `_calc_all_matrices`) and uses them
        to build a RayMatrixSystem instance which can be used to calculate
        properties of the optical system.

        :return: a RayMatrix System representation of the optical system
        :rtype: ray_matrix.RayMatrixSystem
        """
        # Get the ray matrices
        mats = self._calc_all_matrices()
        return ray_matrix.RayMatrixSystem(ray_matrices=[x[1] for x in mats],
                                          positions=[x[0] for x in mats])

    def _calc_all_qs(self, beam):
        """ Propagates the q parameter to all critical locations in the system

        This method returns a list of q parameters which has length N+1 where
        `len(self._calc_all_matrices())` has length N.  The first q
        parameter is immediately prior to the first ray matrix while every
        other q parameter is immediately after the corresponding ray matrix.
        This allows any point in the system to be reached by simply adding
        (or subtracting) the appropriate amount of distance to the appropriate
        q parameter.

        :return: list of qParameters
        :rtype: list of q_param.qParameter
        """
        assert type(beam) is q_param.Beam
        # Get all ray matrices and identify location of q parameter
        mats = self._calc_all_matrices()
        ind = bisect_left((x[0] for x in mats), beam.position)
        # Propagate q parameter to nearest location
        q = beam.get_q()
        if ind == 0:
            q = q + mats[ind][0] - beam.position
        else:
            q = q + mats[ind-1][0] - beam.position

    ###########################################################################
    # Add Optical Elements
    ###########################################################################
    def add_element_thin_lens(self, z, label, f):
        """ Adds a thin lens with focal length `f` to the list of elements

        :param z: position
        :param label: string label for the element
        :param f: focal length
        :type z: float
        :type label: str
        :type f: float
        """
        # Add the element to the list
        self._add_element(optical_element.ThinLensEL(z=z, label=label, f=f))

    def add_element_mirror(self, z, label, roc=None, aoi=None,
                           orientation='sagittal'):
        """ Adds a mirror to the list of elements in the optical system

        Adds a curved or flat mirror which can be at normal incidence or at an
        angle.  If the optical axis has a non-zero angle of incidence with the
        mirror, then it is important to specify if the ray matrix is for the
        sagittal or tangential rays.

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
        # Add the element to the list
        self._add_element(optical_element.MirrorEL(z=z,
                                                   label=label,
                                                   roc=roc,
                                                   aoi=aoi,
                                                   orientation=orientation))

    def add_element_prism(self, z, label, n_air, n_mat, theta1, alpha, s):
        """ Adds a prism to the list of elements

        Prisms have the ability to shape the beam, and can be used as beam
        shaping devices along one of the axes.  This method adds a standard
        beam-shaping prism to the list of elements.

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
        # Add the element to the list
        self._add_element(optical_element.PrismEL(z=z,
                                                  label=label,
                                                  n_air=n_air,
                                                  n_mat=n_mat,
                                                  theta1=theta1,
                                                  alpha=alpha,
                                                  s=s))

    def add_element_interface(self, z, label, ior_init, ior_fin, roc=None,
                              aoi=None, orientation='sagittal'):
        """ Adds an interface to the list of elements

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
        # Add the element to the list
        self._add_element(optical_element.InterfaceEL(z=z,
                                                      label=label,
                                                      ior_init=ior_init,
                                                      ior_fin=ior_fin,
                                                      roc=roc,
                                                      aoi=aoi,
                                                      orientation=orientation))

    ###########################################################################
    # Add Beams
    ###########################################################################
    def add_beam_from_parameters(self, z, label, beam_size, distance_to_waist,
                                 wvlnt):
        """ Adds a beam to the optical system

        This method allows the user to add a beam with real-world parameters,
        namely the waist size and the distance to the waist.  Note that **all
        beam sizes are specified by the 1/e^2 radius**.  The `z` parameter
        specifies the location of the beam along the optical axis while the
        `distance_to_waist` parameter specifies the distance from that location
        to the beam's waist.

        :param z: position along optical axis at which beam is defined
        :param label: a string label to access the beam later
        :param beam_size: 1/e^2 radius of beam at location specified by `z`
        :param distance_to_waist: distance between waist and location at which
            beam is defined. A negative number means the waist is farther along
            the optical axis.
        :param wvlnt: the wavelength of the light
        :type z: float
        :type label: str
        :type beam_size: float
        :type distance_to_waist: float
        :type wvlnt: float
        """
        # Create Beam instance
        beam = q_param.Beam(position=z, label=label, wvlnt=wvlnt)
        # Set the q parameter
        beam.set_q(beamsize=beam_size, position=distance_to_waist)
        # Add the beam to the list
        self._add_beam(beam)
