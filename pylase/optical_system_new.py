""" Uses ray matrices and q parameters to calculate properties of a laser system

The Gaussian beam q parameter, encapsulated in the qParameter class, completely
defines a Gaussian laser beam in the paraxial limit.  Similarly, a system of
geometrical optical ray matrices, encapsulated in the RayMatrix class,
completely defines a series of optical system in the paraxial limit.  This
package brings the two of those together to allow the user to calculate
the properties of an optical system consisting of a number of optical elements
and a Gaussian laser beam.
"""

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

    ###########################################################################
    # Internal Add/Remove Methods
    ###########################################################################
    def _add_beam(self, beam):
        """ Adds a Beam instance to the list of beams

        :param beam: An instance of the Beam class
        :type beam: q_param.Beam
        """
        assert type(beam) is q_param.Beam
        # Append beam to list
        self.beams.append(beam)

    def _add_element(self, element):
        """ Adds an OpticalElement instance to the list of elements

        :param element: An instance of the OpticalElement class
        :type element: optical_element.OpticalElement
        """
        assert isinstance(element, optical_element.OpticalElement)
        # Append element to list
        self.elements.append(element)
        # Sort list to maintain order
        self.elements.sort(key=lambda x: x.position)

    ###########################################################################
    # System Calculation
    ###########################################################################

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
        beam = q_param.Beam(z=z, label=label, wvlnt=wvlnt)
        # Set the q parameter
        beam.set_q(beamsize=beam_size, position=distance_to_waist)
        # Add the beam to the list
        self._add_beam(beam)
