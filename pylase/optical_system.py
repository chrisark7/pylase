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
from copy import deepcopy
import warnings
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
        self._beam_hash = 0
        # Initialize the elements list and the elements hash
        self.elements = [optical_element._EmptySystemEL()]
        self._el_hash = 0
        # Initialize the RayMatrixSystem and q parameters
        self._update()

    def copy(self):
        """ Creates a deep copy of the OpticalSystem instance

        This copy of the Optical System instance may have its attributes
        (elements, beams, positions, etc.) modified without affecting the
        original OpticalSystem instance since it is a 'deep copy'

        :return: Copy of the OpticalSystem object
        :rtype: OpticalSystem
        """
        return deepcopy(self)

    def summary(self, return_string=False):
        """ Generates a system summary and prints or returns the string

        If `return_string` evaluates to False, then the system summary is
        printed to the screen.  Otherwise, it is returned to the user.

        :param return_string: Prints to screen if False, returns if True
        :type return_string: bool
        :return: Returns the summary string if `return_string` is True else None
        :rtype: str or NoneType
        """
        if return_string:
            return self.rms.print_summary(return_string=return_string)
        else:
            self.rms.print_summary(return_string=return_string)


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
        # Update
        self._update()

    def _remove_beam(self, label):
        """ Removes a Beam instance from the list of beams

        :param label: The string label of the beam to remove
        :type label: str
        """
        # Get index
        bm_ind = self._get_beamindex(label)
        # Remove beam
        del self.beams[bm_ind]
        # Update
        self._update()

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
        # Update the internal optical system
        self._update()

    def _remove_element(self, label):
        """ Removes an OpticalElement instance from the list of elements

        :param label: The string label for the element to remove
        :type label: str
        """
        # Get index
        el_ind = self._get_elindex(label)
        # Remove element
        del self.elements[el_ind]
        # Update
        self._update()

    def _update(self):
        """ Updates the internal optical system when changes are made

        This routine is run whenever the system is modified to keep the
        internally held optical system up to date with the optical elements.
        Namely, it regenerates the RayMatrixSystem and updates the q
        parameters at the key locations.
        """
        # If elements has more than 1, remove the null element if it exists
        if len(self.elements) > 1:
            try:
                null_ind = next(i for i, v in enumerate(self.elements) if
                                type(v) is optical_element._EmptySystemEL)
                del self.elements[null_ind]
            except StopIteration:
                pass
        # Add the null element if there are no elements left
        elif len(self.elements) == 0:
            self.elements = [optical_element._EmptySystemEL()]
            self.elements = [optical_element._EmptySystemEL()]
        # If the element hash has changed, update elements and beams
        if not hash(tuple(self.elements)) == self._el_hash:
            self.rms = self._calc_ray_matrix_system()
            if self.beams:
                self.all_qs = {beam.label: self._calc_all_qs(beam)
                               for beam in self.beams}
            else:
                self.all_qs = {}
            # Update hashes
            self._el_hash = hash(tuple(self.elements))
            self._beam_hash = hash(tuple(self.beams))
        # If only the beam hash has changed, update beams
        elif not hash(tuple(self.beams)) == self._beam_hash:
            if self.beams:
                self.all_qs = {beam.label: self._calc_all_qs(beam)
                               for beam in self.beams}
            else:
                self.all_qs = {}
            # Update hash
            self._beam_hash = hash(tuple(self.beams))

    def _get_beamindex(self, beam_label):
        """ Returns the index of the beam for a given beam label

        :param beam_label: The label used to identify the beam
        :type beam_label: str
        :return: beam index
        :rtype: int
        """
        assert type(beam_label) is str
        try:
            beam_ind = next((i for i, v in enumerate(self.beams)
                             if v.label == beam_label))
        except StopIteration:
            raise ValueError("beam_label {0} does not match any beams".format(
                beam_label))
        return beam_ind

    def _get_elindex(self, el_label):
        """ Returns the index of the element for a given element label

        :param el_label: The label used to identify the element
        :type el_label: str
        :return: beam index
        :rtype: int
        """
        assert type(el_label) is str
        try:
            el_ind = next((i for i, v in enumerate(self.elements)
                           if v.label == el_label))
        except StopIteration:
            raise ValueError("el_label {0} does not match any elements".format(
                el_label))
        return el_ind

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
        if not self.elements:
            return ray_matrix.RayMatrixSystem()
        else:
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
        ind_from = bisect_left([x[0] for x in mats], beam.position)
        # Propagate q parameter to nearest location
        if ind_from == 0:
            beam = beam + mats[ind_from][0] - beam.position
        else:
            beam = beam + mats[ind_from-1][0] - beam.position
        # Calculate q parameters at all locations
        q_out = []
        for ind_to in range(0, len(mats)+1):
            # Get composite ray matrix
            if ind_from > ind_to:
                mat = self.rms.get_matrix(z_from=ind_to,
                                          z_to=ind_from,
                                          inverse=True,
                                          pos_num=True)
            else:
                mat = self.rms.get_matrix(z_from=ind_from,
                                          z_to=ind_to,
                                          inverse=False,
                                          pos_num=True)
            # Calculate new q parameter
            q_out.append(self._calc_prop_q(beam, mat))
        # Return
        return q_out

    ###########################################################################
    # Add, Remove, Modify Optical Elements
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

    def add_element_thick_lens(self, z, label, r1, r2, t, ior_lens, ior_air=1):
        """ Adds a thick lens to the elements of the optical system

        The input and output radii of curvature are specified such that a
        negative curvature is concave looking along the beam line while a
        positive curvature is convex.  I.E. a negative input curvature will
        result in a negative lens while a positive input curvature will result
        in a positive lens.  The output curvature, on the other hand, works in
        the opposite way; a negative output curvature will result in a positive
        lens while a positive output curvature will result in a negative lens.
        A flat interface can be specified by setting `r1` or `r2` to `None`

        Note that the position associated with the lens is the position of the
        input surface.  I.E. the position of the input face is `z` while the
        position of the output face is `z+t`.  The specified thickness of the
        lens should be the thickness along the optical axis, i.e. the center
        thickness.

        All dimensions are nominally in meters.

        :param z: The position of the input face along the optical axis
        :param label: A string label associated with the thick lens
        :param r1: The input curvature in meters, negative = concave, None = flat
        :param r2: The output curvature in meters, negative = concave, None = flat
        :param t: The thickness of the lens in meters
        :param ior_lens: The index of refraction of the lens material
        :param ior_air: The index of refraction of the surrounding medium
        :type z: float
        :type label: str
        :type r1: float or NoneType
        :type r2: float or NoneType
        :type t: float
        :type ior_lens: float
        :type ior_air: float
        """
        self._add_element(optical_element.ThickLensEL(z=z,
                                                      label=label,
                                                      r1=r1,
                                                      r2=r2,
                                                      t=t,
                                                      ior_lens=ior_lens,
                                                      ior_air=ior_air))

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
    def add_element_null(self, z, label):
        """ Adds a null element to the optical system

        Null elements can be useful as place-holders for components in the
        system which otherwise have no effect.  They will appear as a label
        on the plots generated from the optical system, but will have no
        effect on the beam.

        :param z: position along the optical axis
        :param label: string label for the interface
        :type z: float
        :type label: str
        """
        # Add the element to the list
        self._add_element(optical_element.NullEL(z=z,
                                                 label=label))

    def adjust_element_position(self, z, label):
        """ Adjusts the position of the optical element specified by `label`

        This method adjusts the position of an optical element.  The new
        position is given by parameter `z`, and the optical element is
        specified by `label`.

        :param z: position along the optical axis
        :param label: string label for the interface
        :type z: float
        :type label: str
        """
        assert issubclass(type(z), float) or issubclass(type(z), int)
        # Get element index
        el_ind = self._get_elindex(el_label=label)
        # Set new position
        self.elements[el_ind].position = z
        # Update system
        self._el_hash = 0
        self._update()

    def remove_element(self, label):
        """ Removes an optical element from the system

        This method removes an optical element from the optical system.  The
        element is specified by its string label.

        :param label: the string label of the element to be removed
        :type label: str
        """
        self._remove_element(label=label)

    def get_element(self, label):
        """ Returns the element specified by `label`

        :param label: string label for the OpticalElement
        :type label: str
        :return: The OpticalElement instance associated with `label`
        :rtype: optical_element.OpticalElement
        """
        return self.elements[self._get_elindex(el_label=label)]

    ###########################################################################
    # Add, Remove, Modify Beams
    ###########################################################################
    def add_beam_from_parameters(self, z, label, beam_size, distance_to_waist,
                                 wvlnt):
        """ Adds a beam to the optical system using common parameters

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

    def add_beam_from_q(self, z, label, q, wvlnt=None):
        """ Adds a beam to the optical system using the q parameter

        This method allows the user to add a beam using a q parameter.  The q
        parameter can either be an instance of the qParameter class or a
        complex number.  In the latter case the wavelength must be specified
        as well.  If `q` is an instance of the qParameter class, then the
        `wvlnt` parameter is ignored.

        :param z: location of the q parameter along the optical axis
        :param label: a string label associated with the beam
        :param q: either a qParameter instance or a complex number
        :param wvlnt: the wavelength of the radiation (ignored if `q` is a
            qParameter instance)
        :type z: float
        :type label: str
        :type q: q_param.qParameter or complex
        :type wvlnt: float
        """
        if type(q) is q_param.qParameter:
            self._add_beam(q_param.Beam(z, label, q=q, wvlnt=None))
        elif type(q) is complex:
            if wvlnt is None:
                raise ValueError("wvlnt should be specified for complex q")
            else:
                self._add_beam(q_param.Beam(z, label, q=q, wvlnt=wvlnt))

    def remove_beam(self, label):
        """ Removes a beam from the optical system

        This method removes a beam from the optical system instance.

        :param label: The label identifying the beam to remove
        :type label: str
        """
        self._remove_beam(label=label)

    ###########################################################################
    # System Property Calculations
    ###########################################################################
    def q(self, z, beam_label):
        """ Calculates the q parameter at a given point in the optical system

        This method calculates the q parameter at any point in the optical
        system.  The position is specified by `z`, and can be any point along
        the optical axis.  The `beam_label` parameter identifies which beam to
        propagate to the specified position.

        :param z: location along the optical axis
        :param beam_label: the label given to the beam of interest
        :type z: float
        :type beam_label: str
        :return: q parameter at location z
        :rtype: q_param.qParameter
        """
        # Get the position number and distance
        pos_num, dist = self.rms.get_pos_num_and_distance(z)
        # Get the q paremeter
        q = self.all_qs[beam_label][pos_num] + dist
        # Return
        return q

    def w(self, z, beam_label):
        """ Calculates the 1/e^2 beam radius at position z

        This method is one of the most commonly used methods in the
        OpticalSystem package.  It calculates the 1/e^2 beam radius at the
        position `z` along the optical axis.

        :param z: location along the optical axis
        :param beam_label: the label given to the beam of interest
        :type z: float
        :type beam_label: str
        :return: 1/e^2 beam radius at specified position
        :rtype: float
        """
        # Get the position number, distance, and ior
        pos_num, dist = self.rms.get_pos_num_and_distance(z)
        ior = self.rms.iors[pos_num]
        # Get the correct q parameter, add the distance, and scale by the ior
        q = (self.all_qs[beam_label][pos_num] + dist)/ior
        # Return
        return q.w(m2=1)

    def fit_q(self, guess_beam_label, data):
        """ Uses measured data to fit a beam starting with an initial guess

        This method fits a q parameter to measured data using the
        `scipy.optimize.minimize` routine.  The optical system needs to have a
        beam defined which this method will use as its initial guess.  The
        data should be passed in list of list format with the first element
        being the position along the optical axis, and the second being the
        beam size at that position, i.e.
           - data = [[z_1, w_z], [z_2, w_2], [z_3, w_3], ....]

        :param guess_beam_label: The beam_label of the beam which will be used
            as a guess
        :param data: The data to which the beam will be fit
        :type guess_beam_label: str
        :type data: list of list of float
        :return: The q parameter of the best fit
        :rtype: q_param.qParameter
        """
        # Check format of data
        try:
            zs = [x[0] for x in data]
            ws = [x[1] for x in data]
        except:
            raise TypeError('data should be a list of 2-element lists')
        # Scale beam
        guess_bm = self.beams[self._get_beamindex(guess_beam_label)]
        scale_w0 = guess_bm.w0()
        # Define fitting function
        def fit_fun(params):
            """ params = (w, z)
            """
            # Waist size can't be less than zero
            if params[0] < 0:
                params[0] = 1e-6*scale_w0
            # Add the beam
            self.add_beam_from_parameters(z=guess_bm.position,
                                          label='fitting_beam',
                                          beam_size=params[0] * scale_w0,
                                          distance_to_waist=params[1],
                                          wvlnt=guess_bm.get_wvlnt())
            # Calculate sum of squares of differences
            ssq = sum(((self.w(z=z, beam_label='fitting_beam') - w)**2
                       for z, w in zip(zs, ws)))
            # Remove beam
            self._remove_beam(label='fitting_beam')
            return ssq
        # Minimize the sum of squares
        res = minimize(fit_fun, (1, guess_bm.z()), tol=1e-5, method='Nelder-Mead')
        if res.success:
            print(res.message)
        else:
            warnings.warn('Optimization failed with message: {0}'.format(
                res.message))
        # Return a q parameter
        out_q = q_param.qParameter(wvlnt=guess_bm.get_wvlnt())
        out_q.set_q(beamsize=res.x[0]*scale_w0, position=res.x[1])
        return out_q

    ###########################################################################
    # Graphics
    ###########################################################################
    def plot_w(self, zs, fig_num=None, other_sys=None):
        """ Plots the beam size for all beams at the points given in `zs`

        :param zs: the positions along the optical axis at which to plot the
            beams size
        :param fig_num: the figure number to use (next available if None)
        :param other_sys: another optical system or a list of other optical
            systems
        :type zs: list or np.ndarray
        :type fig_num: int or None
        :type other_sys: OpticalSystem or list of OpticalSystem
        :return: (figure_handle, axis_handle)
        :rtype: (plt.figure, plt.axis)
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
                    raise TypeError('other_sys should be an OpticalSystem '
                                    'instance or a list of OpticalSystem instances')
        # Get beam sizes
        beam_labels, ws, elements = [], [], []
        for sys in systems:
            beam_label_n = [beam.label for beam in sys.beams]
            beam_labels.append(beam_label_n)
            elements.append(sys.elements)
            ws_n = {}
            for beam_label in beam_label_n:
                ws_n[beam_label] = [sys.w(z=z, beam_label=beam_label) for z in zs]
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
        fig.clf()
        # Grid
        grd0 = GridSpec(1, 1)
        grd0.update(left=0.1, right=0.90, bottom=0.1, top=0.93,
                    hspace=0.30, wspace=0.28)
        # Initialize axes
        ax0 = fig.add_subplot(grd0[:, :])
        xlims = [np.min(zs), np.max(zs)]
        # Plot
        for ws_n, beam_label_n in zip(ws, beam_labels):
            for beam_label in beam_label_n:
                ln = ax0.plot(zs, [scale * w for w in ws_n[beam_label]], lw=2, label=beam_label)
                ax0.plot(zs, [-1 * scale * w for w in ws_n[beam_label]], lw=2, color=ln[0].get_color())
        ax0.grid(True)
        ax0.set_xlabel('Position [m]')
        ax0.set_ylabel('Beam Radius [{0}]'.format(unit))
        ax0.set_xlim(xlims)
        ax0.legend(loc='best', fontsize=12)
        # Get scales for y
        ylims = ax0.get_ylim()
        dx, dy = xlims[1] - xlims[0], ylims[1] - ylims[0]
        for elements_n in elements:
            for el in elements_n:
                pos, label = el.position, el.label
                ax0.plot([pos, pos], ylims, color='black', lw=1, ls='--')
                ax0.text(pos, ylims[1] - 0.03 * dy, label, rotation=-90)
        ax0.set_xlim(xlims)
        ax0.set_ylim(ylims)
        # Return
        return fig, ax0

