""" pylase example: The effects of an off-center lens

This example file simulates an off-center lens as part of a two-element
telescope.  The off-center lens is simulated as a tilted curved surface before
a plane surface (plano-concave) where the tilt and distance between the
surfaces grow with decentering.

Interestingly, the beam-shaping effect (due to the difference in angles between
input and output surfaces) plays a larger role than the focal length shift.
"""

import numpy as np
from pylase.optical_system import OpticalSystem

__author__ = "Chris Mueller"
__status__ = "Example"


## Parameters #################################################################
roc = -23.1e-3
center_thickness = 1.5e-3
decentering = [0e-3, 5e-3, 10e-3]
n_lens = 1.462

w_x = 0.15e-3
w_y = w_x
wvlnt = 1064e-9

## Quick calculations #########################################################
roc_a = abs(roc)

# lens1 position
f_lens1 = roc/(n_lens - 1)
z_lens1 = 3*roc_a

# lens2 focal length and position
f_lens2 = abs(f_lens1) * 3
z_lens2 = z_lens1 + 2*abs(f_lens1) if roc < 0 else 4*abs(f_lens1)

## Create multiple systems ####################################################
sys_x = []
sys_y = []
for decenter in decentering:
    # Skip if decenter is too large
    if decenter > roc_a:
        continue
    # Initialize optical system
    os_x = OpticalSystem()
    os_y = OpticalSystem()
    # Calculate angle of incidence and thickness
    aoi_1 = np.arcsin(decenter/roc_a)
    aoi_2 = aoi_1 - np.arcsin(np.sin(aoi_1)/n_lens)
    extra_thickness = roc_a * (1 - np.cos(aoi_1))
    # Add elements
    os_x.add_element_interface(z=(z_lens1 - extra_thickness),
                               label='curved interface',
                               ior_init=1,
                               ior_fin=n_lens,
                               roc=roc,
                               aoi=aoi_1,
                               orientation='tangential')
    os_x.add_element_interface(z=z_lens1 + center_thickness,
                               label='flat interface',
                               ior_init=n_lens,
                               ior_fin=1,
                               roc=None,
                               aoi=aoi_2,
                               orientation='tangential')
    os_x.add_element_thin_lens(z=z_lens2,
                               label='collimating lens',
                               f=f_lens2)
    os_y.add_element_interface(z=z_lens1 - extra_thickness,
                               label='curved interface',
                               ior_init=1,
                               ior_fin=n_lens,
                               roc=roc,
                               aoi=aoi_1,
                               orientation='sagittal')
    os_y.add_element_interface(z=z_lens1 + center_thickness,
                               label='flat interface',
                               ior_init=n_lens,
                               ior_fin=1,
                               roc=None,
                               aoi=aoi_2,
                               orientation='sagittal')
    os_y.add_element_thin_lens(z=z_lens2,
                               label='collimating lens',
                               f=f_lens2)
    # Add beams
    os_x.add_beam_from_parameters(z=0,
                                  label='x {0:0.3g}'.format(decenter),
                                  beam_size=w_x,
                                  distance_to_waist=0,
                                  wvlnt=wvlnt)
    os_y.add_beam_from_parameters(z=0,
                                  label='y {0:0.3g}'.format(decenter),
                                  beam_size=w_x,
                                  distance_to_waist=0,
                                  wvlnt=wvlnt)
    # Append to lists
    sys_x.append(os_x)
    sys_y.append(os_y)


## Plot #######################################################################
fig, ax = sys_x[0].plot_w(zs=np.linspace(0, z_lens2*3, 1000), other_sys=sys_x[1:] + sys_y)
fig.show()