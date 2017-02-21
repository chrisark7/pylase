""" pylase example:  A simple two-element telescope with an elliptical beam

This example defines a simple two-element telescope and two beams, one for x and one for y, and
plots the beam size through the system.
"""

from numpy import linspace
from pylase.optical_system import OpticalSystem

__author__ = "Chris Mueller"
__status__ = "Example"


# Create optical system with two lenses
os = OpticalSystem()
os.add_element_thin_lens(z=100e-3, label='lens 1', f=50e-3)
os.add_element_thin_lens(z=300e-3, label='lens 2', f=150e-3)

# Add two beams
os.add_beam_from_parameters(z=0,
                            label='x',
                            beam_size=0.1e-3,
                            distance_to_waist=0,
                            wvlnt=1064e-9)
os.add_beam_from_parameters(z=0,
                            label='y',
                            beam_size=0.2e-3,
                            distance_to_waist=0,
                            wvlnt=1064e-9)

# Plot
fig, ax = os.plot_w(zs=linspace(0, 500e-3, 500))
ax.set_title('Elliptical Beam in 3X Telescope')
fig.show()
