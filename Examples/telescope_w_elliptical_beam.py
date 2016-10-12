""" pylase example:  A simple two-element telescope with an elliptical beam

This example defines a simple two-element telescope and two beams, one for x and one for y, and
plots the beam size through the system.
"""

import numpy as np
from pylase.optical_system import OpticalSystem

# Create optical system with two lenses
os = OpticalSystem()
os.add_lens(f=50e-3, z=100e-3, label='lens 1')
os.add_lens(f=150e-3, z=300e-3, label='lens 2')

# Add two beams
os.add_beam(waist_size=0.1e-3, distance_to_waist=0, wvlnt=1064e-9, z=0, beam_label='x')
os.add_beam(waist_size=0.2e-3, distance_to_waist=0, wvlnt=1064e-9, z=0, beam_label='y')

# Plot
fig, ax = os.plot_w(zs=np.linspace(0, 500e-3, 500))
ax.set_title('Elliptical Beam in 3X Telescope')
fig.show()








