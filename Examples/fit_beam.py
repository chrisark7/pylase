""" pylase example: Fiting a beam to some measured data

This example illustrates fitting a beam to measured data using the `fit_q`
method of the OpticalSystem class.  This example uses an empty optical system
since the data was taken without any intervening elements, but it is just as
valid with optical elements in between some of the measurements.
"""

from pylase.optical_system import OpticalSystem
from numpy import linspace

__author__ = "Chris Mueller"
__status__ = "Example"


# Parameters and data
guess_w0, guess_z = 200e-6, 2
wvlnt = 355e-9

data = [[0.144, 226e-6],
        [0.290, 287e-6],
        [0.442, 362e-6]]

# Build OpticalSytem instance
os = OpticalSystem()
os.add_beam_from_parameters(z=0,
                            label='guess',
                            beam_size=guess_w0,
                            distance_to_waist=guess_z,
                            wvlnt=wvlnt)

# Fit and add fitted beam to system
q_fit = os.fit_q(guess_beam_label='guess', data=data)
os.add_beam_from_q(z=0, label='fit', q=q_fit)


# Plot
fig, ax = os.plot_w(zs=linspace(0, 0.6, 200), fig_num=1)

# Add data
ax.hold(True)
ax.plot([x[0] for x in data], [x[1]*1e3 for x in data], 'xb', ms=10, lw=2)
ax.hold(False)
fig.show()





