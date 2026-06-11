#!/usr/bin/env python3
#
# demo calculating the dispersion for CSO
#
# @author Tobias Weber <tweber@ill.fr>
# @date 11-jun-2026
# @license see 'LICENSE' file
#
# The magnetic model for CSO and its parameters are from this paper:
# https://doi.org/10.1103/PhysRevB.101.144411
# (which is also available here: https://arxiv.org/abs/2002.06283)
#

import lswt
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt


num_Q_points = 512    # number of Q points to calculate
only_pos_E   = True   # hide magnon annihilation?
verbose      = False  # debug output
weight_scale = 8.


# exchange constants
J1, J2        = -0.58, -0.93
D1x, D1y, D1z =  0.15, -0.24, -0.05
D2x, D2y, D2z =  0.16,   0.1,  0.36

# magnetic sites
sites = [
	{ "S" : 1, "Sdir" : [ 0, 0, 1 ], "pos" : [ 0.5,   0, 0.5 ] },
	{ "S" : 1, "Sdir" : [ 0, 0, 1 ], "pos" : [ 0,   0.5, 0.5 ] },
	{ "S" : 1, "Sdir" : [ 0, 0, 1 ], "pos" : [ 0.5, 0.5,   0 ] },
	{ "S" : 1, "Sdir" : [ 0, 0, 1 ], "pos" : [ 0,    0,    0 ] },
]

# magnetic couplings
couplings = [
	{ "sites" : [ 2, 3 ], "dist" : [  0,  1,  0 ], "J" : (J1), "DMI" : [ -D1x, -D1y,  D1z ] },
	{ "sites" : [ 3, 2 ], "dist" : [ -1, -1,  0 ], "J" : (J1), "DMI" : [ -D1x,  D1y, -D1z ] },
	{ "sites" : [ 1, 2 ], "dist" : [  0,  0,  1 ], "J" : (J1), "DMI" : [ -D1y, -D1z,  D1x ] },
	{ "sites" : [ 0, 3 ], "dist" : [  1,  0,  0 ], "J" : (J1), "DMI" : [ -D1y,  D1z, -D1x ] },
	{ "sites" : [ 3, 1 ], "dist" : [  0, -1, -1 ], "J" : (J1), "DMI" : [ -D1z, -D1x,  D1y ] },
	{ "sites" : [ 2, 0 ], "dist" : [  0,  1,  0 ], "J" : (J1), "DMI" : [ -D1z,  D1x, -D1y ] },
	{ "sites" : [ 1, 3 ], "dist" : [  0,  0,  1 ], "J" : (J1), "DMI" : [  D1z, -D1x, -D1y ] },
	{ "sites" : [ 0, 2 ], "dist" : [  0,  0,  0 ], "J" : (J1), "DMI" : [  D1z,  D1x,  D1y ] },
	{ "sites" : [ 3, 0 ], "dist" : [ -1,  0, -1 ], "J" : (J1), "DMI" : [  D1y, -D1z, -D1x ] },
	{ "sites" : [ 2, 1 ], "dist" : [  0,  0,  0 ], "J" : (J1), "DMI" : [  D1y,  D1z,  D1x ] },
	{ "sites" : [ 0, 1 ], "dist" : [  1,  0,  0 ], "J" : (J1), "DMI" : [  D1x, -D1y, -D1z ] },
	{ "sites" : [ 1, 0 ], "dist" : [  0,  0,  0 ], "J" : (J1), "DMI" : [  D1x,  D1y,  D1z ] },
	{ "sites" : [ 0, 3 ], "dist" : [  0,  0,  1 ], "J" : (J2), "DMI" : [ -D2x, -D2y,  D2z ] },
	{ "sites" : [ 1, 2 ], "dist" : [ -1,  0,  0 ], "J" : (J2), "DMI" : [ -D2x,  D2y, -D2z ] },
	{ "sites" : [ 0, 2 ], "dist" : [  0, -1,  1 ], "J" : (J2), "DMI" : [ -D2y, -D2z,  D2x ] },
	{ "sites" : [ 1, 3 ], "dist" : [  0,  1,  0 ], "J" : (J2), "DMI" : [ -D2y,  D2z, -D2x ] },
	{ "sites" : [ 0, 1 ], "dist" : [  0, -1,  0 ], "J" : (J2), "DMI" : [ -D2z, -D2x,  D2y ] },
	{ "sites" : [ 1, 0 ], "dist" : [ -1,  1,  0 ], "J" : (J2), "DMI" : [ -D2z,  D2x, -D2y ] },
	{ "sites" : [ 2, 3 ], "dist" : [  1,  0,  0 ], "J" : (J2), "DMI" : [  D2z, -D2x, -D2y ] },
	{ "sites" : [ 3, 2 ], "dist" : [  0,  0,  0 ], "J" : (J2), "DMI" : [  D2z,  D2x,  D2y ] },
	{ "sites" : [ 2, 0 ], "dist" : [  0,  0, -1 ], "J" : (J2), "DMI" : [  D2y, -D2z, -D2x ] },
	{ "sites" : [ 3, 1 ], "dist" : [  0,  0,  0 ], "J" : (J2), "DMI" : [  D2y,  D2z,  D2x ] },
	{ "sites" : [ 2, 1 ], "dist" : [  1,  0, -1 ], "J" : (J2), "DMI" : [  D2x, -D2y, -D2z ] },
	{ "sites" : [ 3, 0 ], "dist" : [  0,  0,  0 ], "J" : (J2), "DMI" : [  D2x,  D2y,  D2z ] },
]

print("Calculating CSO dispersion...")
lswt.init(sites, couplings, verbose)

# plot a dispersion branch
hs, Es, ws = [], [], []
for h in np.linspace(-1, 1, num_Q_points):
	try:
		Qvec = np.array([ h, 0, 0 ])
		for E, w in zip(*lswt.get_energies(Qvec, sites, couplings)):
			if only_pos_E and E < 0.:
				continue
			hs.append(h)
			Es.append(E)
			ws.append(w * weight_scale)
	except la.LinAlgError:
		pass

plt.plot()
plt.xlabel("h (rlu)")
plt.ylabel("E (meV)")
plt.scatter(hs, Es, marker = '.', s = ws, label = "CSO")


print("Plotting...")
plt.legend(loc = "upper right")
plt.tight_layout()
plt.show()
