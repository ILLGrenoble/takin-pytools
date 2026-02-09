#!/usr/bin/env python3
#
# test for the minimalistic implementation of linear spin-wave theory (see: https://arxiv.org/abs/1402.6069)
# @author Tobias Weber <tweber@ill.fr>
# @date 9-feb-2026
# @license see 'LICENSE' file
# @see https://github.com/ILLGrenoble/magpie for full implementation and references
#

import lswt
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt


num_Q_points = 1024   # number of Q points to calculate
only_pos_E   = False  # hide magnon annihilation?
verbose      = False  # debug output


def calc_dispersion():
	#
	# set up magnetic sites and couplings:
	#
	#   - magnetic sites:
	#     "S": spin magnitude
	#     "Sdir": spin direction
	#
	#   - magnetic couplings:
	#     "sites": indices of the sites to couple
	#     "J": (symmetric) exchange interaction
	#     "DMI": (antisymmetric) Dzyaloshinskii-Moryia interaction
	#     "gen": arbitrary interaction matrix, can be used for single-ion anisotropy
	#     "dist": distance in rlu to the next unit cell for the coupling
	#
	title = "Ferromagnetic, Non-Reciprocal"
	sites = [
		{ "S" : 1., "Sdir" : [ 0, 0, 1 ] },
	]

	sia = np.zeros([3, 3])
	sia[2, 2] = -0.06
	couplings = [
		{ "sites" : [ 0, 0 ], "J" : -1., "DMI" : [ 0, 0, 0.25 ], "gen" : sia, "dist" : [ 1, 0, 0 ] },
	]

	lswt.init(sites, couplings, verbose)

	# plot a dispersion branch
	hs = []
	Es = []
	for h in np.linspace(-1, 1, num_Q_points):
		try:
			Qvec = np.array([ h, 0, 0 ])
			for E in lswt.get_energies(Qvec, sites, couplings):
				if only_pos_E and E < 0.:
					continue
				hs.append(h)
				Es.append(E)
		except la.LinAlgError:
			pass

	plt.plot()
	plt.xlabel("h (rlu)")
	plt.ylabel("E (meV)")
	plt.scatter(hs, Es, marker = '.', label = title)


print("Calculating non-reciprocal ferromagnetic dispersion...")
calc_dispersion()

print("Plotting...")
plt.legend(loc = "upper right")
plt.tight_layout()
plt.show()
