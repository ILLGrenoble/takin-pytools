#
# minimalistic implementation of linear spin-wave theory (see: https://arxiv.org/abs/1402.6069)
# @author Tobias Weber <tweber@ill.fr>
# @date 24-oct-2024
# @license see 'LICENSE' file
# @see https://github.com/ILLGrenoble/magpie for full implementation and references
#

import numpy as np, numpy.linalg as la
import itertools as iter

# debug output
verbose_print = False    # print intermediate results
def print_infos(str):
	if verbose_print: print(str)

# calculate magnetic structure properties
def init(sites, couplings, verbose = False):
	global verbose_print
	verbose_print = verbose

	# skew-symmetric (cross-product) matrix
	def skew(v):
		return np.array([[ 0., v[2], -v[1] ], [ -v[2], 0., v[0] ], [ v[1], -v[0], 0. ] ])
	
	# calculate spin rotations towards ferromagnetic order along [001]
	for site in sites:
		zdir = np.array([ 0., 0., 1. ])
		Sdir = np.array(site["Sdir"]) / la.norm(site["Sdir"])
		rotaxis = np.array([ 0., 1., 0. ])
		s = 0.

		if np.allclose(Sdir, zdir):
			c = 1.    # spin and z axis parallel
		elif np.allclose(Sdir, -zdir):
			c = -1.   # spin and z axis anti-parallel
		else:  # sine and cosine of the angle between spin and z axis
			rotaxis = np.cross(Sdir, zdir)
			s = la.norm(rotaxis)
			c = Sdir @ zdir
			rotaxis /= s

		# rotation via rodrigues' formula, see (Arens 2015), p. 718 and p. 816
		rot = (1. - c) * np.outer(rotaxis, rotaxis) + np.diag([ c, c, c ]) - skew(rotaxis)*s
		site["u"] = rot[0, :] + 1j * rot[1, :]
		site["v"] = rot[2, :]
		print_infos("\nrot = \n%s\nu = %s\nv = %s" % (rot, site["u"], site["v"]))

	for coupling in couplings:
		coupling["J_real"] = np.diag([ coupling["J"] ]*3)
		if "DMI" in coupling: coupling["J_real"] += skew(coupling["DMI"])
		if "gen" in coupling: coupling["J_real"] += coupling["gen"]
		print_infos("\nJ_real =\n%s" % coupling["J_real"])

# get the energies and neutron spectral weights of the dispersion at the momentum transfer Qvec
def get_energies(Qvec, sites, couplings):
	num_sites = len(sites)
	J_fourier = np.zeros((num_sites, num_sites, 3, 3), dtype = complex)
	J0_fourier = np.zeros((num_sites, num_sites, 3, 3), dtype = complex)

	for coupling in couplings:
		dist = np.array(coupling["dist"])
		J_real = coupling["J_real"]
		site1, site2 = coupling["sites"]

		J_ft = J_real * np.exp(-2j*np.pi * dist @ Qvec)
		J_fourier[site1, site2] += J_ft
		J_fourier[site2, site1] += J_ft.transpose().conj()
		J0_fourier[site1, site2] += J_real
		J0_fourier[site2, site1] += J_real.transpose().conj()
	print_infos("\n\nQ = %s\nJ_fourier =\n%s\n\nJ0_fourier =\n%s" % (Qvec, J_fourier, J0_fourier))

	H = np.zeros((2*num_sites, 2*num_sites), dtype = complex)
	for i, j in iter.product(range(num_sites), range(num_sites)):
		S_i, u_i, v_i = sites[i]["S"], sites[i]["u"], sites[i]["v"]
		S_j, u_j, v_j = sites[j]["S"], sites[j]["u"], sites[j]["v"]
		S = 0.5 * np.sqrt(S_i * S_j)

		H[            i,             j] +=  S   * u_i        @ J_fourier[i, j]  @ u_j.conj()
		H[            i, num_sites + j] +=  S   * u_i        @ J_fourier[i, j]  @ u_j
		H[num_sites + i,             j] += (S   * u_j        @ J_fourier[j, i]  @ u_i).conj()
		H[num_sites + i, num_sites + j] +=  S   * u_i.conj() @ J_fourier[i, j]  @ u_j
		H[            i,             i] -=  S_j * v_i        @ J0_fourier[i, j] @ v_j
		H[num_sites + i, num_sites + i] -=  S_j * v_i        @ J0_fourier[i, j] @ v_j

	C = la.cholesky(H)
	signs = np.diag(np.concatenate((np.repeat(1., num_sites), np.repeat(-1., num_sites))))
	H_trafo = C.transpose().conj() @ signs @ C
	Es, states = la.eigh(H_trafo)
	Es, states = np.flip(Es), np.flip(states, axis = 1)  # sort in descending order

	ops = la.inv(C).transpose().conj() @ states @ np.sqrt(signs @ states.transpose().conj() @ H_trafo @ states)
	S_mats = np.zeros((num_sites*2, 3, 3), dtype = complex)
	for x, y in iter.product(range(3), range(3)):
		M = np.zeros((2*num_sites, 2*num_sites), dtype = complex)
		for i, j in iter.product(range(num_sites), range(num_sites)):
			S_i, u_i = sites[i]["S"], -2. * sites[i]["u"]
			S_j, u_j = sites[j]["S"], -2. * sites[j]["u"]
			S = np.sqrt(S_i * S_j)
			e = np.exp(2j*np.pi * Qvec @ (np.array(sites[j]["pos"]) - np.array(sites[i]["pos"])))

			M[            i,             j] = e * S * u_i[x]        * u_j[y].conj()
			M[            i, num_sites + j] = e * S * u_i[x]        * u_j[y]
			M[num_sites + i,             j] = e * S * u_i[x].conj() * u_j[y].conj()
			M[num_sites + i, num_sites + j] = e * S * u_i[x].conj() * u_j[y]

		M = ops.transpose().conj() @ M @ ops
		for E_idx in range(num_sites * 2):
			S_mats[E_idx, x, y] += M[E_idx, E_idx] / (2. * num_sites)

	proj = np.eye(3) - np.outer(Qvec, Qvec) / (Qvec @ Qvec)
	weights = np.array([ np.abs((proj @ S_mats[E_idx, :, :]).trace().real) for E_idx in range(2*num_sites) ])
	print_infos("\nH =\n%s\nC =\n%s\nH_trafo =\n%s\nEs = %s\nws = %s" % (H, C, H_trafo, Es, weights))

	return Es, weights
