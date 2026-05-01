#!/usr/bin/env python3
#
# library to calculate TAS angles from rlu
# @author Tobias Weber <tweber@ill.fr>
# @date 1-aug-2018
# @license see 'LICENSE' file
#


# -----------------------------------------------------------------------------
# imports
# -----------------------------------------------------------------------------
import sys

try:
    import numpy as np
    import numpy.linalg as la
except ImportError:
    print("Numpy could not be imported!")
    exit(-1)

try:
    import scipy as sp
    import scipy.constants as co

    # calculate constants
    hbar_in_meVs = co.Planck/co.elementary_charge*1000./2./np.pi
    E_to_k2 = 2.*co.neutron_mass/hbar_in_meVs**2. / co.elementary_charge*1000. * 1e-20
except ImportError:
    #print("Scipy could not be imported!")

    # calculated with scipy, using the formula above
    E_to_k2 = 0.482596423544

k2_to_E = 1./E_to_k2
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# library functions
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
#
# rotate a vector around an axis using Rodrigues' formula
# see https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
#
def rotate(_axis, vec, phi):
    axis = _axis / la.norm(_axis)

    s = np.sin(phi)
    c = np.cos(phi)

    return c*vec + (1.-c)*vec@axis*axis + s*np.cross(axis, vec)


#
# get metric from crystal B matrix
# basis vectors are in the columns of B, i.e. the second index
# see (Arens 2015), p. 815
#
def get_metric(B):
    #return np.einsum("ij,ik -> jk", B, B)
    return np.transpose(B) @ B


#
# cross product in fractional coordinates: c^l = eps_ijk g^li a^j b^k
# see (Arens 2015), p. 815
#
def cross(a, b, B):
    # levi-civita in fractional coordinates
    def levi(i,j,k, B):
        M = np.array([B[:,i], B[:,j], B[:,k]])
        return la.det(M)

    metric_inv = la.inv(get_metric(B))
    eps = [[[ levi(i,j,k, B)
        for k in range(0,3) ]
            for j in range(0,3) ]
                for i in range(0,3) ]
    return np.einsum("ijk,j,k,li -> l", eps, a, b, metric_inv)


#
# dot product in fractional coordinates
# see (Arens 2015), p. 808
#
def dot(a, b, metric):
    return a @ metric @ b


#
# angle between peaks in fractional coordinates
# see (Arens 2015), p. 808
#
def angle(a, b, metric):
    len_a = np.sqrt(dot(a, a, metric))
    len_b = np.sqrt(dot(b, b, metric))

    c = dot(a, b, metric) / (len_a * len_b)

    # check for rounding errors
    if c > 1.:
        #print("arccos precision overflow: " + str(c) + ".")
        c = 1.
    if c < -1.:
        #print("arccos precision underflow: " + str(c) + ".")
        c = -1.

    return np.arccos(c)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
def k_to_lam(k):
    return 2.*np.pi / k


#
# mono (or ana) k  ->  A1 & A2 angles (or A5 & A6)
#
def get_mono_angle(k, d, only_a1 = False):
    s = np.pi/(d*k)
    a1 = np.arcsin(s)
    if only_a1:
        return a1
    return [a1, 2.*a1]


#
# a1 angle (or a5)  ->  mono (or ana) k
#
def get_monok(theta, d):
    s = np.sin(theta)
    k = np.pi/(d*s)
    return k
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
#
# sample scattering angle a4
#
def get_scattering_angle(ki, kf, Q):
    c = (ki**2. + kf**2. - Q**2.) / (2.*ki*kf)
    return np.arccos(c)


#
# get |Q| from ki, kf and a4
#
def get_Q(ki, kf, a4):
    c = np.cos(a4)
    return np.sqrt(ki**2. + kf**2. - c*(2.*ki*kf))
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
#
# angle rotating Q into ki (i.e. angle inside scattering triangle)
#
def get_psi(ki, kf, Q, sense = 1.):
    c = (ki**2. + Q**2. - kf**2.) / (2.*ki*Q)
    return sense*np.arccos(c)


#
# angle rotating Q into kf (i.e. angle outside scattering triangle),
# Q_kf = Q_ki + twotheta_s
#
def get_eta(ki, kf, Q, sense = 1.):
    c = (ki**2. - Q**2. - kf**2.) / (2.*kf*Q)
    return sense*np.arccos(c)


#
# crystallographic A matrix converting fractional to lab coordinates
# see https://de.wikipedia.org/wiki/Fraktionelle_Koordinaten
#
def get_A(lattice, angles):
    cs = np.cos(angles)
    s1 = np.sin(angles[1])
    s2 = np.sin(angles[2])

    a = lattice[0] * np.array([1, 0, 0])
    b = lattice[1] * np.array([cs[2], s2, 0])
    c = lattice[2] * np.array([cs[1], \
        (cs[0]-cs[1]*cs[2]) / s2, \
        np.sqrt(s1*s1 - ((cs[0] - cs[2]*cs[1])/s2)**2.)])

    # the real-space basis vectors form the columns of the A matrix
    return np.transpose(np.array([a, b, c]))


#
# crystallographic B matrix converting rlu to 1/A
# the reciprocal-space basis vectors form the columns of the B matrix
#
def get_B(lattice, angles):
    A = get_A(lattice, angles)
    B = 2.*np.pi * np.transpose(la.inv(A))
    return B


#
# UB orientation matrix
# see https://dx.doi.org/10.1107/S0021889805004875
#
def get_UB(B, orient1_rlu, orient2_rlu, orientup_rlu):
    orient1_invA = B @ orient1_rlu
    orient2_invA = B @ orient2_rlu
    orientup_invA = B @ orientup_rlu

    orient1_invA = orient1_invA / la.norm(orient1_invA)
    orient2_invA = orient2_invA / la.norm(orient2_invA)
    orientup_invA = orientup_invA / la.norm(orientup_invA)

    U_invA = np.array([orient1_invA, orient2_invA, orientup_invA])
    UB = U_invA @ B
    return UB


#
# a3 sample rotation & a4 sample scattering angles
# see https://dx.doi.org/10.1107/S0021889805004875
#
def get_sample_angle(ki, kf, Q_rlu, orient_rlu, orient_up_rlu, B, sense_sample = 1., a3_offs = np.pi):
    metric = get_metric(B)
    #print("Metric: " + str(metric))

    # angle xi between Q and orientation reflex
    xi = angle(Q_rlu, orient_rlu, metric)

    # sign of xi
    if dot(cross(orient_rlu, Q_rlu, B), orient_up_rlu, metric) < 0.:
        xi = -xi

    # length of Q
    Qlen = np.sqrt(dot(Q_rlu, Q_rlu, metric))

    # distance to plane
    up_len = np.sqrt(dot(orient_up_rlu, orient_up_rlu, metric))
    dist_Q_plane = dot(Q_rlu, orient_up_rlu, metric) / up_len

    # angle psi enclosed by ki and Q
    psi = get_psi(ki, kf, Qlen, sense_sample)

    a3 = - psi - xi + a3_offs
    a4 = get_scattering_angle(ki, kf, Qlen)

    return [a3, a4, dist_Q_plane]


#
# (hkl) position
# see https://dx.doi.org/10.1107/S0021889805004875
#
def get_hkl(ki, kf, a3, Qlen, orient_rlu, orient_up_rlu, B, sense_sample = 1., a3_offs = np.pi):
    B_inv = la.inv(B)

    # angle enclosed by ki and Q
    psi = get_psi(ki, kf, Qlen, sense_sample)

    # angle between Q and orientation reflex
    xi = - a3 + a3_offs - psi

    Q_lab = rotate(B @ orient_up_rlu, B @ orient_rlu*Qlen, xi)
    Q_lab *= Qlen / la.norm(Q_lab)
    Q_rlu = B_inv @ Q_lab

    return Q_rlu
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
#
# get ki from kf and energy transfer
#
def get_ki(kf, E):
    return np.sqrt(kf**2. + E_to_k2*E)


#
# get kf from ki and energy transfer
#
def get_kf(ki, E):
    return np.sqrt(ki**2. - E_to_k2*E)


#
# get energy transfer from ki and kf
#
def get_E(ki, kf):
    return (ki**2. - kf**2.) / E_to_k2
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
#
# get the difference in tas angles for two positions
#
def get_angle_deltas(ki1, kf1, Q_rlu1, dm1, da1, \
    ki2, kf2, Q_rlu2, dm2, da2, \
    orient_rlu, orient_up_rlu, B, sense_sample=1., a3_offs=np.pi):

    # position 1
    [a1_1, a2_1] = get_mono_angle(ki1, dm1)
    [a5_1, a6_1] = get_mono_angle(kf1, da1)
    [a3_1, a4_1, dist_Q_plane_1] = get_sample_angle(ki1, kf1, Q_rlu1, orient_rlu, orient_up_rlu, B, sense_sample, a3_offs)

    # position 2
    [a1_2, a2_2] = get_mono_angle(ki2, dm2)
    [a5_2, a6_2] = get_mono_angle(kf2, da2)
    [a3_2, a4_2, dist_Q_plane_2] = get_sample_angle(ki2, kf2, Q_rlu2, orient_rlu, orient_up_rlu, B, sense_sample, a3_offs)

    return [a1_2-a1_1, a2_2-a2_1, a3_2-a3_1, a4_2-a4_1, a5_2-a5_1, a6_2-a6_1, dist_Q_plane_1, dist_Q_plane_2]


#
# get the instrument driving time
#
def driving_time(deltas, rads_per_times):
    times = np.abs(deltas) / rads_per_times
    return np.max(times)

# -----------------------------------------------------------------------------



if __name__ == "__main__":
    from tasgui import run_tas
    run_tas()
