#
# Covariance matrix cov(Q,\hbar\omega) calculations for different shape of the detector
#
# @author Mecoli Victor <mecoli@ill.fr>
# @date feb-2025
# @license GPLv2
#
# @desc for algorithm: [vio14] N. Violini et al., NIM A 736 (2014) pp. 31-39, doi: 10.1016/j.nima.2013.10.042
#
# ----------------------------------------------------------------------------
# Takin (inelastic neutron scattering software package)
# Copyright (C) 2017-2025  Tobias WEBER (Institut Laue-Langevin (ILL),
#                          Grenoble, France).
# Copyright (C) 2013-2017  Tobias WEBER (Technische Universitaet Muenchen
#                          (TUM), Garching, Germany).
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; version 2 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
# ----------------------------------------------------------------------------
#


### See definition in my notes (Important to modify this line when a publication is done) /!\/!\/!\/!\/!\

import numpy as np
import numpy.linalg as la

# Conversion
meV2J = 1.602176634e-22
m2A = 1e10
#Constant
m_n = 1.67492749804e-27
h = 6.62607015e-34
hbar = np.divide(h, 2*np.pi)

#Initialisation of the covariance Matrix
covQhw = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])

def k2v(k):
    return np.divide(k*np.square(m2A)*hbar, m_n)

def RPM2rados(rot_speed):
    return np.divide(rot_speed*np.pi, 30)

# Calcul of the length of a segment divided in small segments
def calcDistLin(dist):
    """dist: [L1, delta1, L2, delta2, ...]"""
    L, nb = 0, len(dist)
    for i in range(0,nb,2):
        L += dist[i]
    return L

def reducedCoef(v_i, v_f, L_PM, l_angles, shape):
    """v_i, v_f: velocities, L_PM: distance, l_angles: list of angles, shape in ("SPHERE", "VCYL", "HCYL")"""
    theta_i, phi_i, theta_f, phi_f = l_angles[0], l_angles[1], l_angles[2], l_angles[3]

    Ae, Be = np.divide(np.power(v_i,3), L_PM), np.divide(np.power(v_f,3), L_PM)
    Aq, Bq = np.divide(np.square(v_i), L_PM), np.divide(np.square(v_f), L_PM)
    Ax, Ay, Az = Aq*np.cos(theta_i)*np.cos(phi_i), Aq*np.sin(theta_i)*np.cos(phi_i), Aq*np.sin(phi_i)
    if shape in ("SPHERE", "VCYL"):
        Bx, By, Bz = Bq*np.cos(theta_f)*np.cos(phi_f), Bq*np.sin(theta_f)*np.cos(phi_f), Bq*np.sin(phi_f)
    elif(shape == "HCYL"):
        Bx, By, Bz = Bq*np.sin(phi_f), Bq*np.sin(theta_f)*np.cos(phi_f), Bq*np.cos(theta_f)*np.cos(phi_f)
    return(Ae, Ax, Ay, Az, Be, Bx, By, Bz)

# Calcul of Jacobian's terms for every shape
def jacobTerms(v_i, v_f, l_dist, l_angles, l_AB, l_sizes, shape, verbose=False):
    '''v_i, v_f: velocities, l_dist:list of 6 distances, l_angles: list of 4 angles, l_AB: list of 8 coef got from reducedCoef(), l_sizes: list of 6 numbers, shape in (SPHERE, HCYL, VCYL)'''
    moh = np.divide(m_n, hbar*np.square(m2A))
    L_PM, L_MS, L_SD, rad = l_dist[0], l_dist[1], l_dist[2], l_dist[4]
    theta_i, phi_i, theta_f, phi_f = l_angles[0], l_angles[1], l_angles[2], l_angles[3]
    Ae, Ax, Ay, Az, Be, Bx, By, Bz = l_AB[0], l_AB[1], l_AB[2], l_AB[3], l_AB[4], l_AB[5], l_AB[6], l_AB[7]
    size_PM, size_MS = l_sizes[0], l_sizes[1]
    dQx, dQy, dQz, dE = np.array([]), np.array([]), np.array([]), np.array([])

    # x terms in common
    dQxdL_PMj = np.divide(moh, v_i)*(Ax + Bx*np.divide(L_MS, L_SD))
    dQxdL_MSj = -np.divide(moh, v_i)*Bx*np.divide(L_PM,L_SD)
    dQxdt_PM = -moh*(Ax + Bx*np.divide(L_MS, L_SD))
    dQxdt_MD = moh*Bx*np.divide(L_PM, L_SD)
    dQxdtheta_i = -moh*v_i*np.sin(theta_i)*np.cos(phi_i)
    dQxdphi_i = -moh*v_i*np.cos(theta_i)*np.sin(phi_i)
    # x terms that belong to a precise shape
    dQxdx, dQxdrad, dQxdz, dQxdtheta_f, dQxdphi_f = 0, 0, 0, 0, 0
    if(shape == 'SPHERE'):
        dQxdrad = -np.divide(moh, v_f)*Bx*np.divide(L_PM,L_SD)
        dQxdtheta_f = moh*v_f*np.sin(theta_f)*np.cos(phi_f)
        dQxdphi_f = moh*v_f*np.cos(theta_f)*np.sin(phi_f)
        dQx = np.concatenate(( np.array([dQxdL_PMj]*size_PM), np.array([dQxdL_MSj]*size_MS), np.array([dQxdrad, dQxdt_PM, dQxdt_MD, dQxdtheta_i, dQxdphi_i, dQxdtheta_f, dQxdphi_f]) ))
    elif(shape == 'HCYL'):
        dQxdx = -moh*np.divide(v_f, L_SD)
        dQx = np.concatenate(( np.array([dQxdL_PMj]*size_PM), np.array([dQxdL_MSj]*size_MS), np.array([dQxdx, dQxdrad, dQxdt_PM, dQxdt_MD, dQxdtheta_i, dQxdphi_i, dQxdtheta_f]) ))
    elif(shape == 'VCYL'):
        dQxdrad = -np.divide(moh, v_f)*Bx*np.divide(L_PM,rad)
        dQxdtheta_f = moh*v_f*np.sin(theta_f)*np.cos(phi_f)
        dQx = np.concatenate(( np.array([dQxdL_PMj]*size_PM), np.array([dQxdL_MSj]*size_MS), np.array([dQxdrad, dQxdz, dQxdt_PM, dQxdt_MD, dQxdtheta_i, dQxdphi_i, dQxdtheta_f]) ))
    
    # y terms in common
    dQydL_PMj = np.divide(moh, v_i)*(Ay + By*np.divide(L_MS, L_SD))
    dQydL_MSj = -np.divide(moh, v_i)*By*np.divide(L_PM, L_SD)
    dQydt_PM = -moh*(Ay + By*np.divide(L_MS, L_SD))
    dQydt_MD = moh*By*np.divide(L_PM, L_SD)
    dQydtheta_i = moh*v_i*np.cos(theta_i)*np.cos(phi_i)
    dQydphi_i = -moh*v_i*np.sin(theta_i)*np.sin(phi_i)
    dQydtheta_f = -moh*v_f*np.cos(theta_f)*np.cos(phi_f)
    # y terms that belong to a precise shape
    dQydx, dQydrad, dQydz, dQydphi_f = 0, 0, 0, 0
    if(shape == 'SPHERE'):
        dQydrad = -np.divide(moh, v_f)*By*np.divide(L_PM, L_SD)
        dQydphi_f = moh*v_f*np.sin(theta_f)*np.sin(phi_f)
        dQy = np.concatenate(( np.array([dQydL_PMj]*size_PM), np.array([dQydL_MSj]*size_MS), np.array([dQydrad, dQydt_PM, dQydt_MD, dQydtheta_i, dQydphi_i, dQydtheta_f, dQydphi_f]) ))
    elif(shape == 'HCYL'):
        dQydrad = -np.divide(moh, v_f)*By*np.divide(L_PM, rad)
        dQy = np.concatenate(( np.array([dQydL_PMj]*size_PM), np.array([dQydL_MSj]*size_MS), np.array([dQydx, dQydrad, dQydt_PM, dQydt_MD, dQydtheta_i, dQydphi_i, dQydtheta_f]) ))
    elif(shape == 'VCYL'):
        dQydrad = -np.divide(moh, v_f)*By*np.divide(L_PM, rad)
        dQy = np.concatenate(( np.array([dQydL_PMj]*size_PM), np.array([dQydL_MSj]*size_MS), np.array([dQydrad, dQydz, dQydt_PM, dQydt_MD, dQydtheta_i, dQydphi_i, dQydtheta_f]) ))

    # z terms in common
    dQzdL_PMj = np.divide(moh, v_i)*(Az + Bz*np.divide(L_MS, L_SD))
    dQzdL_MSj = -np.divide(moh, v_i)*Bz*np.divide(L_PM, L_SD)
    dQzdt_PM = -moh*(Az + Bz*np.divide(L_MS, L_SD))
    dQzdt_MD = moh*Bz*np.divide(L_PM, L_SD)
    dQzdphi_i = moh*v_i*np.cos(phi_i)
    # z terms that belong to a precise shape
    dQzdx, dQzdrad, dQzdz, dQzdtheta_i, dQzdtheta_f, dQzdphi_f = 0, 0, 0, 0, 0, 0
    if(shape == 'SPHERE'):
        dQzdrad = -np.divide(moh, v_f)*Bz*np.divide(L_PM, L_SD)
        dQzdphi_f = -moh*v_f*np.cos(phi_f)
        dQz = np.concatenate(( np.array([dQzdL_PMj]*size_PM), np.array([dQzdL_MSj]*size_MS), np.array([dQzdrad, dQzdt_PM, dQzdt_MD, dQzdtheta_i, dQzdphi_i, dQzdtheta_f, dQzdphi_f]) ))
    elif(shape == 'HCYL'):
        dQzdrad = -np.divide(moh, v_f)*Bz*np.divide(L_PM, rad)
        dQzdtheta_f = moh*v_f*np.sin(theta_f)*np.cos(phi_f)
        dQz = np.concatenate(( np.array([dQzdL_PMj]*size_PM), np.array([dQzdL_MSj]*size_MS), np.array([dQzdx, dQzdrad, dQzdt_PM, dQzdt_MD, dQzdtheta_i, dQzdphi_i, dQzdtheta_f]) ))
    elif(shape == 'VCYL'):
        dQzdz = -moh*np.divide(v_f, L_SD)
        dQz = np.concatenate(( np.array([dQzdL_PMj]*size_PM), np.array([dQzdL_MSj]*size_MS), np.array([dQzdrad, dQzdz, dQzdt_PM, dQzdt_MD, dQzdtheta_i, dQzdphi_i, dQzdtheta_f]) ))

    # energy terms in common
    dEdL_PMj = (np.divide(m_n, v_i)*(Ae + Be*np.divide(L_MS,L_SD)))*np.divide(1, np.square(m2A)*meV2J)
    dEdL_MSj = -(np.divide(m_n, v_i)*Be*np.divide(L_PM,L_SD))*np.divide(1, np.square(m2A)*meV2J)
    dEdt_PM = -(m_n*(Ae + Be*np.divide(L_MS,L_SD)))*np.divide(1, np.square(m2A)*meV2J)
    dEdt_MD = m_n*Be*np.divide(L_PM,L_SD)*np.divide(1, np.square(m2A)*meV2J)
    # energy terms that belong to a precise shape
    dEdx, dEdrad, dEdz, dEdtheta_i, dEdphi_i, dEdtheta_f, dEdphi_f = 0, 0, 0, 0, 0, 0, 0
    if(shape == 'SPHERE'):
        dEdrad = -(np.divide(m_n, v_f)*Be*np.divide(L_PM,L_SD))*np.divide(1, np.square(m2A)*meV2J)
        dE = np.concatenate(( np.array([dEdL_PMj]*size_PM), np.array([dEdL_MSj]*size_MS), np.array([dEdrad, dEdt_PM, dEdt_MD, dEdtheta_i, dEdphi_i, dEdtheta_f, dEdphi_f]) ))
    elif(shape == 'HCYL'):
        dEdx = -(np.divide(m_n, v_f)*Be*np.divide(L_PM,L_SD)*np.sin(phi_f))*np.divide(1, np.square(m2A)*meV2J)
        dEdrad = -(np.divide(m_n, v_f)*Be*np.divide(L_PM,L_SD)*np.cos(phi_f))*np.divide(1, np.square(m2A)*meV2J)
        dE = np.concatenate(( np.array([dEdL_PMj]*size_PM), np.array([dEdL_MSj]*size_MS), np.array([dEdx, dEdrad, dEdt_PM, dEdt_MD, dEdtheta_i, dEdphi_i, dEdtheta_f]) ))
    elif(shape == 'VCYL'):
        dEdrad = -(np.divide(m_n, v_f)*Be*np.divide(L_PM,L_SD)*np.cos(phi_f))*np.divide(1, np.square(m2A)*meV2J)
        dEdz = -(np.divide(m_n, v_f)*Be*np.divide(L_PM,L_SD)*np.sin(phi_f))*np.divide(1, np.square(m2A)*meV2J)
        dE = np.concatenate(( np.array([dEdL_PMj]*size_PM), np.array([dEdL_MSj]*size_MS), np.array([dEdrad, dEdz, dEdt_PM, dEdt_MD, dEdtheta_i, dEdphi_i, dEdtheta_f]) ))
    ###########################################################################################
    if(verbose):
        print('x terms:')
        print('dQx/dL_PMj =', dQxdL_PMj, '; dQx/dL_MSj =', dQxdL_MSj, '; dQx/dx =', dQxdx, '; dQx/drad =', dQxdrad)
        print('dQx/dt_PM =', dQxdt_PM, '; dQx/dt_MD =', dQxdt_MD)
        print('dQx/dtheta_i =', dQxdtheta_i, '; dQx/dphi_i =', dQxdphi_i, '; dQx/dtheta_f =', dQxdtheta_f, '; dQx/dphi_f =', dQxdphi_f, '\n')

        print('y terms:')
        print('dQy/dL_PMj =', dQydL_PMj, '; dQy/dL_MSj =', dQydL_MSj, '; dQy/drad =', dQydrad)
        print('dQy/dt_PM =', dQydt_PM, '; dQy/dt_MD =', dQydt_MD)
        print('dQy/dtheta_i =', dQydtheta_i, '; dQy/dphi_i =', dQydphi_i, '; dQy/dtheta_f =', dQydtheta_f, '; dQy/dphi_f =', dQydphi_f, '\n')

        print('z terms:')
        print('dQz/dL_PMj =', dQzdL_PMj, '; dQz/dL_MSj =', dQzdL_MSj, '; dQz/drad =', dQzdrad, '; dQz/dz =', dQzdz)
        print('dQz/dt_PM =', dQzdt_PM, '; dQz/dt_MD =', dQzdt_MD)
        print('dQz/dphi_i =', dQzdphi_i, 'dQz/dphi_f =', dQzdphi_f, '; dQzdtheta_f =', dQzdtheta_f, '\n')

        print('E  terms:')
        print('dE/dL_PMj =', dEdL_PMj, '; dE/dL_MSj =', dEdL_MSj, '; dE/dx =', dEdx, '; dE/drad =', dEdrad, '; dE/dz =', dEdz)
        print('dE/dt_PM =', dEdt_PM, '; dE/dt_MD =', dEdt_MD, '\n')
    return(dQx, dQy, dQz, dE)


# Getting parameters of choppers
def getParamChopper(l_param_chopper, idx_rot=3):
    '''l_param_chopper: [win_angle, min_rot_speed, max_rot_speed, rot_speed, delta_t], idx_rot in {0, 1, 2, 3}'''
    win_angle, rot_speed = l_param_chopper[0], l_param_chopper[idx_rot]
    if l_param_chopper[4] == 0:
        return win_angle, rot_speed
    else:
        return l_param_chopper[4], 0.5

# Storage of the given uncerntainty in a list
def listDeltaGeo(list_param):
    """list_param: [P1, delta1, P2, delta2, ...]"""
    nb = len(list_param)
    dlt = np.zeros(int(nb/2))
    for i in range(1,nb,2):
        idx = int((i-1)/2)
        dlt[idx]=list_param[i]
    return dlt

# Calcul of the chopper's time uncertainty for a pair of counter rotating choppers
def deltaTimeChopper(window_angle, rot_speed):
    return np.divide(window_angle, 2*rot_speed)

# Return the time uncerntainty for t_PM and t_MD
def listDeltaTime(window_angleP, rot_speedP, window_angleM, rot_speedM, dltD = 0, verbose = False):
    dltP = deltaTimeChopper(window_angleP, rot_speedP)
    dltM = deltaTimeChopper(window_angleM, rot_speedM)
    ###########################################################################################
    if(verbose):
        print('\ndltP =', dltP, 'dltM =', dltM, '\n')

    return np.array([ np.sqrt(np.square(dltP) + np.square(dltM)), np.sqrt(np.square(dltM) + np.square(dltD)) ])

# Getting list of uncerntainty
def getDeltas(param_geo, param_choppers, l_sizes, verbose = False):
    '''param_geo, param_choppers: see cov(...), l_sizes: list of 6 numbers'''
    win_angleP, rot_speedP = getParamChopper(param_choppers['chopperP'])
    win_angleM, rot_speedM = getParamChopper(param_choppers['chopperM'])
    ###########################################################################################
    if(verbose):
        print('win_angleP =', win_angleP, '; rot_speedP = ', rot_speedP)
        print('win_angleM =', win_angleM, '; rot_speedM = ', rot_speedM, '\n')

    deltas = np.zeros(l_sizes[5])
    deltas[:l_sizes[0]] = listDeltaGeo(param_geo['dist_PM'])
    deltas[l_sizes[0]:(l_sizes[0] + l_sizes[1])] = listDeltaGeo(param_geo['dist_MS'])
    deltas[(l_sizes[0] + l_sizes[1]):(l_sizes[0] + l_sizes[1] + l_sizes[2])] = listDeltaGeo(param_geo['dist_SD'])
    deltas[(l_sizes[0] + l_sizes[1] + l_sizes[2]):(l_sizes[0] + l_sizes[1] + l_sizes[2] + l_sizes[3])] = listDeltaTime(win_angleP, rot_speedP, win_angleM, rot_speedM, param_geo['delta_time_detector'], verbose)
    deltas[(l_sizes[0] + l_sizes[1] + l_sizes[2] + l_sizes[3]):] = listDeltaGeo(param_geo['angles'])
    return deltas

# Creation and filling of the Jacobian matrix
def jacobianMatrix(dQx, dQy, dQz, dE):
    """dQi and dE are lists of derivations in respect to every parameters"""
    nb_col = len(dQx)
    jacob = np.array([dQx, dQy, dQz, dE])
    return jacob

# Creation and filling of the covxi matrix (uncerntainty on independant parameters xi)
def covxiMatrix(deltas):
    """delta: list of uncerntainties (for independant variables)"""
    nb_dlt = len(deltas)
    covxi = np.eye(nb_dlt)
    for i in range(nb_dlt):
        covxi[i][i] = np.square(deltas[i])
    return covxi

def cov(param_geo, param_choppers, v_i, v_f, shape, verbose=False):
    """param_geo: {dist_PM:[PM1, sigma1, PM2, sigma2, ...], dist_MS:[MS1, sigma1, MS2, sigma2, ...], dist_SD:[ (if HCYL: x, sigma_x), radius, sigma_r, (if VCYL: z, sigma_z)], angles:[theta_i, sigma_theta_i, phi_i, sigma_phi_i, theta_f, sigma_theta_f, (if SPHERE: phi_f, sigma_phi_f)], delta_time_detector:value (0 by default)},
    param_choppers: {chopperP:[window_angle, min_rot_speed, max_rot_speed, rot_speed, delta_tp], chopperM:[window_angle, min_rot_speed, max_rot_speed, rot_speed, delta_tm]},
    v_i, v_f: velocity of the incident and scattered neutron,
    shape = 'SPHERE', 'VCYL', 'HCYL': shape of the detector (sphere, vertical cylinder or horizontal cylinder)"""
    # Calcul of distances and angles for each shape of detectors
    L_PM, L_MS = calcDistLin(param_geo["dist_PM"]), calcDistLin(param_geo["dist_MS"])
    x, rad, z, theta_i, phi_i, theta_f, phi_f = 0, 0, 0, 0, 0, 0, 0
    if(shape == "SPHERE"):
        L_SD = param_geo["dist_SD"][0]
        theta_i, phi_i, theta_f, phi_f = param_geo["angles"][0], param_geo["angles"][2], param_geo["angles"][4], param_geo["angles"][6]
    if(shape == "HCYL"):
        # Distances
        x, rad = param_geo["dist_SD"][0], param_geo["dist_SD"][2]
        L_SD = np.sqrt(np.square(x) + np.square(rad))
        # Angles
        theta_i, phi_i, theta_f, phi_f = param_geo["angles"][0], param_geo["angles"][2], param_geo["angles"][4], np.arccos(np.divide(rad, L_SD))
    if(shape == "VCYL"):
        # Distances
        rad, z = param_geo["dist_SD"][0], param_geo["dist_SD"][2]
        L_SD = np.sqrt(np.square(rad) + np.square(z))
        # Angles
        theta_i, phi_i, theta_f, phi_f = param_geo["angles"][0], param_geo["angles"][2], param_geo["angles"][4], np.arccos(np.divide(rad, L_SD))
    l_dist, l_angles = np.array([L_PM, L_MS, L_SD, x, rad, z]), np.array([theta_i, phi_i, theta_f, phi_f])

    # Definition of variables depending on the shape of the detector
    Ae, Ax, Ay, Az, Be, Bx, By, Bz = reducedCoef(v_i, v_f, L_PM, l_angles, shape)
    l_AB = np.array([Ae, Ax, Ay, Az, Be, Bx, By, Bz])

    # Number of variables for each set of instrument's parameters (distances, times, angles)
    size_PM, size_MS, size_SD, size_tps, size_angles = int(np.divide(len(param_geo["dist_PM"]), 2)), int(np.divide(len(param_geo["dist_MS"]), 2)), int(np.divide(len(param_geo["dist_SD"]), 2)), 2, int(np.divide(len(param_geo["angles"]), 2))
    nb_param = size_PM + size_MS + size_SD + size_tps + size_angles
    l_sizes = np.array([size_PM, size_MS, size_SD, size_tps, size_angles, nb_param])

    # Calcul of Jacobian's terms
    dQx, dQy, dQz, dE = jacobTerms(v_i, v_f, l_dist, l_angles, l_AB, l_sizes, shape, verbose)
    # List of uncertainty
    deltas = getDeltas(param_geo, param_choppers, l_sizes, verbose)
    # Jacobian and parameter's uncertainty matrices
    jacobian, covxi = jacobianMatrix(dQx, dQy, dQz, dE), covxiMatrix(deltas)

    covQhw = np.dot(jacobian, np.dot(covxi, jacobian.T))
    ###########################################################################################
    if(verbose):
        print('param_geo =', param_geo)
        print('param_choppers =', param_choppers)
        print('v_i =', v_i, '; v_f =', v_f)
        print('det_shape =', shape, '\n')
        print('L_PM =', L_PM, '; L_MS =', L_MS, '; L_SD =', L_SD)
        print('x =', x, '; rad =', rad, '; z=', z, '; theta_i =', theta_i, '; phi_i =', phi_i, '; theta_f =', theta_f, '; phi_f =', phi_f, '\n')
        print('size_PM =', size_PM, '; size_MS =', size_MS, '; size_SD =', size_SD, '; size_tps =', size_tps, '; size_angles =', size_angles, '\nl_sizes =', l_sizes, '\n')
        print('Ae =', Ae, '; Ax =', Ax, '; Ay =', Ay, '; Az =', Az, '\nBe =', Be, '; Bx =', Bx, '; By =', By, '; Bz =', Bz, '\n')
        print('dQx =', dQx, '\ndQy =', dQy, '\ndQz =', dQz, '\ndE =', dE, '\n')
        print('deltas =', deltas, '\n')
        print('Jacobian =', jacobian, '\n')
        print('covxi = ', covxi, '\n')
        print('covQhw =', covQhw, '\n')

    return covQhw