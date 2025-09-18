#
# TODO: implementation of the violini algo
# 
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

import sys
import os
sys.path.append(os.path.dirname(__file__) + "/..")

import numpy as np
import numpy.linalg as la
import libs.tas as tas
import libs.helpers as helpers
import algos.vio_cov as vc
import algos.vio_cov_ext as vce
import algos.vio_cov_ext2 as vce2


#
# resolution algorithm
#

# Qup = Qz

l_lambda = [1, 3, 5, 10, 12, 20]

# vio.py
LP12, LP2M1, LM12, LM2S, Rd = 149.8e7, 7800.6e7, 54.8e7, 1229.5e7, 4000.e7
dPM, dMS, dRd, dthetaf, dDz = np.sqrt(12**2 + 6**2)*1e7, 6e7, 26e7, 0.0065, 30e7
IN5 = {
    "det_shape":"VCYL",
    "L_chopP12":[LP12, 0.],            # [distance, delta]
    "L_chopP2M1":[LP2M1, dPM],           # [distance, delta]
    "L_chopM12":[LM12, 0.],              # [distance, delta]
    "L_chopM2S":[LM2S, dMS],            # [distance, delta]
    "rad_det":[Rd, dRd], #[4000.e7, 0],              # [distance, delta]
    "theta_i":[0, 0], # [0., 2.04e-3],                 # [angle, delta]
    "phi_i":[0, 0], # [0., 1.25e-2],                   # [angle, delta]
    "delta_theta_f":dthetaf,
    "delta_z":dDz, #0,
    "prop_chopP":[np.deg2rad(9.), np.divide(7000.*np.pi, 30), np.divide(17000.*np.pi, 30)],    # [window angle, min rot speed, max rot speed]
    "prop_chopM":[np.deg2rad(3.25), np.divide(7000.*np.pi, 30), np.divide(17000.*np.pi, 30)],    # [window angle, min rot speed, max rot speed]
    "delta_time_det":0
}

d_choppers = {"chopperP":[np.deg2rad(9.), np.divide(7000.*np.pi, 30), np.divide(17000.*np.pi, 30), np.divide(8500.*np.pi, 30)], "chopperM":[np.deg2rad(3.25), np.divide(7000.*np.pi, 30), np.divide(17000.*np.pi, 30), np.divide(8500.*np.pi, 30)]}
d_la = {"L_PM":8005.2e7, "L_MS":1229.5e7, "rad":4000.e7, "z":0, "theta_i":0, "phi_i":0, "theta_f":0, "phi_f":0}
d_delta = {"dlt_Prad":0, "dlt_Pz":0, "dlt_Mrad":0, "dlt_Mz":0, "dlt_Srad":0, "dlt_Sz":0, "dlt_Drad":26e7, "dlt_Dz":30e7, "dlt_tP":8.82e-5, "dlt_tM":3.19e-5, "dlt_tD":0, "dlt_theta_i":0, "dlt_theta_f":6.5e-3}
det_shape = 'VCYL'

lbd = 2
k_i = 2*np.pi/lbd
k_f = 2*np.pi/lbd
Q=1
v_i = vc.k2v(k_i)
v_f = vc.k2v(k_f)
theta_f = tas.get_scattering_angle(k_i, k_f, Q)
d_geo = {"dist_PM":[LP12, 0., LP2M1, dPM, LM12, 0.], "dist_MS":[LM2S, dMS], "dist_SD":[Rd, dRd, 0, dDz], "angles":[0, 0, 0, 0, theta_f, dthetaf], "delta_time_detector":0}
d_la["theta_f"] = theta_f

Q_ki = tas.get_psi(k_i, k_f, Q, 1)
rot = helpers.rotation_matrix_nd(-Q_ki, 4)

print('\n', lbd, k_i, k_f, v_i, v_f, tas.get_E(k_i, 0))
print("!!!!!!!!!! VIO !!!!!!!!!!")
covVio = vc.cov(d_geo, d_choppers, v_i, v_f, det_shape, False)
covQhwVio = np.dot(rot.T, np.dot(covVio, rot))
print(covQhwVio)
lvio = [float(np.sqrt(covQhwVio[0][0])), float(np.sqrt(covQhwVio[1][1])), float(np.sqrt(covQhwVio[2][2])), float(np.sqrt(covQhwVio[3][3]))]




# vio_ext.py








# vio_ext2.py


d_choppers = {"chopperP":[np.deg2rad(9.), np.divide(7000.*np.pi, 30), np.divide(17000.*np.pi, 30), np.divide(8500.*np.pi, 30)], "chopperM":[np.deg2rad(3.25), np.divide(7000.*np.pi, 30), np.divide(17000.*np.pi, 30), np.divide(8500.*np.pi, 30)]}
d_la = {"L_PM":8005.2e7, "L_MS":1229.5e7, "rad":4000.e7, "z":0, "theta_i":0, "phi_i":0, "theta_f":0, "phi_f":0}
d_delta = {"dlt_Prad":0, "dlt_Pz":0, "dlt_Mrad":0, "dlt_Mz":0, "dlt_Srad":0, "dlt_Sz":0, "dlt_Drad":26e7, "dlt_Dz":30e7, "dlt_tP":8.82e-5, "dlt_tM":3.19e-5, "dlt_tD":0, "dlt_theta_i":0, "dlt_theta_f":6.5e-3}
det_shape = 'VCYL'

l_lambda = [1, 3, 5, 10, 12, 20]
nb_lbd = len(l_lambda)
violini = []
violiniExt = []
for i in range(nb_lbd):
    lambdak = l_lambda[i]
    k_i = 2*np.pi/lambdak
    k_f = 2*np.pi/lambdak
    if i >= 3:
        Q = 0.5
    else:
        Q = 1
    v_i = vc.k2v(k_i)
    v_f = vc.k2v(k_f)
    theta_f = tas.get_scattering_angle(k_i, k_f, Q)
    d_geo = {"dist_PM":[149.8e7, 0., 7800.6e7, 0., 54.8e7, 0.], "dist_MS":[1229.5e7, 0.], "dist_SD":[4000.e7, 26e7, 0, 30e7], "angles":[0, 0, 0, 0, theta_f, 6.5e-3], "delta_time_detector":0}
    d_la["theta_f"] = theta_f

    Q_ki = tas.get_psi(k_i, k_f, Q, 1)
    rot = helpers.rotation_matrix_nd(-Q_ki, 4)

    print('\n', lambdak, k_i, k_f, v_i, v_f, tas.get_E(k_i, 0))
    print("!!!!!!!!!! VIO !!!!!!!!!!")
    covVio = vc.cov(d_geo, d_choppers, v_i, v_f, det_shape, False)
    covQhwVio = np.dot(rot.T, np.dot(covVio, rot))
    print(covQhwVio)
    lvio = [float(np.sqrt(covQhwVio[0][0])), float(np.sqrt(covQhwVio[1][1])), float(np.sqrt(covQhwVio[2][2])), float(np.sqrt(covQhwVio[3][3]))]

    print("\n!!!!!!!!!! VIO_ext !!!!!!!!!!")
    covVioExt = vce.cov(v_i, v_f, d_la, d_delta, verbose=False)
    covQhwVioExt = np.dot(rot.T, np.dot(covVioExt, rot))
    print(covQhwVioExt)
    lvioext = [float(np.sqrt(covQhwVioExt[0][0])), float(np.sqrt(covQhwVioExt[1][1])), float(np.sqrt(covQhwVioExt[2][2])), float(np.sqrt(covQhwVioExt[3][3]))]

    violini.append(lvio)
    violiniExt.append(lvioext)

print('\n', '\n', violini, '\n')
print(violiniExt)

for i in range(nb_lbd):
    print('\n', violini[i], '\n', violiniExt[i])
#test

lambd = 4
ki = 2*np.pi/lambd
kf = 2*np.pi/lambd
Q_ = 0.63
vi = vc.k2v(ki)
vf = vc.k2v(kf)
thetaf = tas.get_scattering_angle(ki, kf, Q_)
chopRotSpeed = 12000
dtp = 9/(chopRotSpeed*6*2)
dtm = 3.25/(chopRotSpeed*6*2)


dictla = {"L_PM":8005.2e7, "L_MS":1229.5e7, "rad":4000.e7, "z":0, "theta_i":0, "phi_i":0, "theta_f":thetaf, "phi_f":0}
#dict_a = {"theta_i":0, "phi_i":0, "theta_f":thetaf}
dict_dlt = {"dlt_Prad":0, "dlt_Pz":0, "dlt_Mrad":0, "dlt_Mz":0, "dlt_Srad":0, "dlt_Sz":0, "dlt_Drad":0, "dlt_Dz":0, "dlt_tP":dtp, "dlt_tM":dtm, "dlt_tD":0, "dlt_theta_i":0, "dlt_theta_f":0}

covVioExt = vce.cov(vi, vf, dictla, dict_dlt, verbose=True)
print(covVioExt[3][3], np.sqrt(covVioExt[3][3]))



### vio
#def cov(param_geo, param_choppers, v_i, v_f, shape, verbose=False):
#    """param_geo: {dist_PM:[PM1, sigma1, PM2, sigma2, ...], dist_MS:[MS1, sigma1, MS2, sigma2, ...], dist_SD:[ (if HCYL: x, sigma_x), radius, sigma_r, (if VCYL: z, sigma_z)], angles:[theta_i, sigma_theta_i, phi_i, sigma_phi_i, theta_f, sigma_theta_f, (if SPHERE: phi_f, sigma_phi_f)], delta_time_detector:value (0 by default)},
#    param_choppers: {chopperP:[window_angle, min_rot_speed, max_rot_speed, rot_speed], chopperM:[window_angle, min_rot_speed, max_rot_speed, rot_speed]},
#    v_i, v_f: velocity of the incident and scattered neutron,
#    shape = 'SPHERE', 'VCYL', 'HCYL': shape of the detector (sphere, vertical cylinder or horizontal cylinder)"""
    
    
### vio_ext
#def cov(dict_length, dict_angles, v_i, v_f, delta, shape='VCYL', verbose=False):
#    """ dict_length = {"L_PM":value, "L_MS":value, "rad":value, "z":value}, dict_angles = {"theta_i":value, "phi_i":value, "theta_f":value}, v_i, v_f: velocities,
#    delta = {"dlt_Prad":value, "dlt_Pz":value, "dlt_Mrad":value, "dlt_Mz":value, "dlt_Srad":value, "dlt_Sz":value, "dlt_Drad":value, "dlt_Dz":value, 
#        "dlt_tP":value, "dlt_tM":value, "dlt_tD":value, "dlt_theta_i":value, "dlt_theta_f":value}, shape in ("SPHERE", "HCYL", "VCYL")
#    """


#dict_length = {"L_PM":8005.2e7, "L_MS":1229.5e7, "rad":4000e7}
#dict_angles = {"theta_i":0, "phi_i":0, "theta_f":theta_f,"phi_f":0}
#v_i = vio_cov.k2v(k_i)
#v_f = vio_cov.k2v(k_f)
#delta_plength = [0, 0, 0, 26e7]
#delta_angles = [0, 0, 6.5e-3, 7.5e-3]
#delta_time = [8.82e-5, 3.19e-5, 0]
#shape = 'SPHERE'
#covQhw = vio_cov.cov(dict_length, dict_angles, v_i, v_f, delta_plength, delta_angles, delta_time, shape, verbose=True)
#covQhwInv = la.inv(np.divide(covQhw, helpers.sig2fwhm))
# Going from ki, kf, Qz to Qpara, Qperp, Qz :
#Q_ki = tas.get_psi(k_i, k_f, Q, 1)
#rot = helpers.rotation_matrix_nd(-Q_ki, 4)
#covQhwInv = np.dot(rot.T, np.dot(covQhwInv, rot))

#import libs.reso as reso

#ellipses = reso.calc_ellipses(covQhwInv, verbose = True)
#reso.plot_ellipses(ellipses, verbose = True)