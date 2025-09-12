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
import algos.vio_cov_ext as vio_cov_ext


#
# resolution algorithm
#

# Qup = Qz

def calc(param):
    # Storage of informations
    verbose = param["verbose"]
    # Normes of ki, kf and Q
    ki, kf, Q = param["ki"], param["kf"], param["Q"]
    # Shape of the detector
    det_shape = param["det_shape"]
    if det_shape not in ('SPHERE', 'VCYL', 'HCYL'):
        print("this shape is not taken in account")
        return None
    # Velocity in m/s
    vi, vf = vio_cov_ext.k2v(ki), vio_cov_ext.k2v(kf)
    # Angles
    theta_i, phi_i, theta_f, phi_f = param["angles"][0], param["angles"][2], param["angles"][4], 0
    if(det_shape == 'SPHERE'):
        phi_f = param["angles"][6]
    if(det_shape == 'HCYL'):
        phi_f = np.atan( np.divide(param["dist_SD"][0], param["dist_SD"][2]) )
    if(det_shape == 'VCYL'):
        phi_f = np.atan( np.divide(param["dist_SD"][2], param["dist_SD"][0]) )

    ki_xy, ki_z, kf_xy, kf_z = ki*np.cos(phi_i), ki*np.sin(phi_i), kf*np.cos(phi_f), kf*np.sin(phi_f)
    Q_x = ki_xy*np.cos(theta_i) - kf_xy*np.cos(theta_f)
    Q_y = ki_xy*np.sin(theta_i) - kf_xy*np.sin(theta_f)
    Q_z = ki_z - kf_z
    Q_xy = np.sqrt( np.square(Q_x) + np.square(Q_y) )
    # Information on the instrument
    dict_geo = {"dist_PM":param["dist_PM"], "dist_MS":param["dist_MS"], "dist_SD":param["dist_SD"], "angles":param["angles"], "delta_time_detector":param["delta_time_det"]}
    dict_choppers = {"chopperP":param["chopperP"], "chopperM":param["chopperM"]}
    ###########################################################################################
    if(verbose):
        print("ki =", ki, "; kf =", kf, "; Q =", Q)
        print("det_shape =", det_shape)
        print("vi =", vi, "; vf =", vf)
        print("theta_i =", theta_i, "; phi_i =", phi_i,"; theta_f =", theta_f, "; phi_f =", phi_f)
        print("ki_xy =", ki_xy, "; ki_z =", ki_z,"; kf_xy =", kf_xy, "; kf_z =", kf_z, "; Q_x =", Q_x, "; Q_y =", Q_y, "; Q_xy =", Q_xy, "; Q_z =", Q_z)
        print("\ndict_geo =", dict_geo, "\ndict_choppers =", dict_choppers, "\n")

    # Energy transfer, Q vector and Covariance matrix
    E = tas.get_E(ki, kf)
    vec_Q = np.array([Q_x, Q_y, Q_z])
    covQhw = vio_cov_ext.cov(dict_geo, dict_choppers, vi, vf, det_shape, verbose)
    covQhwInv = la.inv(np.divide(covQhw, helpers.sig2fwhm))
    # Going from ki, kf, Qz to Qpara, Qperp, Qz :
    Q_ki = tas.get_psi(ki_xy, kf_xy, Q_xy, param["sample_sense"])
    rot = helpers.rotation_matrix_nd(-Q_ki, 4)
    covQhwInv = np.dot(rot.T, np.dot(covQhwInv, rot))
    ###########################################################################################
    if(verbose):
        print("E =", E, "; vec_Q =", vec_Q)
        print("covQhw =", covQhw)
        print("covQhwInv =", covQhwInv)
        print("In the base (Qpara, Qperp, Qz) :")
        print("rot =", rot,'\ncovQhwInv =', covQhwInv)

    res={}
    res["ki"] = ki
    res["kf"] = kf
    res["Q"] = Q
    res["vec_Q"] = vec_Q
    res["E"] = E
    res["reso"] = covQhwInv
    res["ok"] = True
    return res

IN5 = {
    "det_shape":"VCYL",
    "L_chopP12":[149.8e7, 0.],            # [distance, delta]
    "L_chopP2M1":[7800.6e7, 0.],           # [distance, delta]
    "L_chopM12":[54.8e7, 0.],              # [distance, delta]
    "L_chopM2S":[1229.5e7, 0.],            # [distance, delta]
    "rad_det":[4000.e7, 26e7], #[4000.e7, 0],              # [distance, delta]
    "theta_i":[0, 0], # [0., 2.04e-3],                 # [angle, delta]
    "phi_i":[0, 0], # [0., 1.25e-2],                   # [angle, delta]
    "delta_theta_f":6.5e-3, #0,
    "delta_z":30e7, #0,
    "prop_chopP":[np.deg2rad(9.), np.divide(7000.*np.pi, 30), np.divide(17000.*np.pi, 30)],    # [window angle, min rot speed, max rot speed]
    "prop_chopM":[np.deg2rad(3.25), np.divide(7000.*np.pi, 30), np.divide(17000.*np.pi, 30)],    # [window angle, min rot speed, max rot speed]
    "delta_time_det":0
}

k_i = 2*np.pi/5.9
k_f = 2*np.pi/5.9
#k_i = 2*np.pi/2
#k_f = 2*np.pi/2
l_Q = [0.632, 0.734, 0.969, 1.158, 1.214, 1.265, 1.421, 1.463, 1.552, 1.597]
#Ei = tas.get_E(k_i,0)
for Q in l_Q:
    print()
    print("Q = ", Q)
    theta_f = tas.get_scattering_angle(k_i, k_f, Q)

    dict_la = {"L_PM":8371.3e7, "L_MS":1285.7e7, "rad":4000e7, "z":0, "theta_i":0, "phi_i":0, "theta_f":theta_f,"phi_f":0}
    #dict_la = {"L_PM":8005.2e7, "L_MS":1229.5e7, "rad":4000e7, "z":0, "theta_i":0, "phi_i":0, "theta_f":theta_f,"phi_f":0}
    v_i = vio_cov_ext.k2v(k_i)
    v_f = vio_cov_ext.k2v(k_f)
    v_rot = 14400

    a = 0.761 #1.
    #dlt = {"dlt_Prad":12e7, "dlt_Pz":0, "dlt_Mrad":6e7, "dlt_Mz":0, "dlt_Srad":0, "dlt_Sz":0, "dlt_Drad":26e7, "dlt_Dz":30e7, 
    #        "dlt_theta_i":0, "dlt_theta_f":0.0065, "dlt_tP":np.divide(9, 2*6*v_rot), "dlt_tM":np.divide(3.25, 2*6*v_rot), "dlt_tD":0}
    #dlt = {"dlt_Prad":a*12e7, "dlt_Pz":a*0, "dlt_Mrad":a*6e7, "dlt_Mz":a*0, "dlt_Srad":a*0, "dlt_Sz":a*0, "dlt_Drad":a*26e7, "dlt_Dz":a*30e7, 
    #        "dlt_theta_i":0, "dlt_theta_f":0.0065, "dlt_tP":np.divide(a*9, 2*6*v_rot), "dlt_tM":np.divide(a*3.25, 2*6*v_rot), "dlt_tD":0}
    dlt = {"dlt_Prad":a*(v_i*np.divide(9, 2*6*v_rot)-12e7), "dlt_Pz":a*0, "dlt_Mrad":a*(v_i*np.divide(3.25, 2*6*v_rot)-6e7), "dlt_Mz":a*0, "dlt_Srad":a*0, "dlt_Sz":a*0, "dlt_Drad":a*26e7, "dlt_Dz":a*30e7, 
            "dlt_theta_i":0, "dlt_theta_f":0.0065, "dlt_tP":np.divide(a*9, 2*6*v_rot), "dlt_tM":np.divide(a*3.25, 2*6*v_rot), "dlt_tD":0}
    #dlt = {"dlt_Prad":a*(v_i*np.divide(9, 2*6*v_rot)-12e7), "dlt_Pz":a*0, "dlt_Mrad":a*(v_i*np.divide(3.25, 2*6*v_rot)-6e7), "dlt_Mz":a*0, "dlt_Srad":a*0, "dlt_Sz":a*0, "dlt_Drad":a*26e7, "dlt_Dz":a*30e7, 
    #        "dlt_theta_i":0, "dlt_theta_f":6.5, "dlt_tP":np.divide(a*9, 2*6*v_rot), "dlt_tM":np.divide(a*3.25, 2*6*v_rot), "dlt_tD":0}

    delta_plength = [0, 0, 0, 26e7]
    delta_angles = [0, 0, 6.5e-3, 7.5e-3]
    delta_time = [8.82e-5, 3.19e-5, 0]
    shape = 'VCYL'
    covQhw = vio_cov_ext.cov(v_i, v_f, dict_la, dlt, shape, verbose=False)
    covQhwInv = la.inv(np.divide(covQhw, helpers.sig2fwhm))
    # Going from ki, kf, Qz to Qpara, Qperp, Qz :
    Q_ki = tas.get_psi(k_i, k_f, Q, 1)
    rot = helpers.rotation_matrix_nd(-Q_ki, 4)
    covQhwInv = np.dot(rot.T, np.dot(covQhwInv, rot))

    import libs.reso as reso

    ellipses = reso.calc_ellipses(covQhwInv, verbose = False)
    #reso.plot_ellipses(ellipses, verbose = True)

#print(Ei)

#print(np.divide(a*9, 2*6*v_rot) * v_i)
#print(v_i/(10**10)*np.divide(a*9, 2*6*v_rot))
#print(a*(v_i*np.divide(9, 2*6*v_rot)-12e7)/(1e7))


#0.0106 / A, 0.0152 / A, 0.0093 / A, 0.0685 meV
#0.0105 / A, 0.0143 / A, 0.0093 / A, 0.0685 meV
#0.0104 / A, 0.0134 / A, 0.0093 / A, 0.0685 meV