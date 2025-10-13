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
import algos.vio_cov_ext2_1 as vce2


#
# resolution algorithm
#

def calc(param):
    verbose = param["verbose"]
    ki, kf, Q = param["ki"], param["kf"], param["Q"]
    theta_f = tas.get_scattering_angle(ki, kf, Q)
    vi, vf = vce2.k2v(ki), vce2.k2v(kf)
    det_shape = param["det_shape"]
    if det_shape not in ('VCYL'): # 'SPHERE' will be added
        print("this shape is not taken in account")
        return None
    v_rot = param["v_rot"]
    Sr, Sh = param["sample_radius"], param["sample_height"]
    thetaCP, thetaBP, wP = param["windows_angle_chopper_P"], param["beam_angle_chopper_P"], param["width_chopper_P"]
    thetaCM, thetaBM, wM = param["windows_angle_chopper_M"], param["beam_angle_chopper_M"], param["width_chopper_M"]
    Eyh, Ezh, Lpe, Lme, Les = param["end_of_guide_y_height"]/2, param["end_of_guide_z_height"]/2, param["distance_P_EG"], param["distance_M_EG"], param["distance_EG_S"]
    Dr, Hdet, Dw = param["detector_radius"], param["detector_height"], param["tube_diameter"]
    
    VarDr = np.divide(np.square(Dw), 12)
    VarDtheta = np.square( np.divide(2*Dw*( Dr - np.sqrt( np.square(Dr) - np.square(Sr) ) ), np.square(Sr)) )
    Vartp, Vartm, Vartd = np.divide( np.square(thetaCP) + np.square(thetaBP), 12*np.square(6*v_rot) ), np.divide( np.square(thetaCM) + np.square(thetaBM), 12*np.square(6*v_rot) ), np.divide( VarDr, np.square(vf) )
    VarPx, slopePx, distribPx = 1, 1, 'Trapeze'
    if wP < vi*np.divide(thetaCP - thetaBP, 6*v_rot):
        VarPx = np.divide(np.square(vi)*( np.square(thetaCP) + np.square(thetaBP) ) - 2*wP*vi*thetaCP*6*v_rot + np.square(wP)*np.square(6*v_rot), 12*np.square(6*v_rot))
        slopePx = np.divide(np.square(6*v_rot), np.square(vi)*thetaBP*thetaCP - wP*vi*thetaBP*(6*v_rot))
    else:
        VarPx = np.divide( np.square(vi*(thetaCP + thetaBP) - wP*6*v_rot), 24*np.square(6*v_rot) )
        distribPx = 'Triangle'
    VarPy = ( np.divide(np.square(Eyh), 3)
            + np.square(Lpe)*( 2*np.divide(np.square(Les), np.square(Sr))*(1 - np.sqrt(1 - np.divide(np.square(Sr), np.square(Les)))) - 1 )
            + np.divide(4*Lpe*Les, 3*np.square(Sr))*np.square(Eyh)*(1 - np.sqrt(1 - np.divide(np.square(Sr), np.square(Les))))
            + np.divide(2*np.square(Lpe), 3*np.square(Sr))*np.square(Eyh)*(np.divide(1, np.sqrt(1 - np.divide(np.square(Sr), np.square(Les)))) - 1) )
    VarPz = ( np.divide(np.square(Ezh), 3)
            + np.divide(np.square(Lpe), 3*np.square(Sr))*(np.divide(np.square(Sh), 2) + 2*np.square(Ezh))*(np.divide(1, np.sqrt(1 - np.divide(np.square(Sr), np.square(Les)))) - 1) 
            + np.divide(4*Lpe*Les, 3*np.square(Sr))*np.square(Ezh)*(1 - np.sqrt(1 - np.divide(np.square(Sr), np.square(Les)))) )
    VarMx, slopeMx, distribMx = 1, 1, 'Trapeze'
    if wM < vi*np.divide(thetaCM - thetaBM, 6*v_rot):
        VarMx = np.divide(np.square(vi)*( np.square(thetaCM) + np.square(thetaBM) ) - 2*wM*vi*thetaCM*6*v_rot + np.square(wM)*np.square(6*v_rot), 12*np.square(6*v_rot))
        slopeMx = np.divide(np.square(6*v_rot), np.square(vi)*thetaBM*thetaCM - wM*vi*thetaBM*(6*v_rot))
    else :
        VarMx = np.divide( np.square(vi*(thetaCM + thetaBM) - wM*6*v_rot), 24*np.square(6*v_rot) )
        distribMx = 'Triangle'
    VarMy = ( np.divide(np.square(Eyh), 3)
            + np.square(Lme)*( 2*np.divide(np.square(Les), np.square(Sr))*(1 - np.sqrt(1 - np.divide(np.square(Sr), np.square(Les)))) - 1 )
            + np.divide(4*Lme*Les, 3*np.square(Sr))*np.square(Eyh)*(1 - np.sqrt(1 - np.divide(np.square(Sr), np.square(Les))))
            + np.divide(2*np.square(Lme), 3*np.square(Sr))*np.square(Eyh)*(np.divide(1, np.sqrt(1 - np.divide(np.square(Sr), np.square(Les)))) - 1) )
    VarMz = ( np.divide(np.square(Ezh), 3)
            + np.divide(np.square(Lme), 3*np.square(Sr))*(np.divide(np.square(Sh), 2) + 2*np.square(Ezh))*(np.divide(1, np.sqrt(1 - np.divide(np.square(Sr), np.square(Les)))) - 1) 
            + np.divide(4*Lme*Les, 3*np.square(Sr))*np.square(Ezh)*(1 - np.sqrt(1 - np.divide(np.square(Sr), np.square(Les)))) )
    VarSx, VarSy, VarSz = np.divide(np.square(Sr), 4), np.divide(np.square(Sr), 4), np.divide(np.square(Sh), 12)
    VarDx = np.square(np.cos(theta_f))*VarDr + np.square(Dr)*np.square(np.sin(theta_f))*VarDtheta
    VarDy = np.square(np.sin(theta_f))*VarDr + np.square(Dr)*np.square(np.cos(theta_f))*VarDtheta
    VarDz = np.square(np.divide(Hdet, 100))
    CovDxDy = np.cos(theta_f)*np.sin(theta_f)*VarDr - np.square(Dr)*np.cos(theta_f)*np.sin(theta_f)*VarDtheta
    covInstr = vce2.covInstrument(VarPx, VarPy, VarPz, VarMx, VarMy, VarMz, VarSx, VarSy, VarSz, VarDx, VarDy, VarDz, Vartp, Vartm, Vartd, CovDxDy)

    LPM, LPMx, LPMy, LPMz, LMS, LMSx, LMSy, LMSz = vce2.length(Sr, Sh, Lpe, Lme, Les, Eyh, Ezh, -(Lpe+Les), VarPx, slopePx, distribPx, -(Lme+Les), VarMx, slopeMx, distribMx)
    LSD, LSDz = Dr, param["det_z"]
    LSDx, LSDy = LSD*np.cos(theta_f), LSD*np.sin(theta_f)
    dict_L = {"L_PM":LPM, "L_PMx":LPMx, "L_PMy":LPMy, "L_PMz":LPMz, "L_MS":LMS, "L_MSx":LMSx, "L_MSy":LMSy, "L_MSz":LMSz, "L_SD":LSD, "L_SDx":LSDx, "L_SDy":LSDy, "L_SDz":LSDz}
    
    E = tas.get_E(ki, kf)
    covQhw = vce2.cov(vi, vf, dict_L, covInstr, det_shape, verbose)
    covQhwInv = la.inv(covQhw)
    Q_ki = tas.get_psi(ki, kf, Q, param["sample_sense"])
    rot = helpers.rotation_matrix_nd(-Q_ki, 4)
    covQhwInv_r = np.dot(rot.T, np.dot(covQhwInv, rot))
    
    ###########################################################################################
    if(verbose):
        print("ki =", ki, "; kf =", kf, "; Q =", Q, "theta_f =", theta_f, "E =", E)
        print("det_shape =", det_shape)
        print("vi =", vi, "; vf =", vf)
        print("dict_L =", dict_L)
        print("covQhw =", covQhw)
        print("covQhwInv =", covQhwInv)
        print("In the base (Qpara, Qperp, Qz) :")
        print("rot =", rot,'\ncovQhwInv =', covQhwInv_r)

    res={}
    res["ki"] = ki
    res["kf"] = kf
    res["Q"] = Q
    res["E"] = E
    res["reso"] = covQhwInv_r
    res["ok"] = True
    return res

print(vce2.k2v(2*np.pi/6)*1e-7)
print(np.divide( 9 - 8.5, 6*14000 ))
print(np.divide( 0.25, 6*14000 )*vce2.k2v(2*np.pi/2)*1e-7)

