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
import algos.vio_cov_ext2 as vce2
import algos.old.vio_cov_ext2_1 as vce21


test = 'vio_ext21'

if test == 'vio_ext2':
    #LPM, LPMx, LPMy, LPMz, LMS, LMSx, LMSy, LMSz = vce2.length(0.1, 0.1, 9064.7, 1114.3, 170, 7, 27, -9234.7, sigPx, -1284.3, sigMx)

    #k_i = 2*np.pi/5.9
    #k_f = 2*np.pi/5.9
    k_i = 2*np.pi/6
    k_f = 2*np.pi/6
    #k_i = 3
    #k_f = 3
    v_i = vce2.k2v(k_i)
    v_f = vce2.k2v(k_f)
    #v_rot = 2*14400
    v_rot = 2*12000
    

    #Sr, Sh = 4e7, 50e7 #5e7, 50e7 #Mn-ac
    Sr, Sh = 6e7, 60e7 #Vanadium
    thetaCP, thetaBP, wP = 9.0, 8.5, 12e7
    thetaCM, thetaBM, wM = 3.25, 3.0, 6e7
    Eyh, Ezh, Lpe, Lme, Les = 7e7, 27e7, 9064.7e7, 1114.3e7, 170e7
    Dr, Hdet, Dw = 4000e7, 3000e7, 26e7
    VarDr = np.divide(np.square(Dw), 12 )
    VarDtheta = np.square(0.0065)#( (1 - np.divide(1, np.sqrt(1 - np.divide(np.square(Sr), np.square(Dr)))))*(2*np.divide(np.square(Dr), np.square(Sr)) - np.divide(np.square(Dw), 6*np.square(Sr)))
                #+ np.divide(2, np.sqrt(1 - np.divide(np.square(Sr), np.square(Dr)))) -1 )
    print(np.sqrt(VarDtheta))

    Vartp, Vartm, Vartd = np.divide( np.square(thetaCP) + np.square(thetaBP), 12*np.square(6*v_rot) ), np.divide( np.square(thetaCM) + np.square(thetaBM), 12*np.square(6*v_rot) ), np.divide( VarDr, np.square(v_f) )
    VarPx = np.square( v_i*np.sqrt(Vartp) - wP )
    VarPy = ( np.divide(np.square(Eyh), 3)
            + np.square(Lpe)*( 2*np.divide(np.square(Les), np.square(Sr))*(1 - np.sqrt(1 - np.divide(np.square(Sr), np.square(Les)))) - 1 )
            + np.divide(4*Lpe*Les, 3*np.square(Sr))*np.square(Eyh)*(1 - np.sqrt(1 - np.divide(np.square(Sr), np.square(Les))))
            + np.divide(2*np.square(Lpe), 3*np.square(Sr))*np.square(Eyh)*(np.divide(1, np.sqrt(1 - np.divide(np.square(Sr), np.square(Les)))) - 1) )
    VarPz = ( np.divide(np.square(Ezh), 3)
            + np.divide(np.square(Lpe), 3*np.square(Sr))*(np.divide(np.square(Sh), 2) + 2*np.square(Ezh))*(np.divide(1, np.sqrt(1 - np.divide(np.square(Sr), np.square(Les)))) - 1) 
            + np.divide(4*Lpe*Les, 3*np.square(Sr))*np.square(Ezh)*(1 - np.sqrt(1 - np.divide(np.square(Sr), np.square(Les)))) )
    VarMx = np.square( v_i*np.sqrt(Vartm) - wM )
    VarMy = ( np.divide(np.square(Eyh), 3)
            + np.square(Lme)*( 2*np.divide(np.square(Les), np.square(Sr))*(1 - np.sqrt(1 - np.divide(np.square(Sr), np.square(Les)))) - 1 )
            + np.divide(4*Lme*Les, 3*np.square(Sr))*np.square(Eyh)*(1 - np.sqrt(1 - np.divide(np.square(Sr), np.square(Les))))
            + np.divide(2*np.square(Lme), 3*np.square(Sr))*np.square(Eyh)*(np.divide(1, np.sqrt(1 - np.divide(np.square(Sr), np.square(Les)))) - 1) )
    VarMz = ( np.divide(np.square(Ezh), 3)
            + np.divide(np.square(Lme), 3*np.square(Sr))*(np.divide(np.square(Sh), 2) + 2*np.square(Ezh))*(np.divide(1, np.sqrt(1 - np.divide(np.square(Sr), np.square(Les)))) - 1) 
            + np.divide(4*Lme*Les, 3*np.square(Sr))*np.square(Ezh)*(1 - np.sqrt(1 - np.divide(np.square(Sr), np.square(Les)))) )
    VarSx, VarSy, VarSz = np.divide(np.square(Sr), 4), np.divide(np.square(Sr), 4), np.divide(np.square(Sh), 12)
    VarDz = np.divide(1, 12) * np.square(np.divide(Hdet, 100))
    Var = [VarPx, VarPy, VarPz, VarMx, VarMy, VarMz, VarSx, VarSy, VarSz, 1, 1, VarDz, Vartp, Vartm, Vartd]

    covInstr = np.eye(15)
    for i in range(15):
        covInstr[i][i] = Var[i]

    sigPx = np.sqrt(VarPx)
    sigMx = np.sqrt(VarMx)
    LPM, LPMx, LPMy, LPMz, LMS, LMSx, LMSy, LMSz = vce2.length(Sr, Sh, Lpe, Lme, Les, Eyh, Ezh, -(Lpe+Les), sigPx, -(Lme+Les), sigMx)
    print(LPM, LMS)
    LSD, LSDz = Dr, 0

    l_Q = [0.632, 0.734, 0.969, 1.158, 1.214, 1.265, 1.421, 1.463, 1.552, 1.597]
    #l_Q = [1.915, 1.502, 1.11, 0.74, 0.46, 0.117]
    #l_Q = [1]
    for Q in l_Q:
        print()
        print("Q = ", Q)
        theta_f = tas.get_scattering_angle(k_i, k_f, Q)

        LSDx, LSDy = LSD*np.cos(theta_f), LSD*np.sin(theta_f)
        VarDx = np.square(np.cos(theta_f))*VarDr + np.square(Dr)*np.square(np.sin(theta_f))*VarDtheta
        VarDy = np.square(np.sin(theta_f))*VarDr + np.square(Dr)*np.square(np.cos(theta_f))*VarDtheta
        CovDxDy = np.cos(theta_f)*np.sin(theta_f)*VarDr - np.square(Dr)*np.cos(theta_f)*np.sin(theta_f)*VarDtheta
        covInstr[9][9], covInstr[10][10], covInstr[9][10], covInstr[10][9] = VarDx, VarDy, CovDxDy, CovDxDy

        dict_la = {"L_PM":LPM, "L_PMx":LPMx, "L_PMy":LPMy, "L_PMz":LPMz, "L_MS":LMS, "L_MSx":LMSx, "L_MSy":LMSy, "L_MSz":LMSz, "L_SD":LSD, "L_SDx":LSDx, "L_SDy":LSDy, "L_SDz":LSDz}
        shape = 'VCYL'
        covQhw = vce2.cov(v_i, v_f, dict_la, covInstr, shape, False)
        covQhwInv = la.inv(covQhw)
        # Going from ki, kf, Qz to Qpara, Qperp, Qz :
        Q_ki = tas.get_psi(k_i, k_f, Q, 1)
        rot = helpers.rotation_matrix_nd(-Q_ki, 4)
        covQhwInv = np.dot(rot.T, np.dot(covQhwInv, rot))

        import libs.reso as reso

        #ellipses = reso.calc_ellipses(covQhwInv, verbose = False)
        ellipses = reso.calc_ellipses(covQhwInv, verbose = True)
        #reso.plot_ellipses(ellipses, verbose = True)

    #print(Ei)

    #print(np.divide(a*9, 2*6*v_rot) * v_i)
    #print(v_i/(10**10)*np.divide(a*9, 2*6*v_rot))
    #print(a*(v_i*np.divide(9, 2*6*v_rot)-12e7)/(1e7))


    #0.0106 / A, 0.0152 / A, 0.0093 / A, 0.0685 meV
    #0.0105 / A, 0.0143 / A, 0.0093 / A, 0.0685 meV
    #0.0104 / A, 0.0134 / A, 0.0093 / A, 0.0685 meV


if test == 'vio_ext21':
    ki = 2*np.pi/6
    kf = 2*np.pi/6
    vi = vce21.k2v(ki)
    vf = vce21.k2v(kf)
    v_rot = 2*12000  #2*14400
    
    #Sr, Sh = 4e7, 50e7 #5e7, 50e7 #Mn-ac
    Sr, Sh = 6e7, 60e7 #Vanadium
    thetaCP, thetaBP, wP = 9.0, 8.5, 12e7
    thetaCM, thetaBM, wM = 3.25, 3.0, 6e7
    Eyh, Ezh, Lpe, Lme, Les = 7e7, 27e7, 9064.7e7, 1114.3e7, 170e7
    Dr, Hdet, Dw = 4000e7, 3000e7, 26e7
    VarDr = np.divide(np.square(Dw), 12 )
    VarDtheta = np.square( np.divide(2*Dw*( Dr - np.sqrt( np.square(Dr) - np.square(Sr) ) ), np.square(Sr)) )
    print(np.sqrt(VarDtheta))
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
    VarDz = np.square(np.divide(Hdet, 100))
    
    LPM, LPMx, LPMy, LPMz, LMS, LMSx, LMSy, LMSz = vce21.length(Sr, Sh, Lpe, Lme, Les, Eyh, Ezh, -(Lpe+Les), VarPx, slopePx, distribPx, -(Lme+Les), VarMx, slopeMx, distribMx)
    print(LPM, LMS)
    LSD, LSDz = Dr, 0
    l_Q = [0.632, 0.734, 0.969, 1.158, 1.214, 1.265, 1.421, 1.463, 1.552, 1.597]
    #l_Q = [1.915, 1.502, 1.11, 0.74, 0.46, 0.117]
    #l_Q = [1]
    for Q in l_Q:
        print()
        print("Q = ", Q)
        theta_f = tas.get_scattering_angle(ki, kf, Q)

        LSDx, LSDy = LSD*np.cos(theta_f), LSD*np.sin(theta_f)
        VarDx = np.square(np.cos(theta_f))*VarDr + np.square(Dr)*np.square(np.sin(theta_f))*VarDtheta
        VarDy = np.square(np.sin(theta_f))*VarDr + np.square(Dr)*np.square(np.cos(theta_f))*VarDtheta
        CovDxDy = np.cos(theta_f)*np.sin(theta_f)*VarDr - np.square(Dr)*np.cos(theta_f)*np.sin(theta_f)*VarDtheta
        covInstr = vce21.covInstrument(VarPx, VarPy, VarPz, VarMx, VarMy, VarMz, VarSx, VarSy, VarSz, VarDx, VarDy, VarDz, Vartp, Vartm, Vartd, CovDxDy)
        dict_la = {"L_PM":LPM, "L_PMx":LPMx, "L_PMy":LPMy, "L_PMz":LPMz, "L_MS":LMS, "L_MSx":LMSx, "L_MSy":LMSy, "L_MSz":LMSz, "L_SD":LSD, "L_SDx":LSDx, "L_SDy":LSDy, "L_SDz":LSDz}
        shape = 'VCYL'
        covQhw = vce21.cov(vi, vf, dict_la, covInstr, shape, False)
        covQhwInv = la.inv(covQhw)
        # Going from ki, kf, Qz to Qpara, Qperp, Qz :
        Q_ki = tas.get_psi(ki, kf, Q, 1)
        rot = helpers.rotation_matrix_nd(-Q_ki, 4)
        covQhwInv = np.dot(rot.T, np.dot(covQhwInv, rot))

        import libs.reso as reso

        ellipses = reso.calc_ellipses(covQhwInv, verbose = True)
        #reso.plot_ellipses(ellipses, verbose = True)
    print(distribPx, distribMx)
        
print(np.sqrt(VarPx)*1e-7, np.sqrt(VarMx)*1e-7)

# lambda = 2, Triangle, Triangle
# vio_ext_2 : VarPx = 37.08820454670751 ; VarMx = 11.538364163866618
# vio_ext_21 : VarPx = 46.61869106066955 ; VarMx = 16.29960541555581                   1.256968 ; 1.412644

# lambda = 1, Trapeze, Trapeze
# vio_ext_2 : VarPx = 86.17640909341502 ; VarMx = 29.076728327733235
# vio_ext_21 : VarPx = 95.68752818925071 ; VarMx = 33.82441988152677                   1.110368 ; 1.163281

# lambda = 6, Triangle, Triangle
# vio_ext_2 : VarPx = 4.362734848902503 ; VarMx = 0.15387861204446107
# vio_ext_21 : VarPx = 13.906570525034395 ; VarMx = 4.616705224257544                   1.110368 ; 1.163281
