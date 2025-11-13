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
sys.path.append(os.path.dirname(__file__) + "/../..")

import numpy as np
import numpy.linalg as la
import calculation.tas as tas
import libs.helpers as helpers
import algos.vio_cov_ext2 as vce2

nb=0.000941

#lbdi = 3.2
#lbdf = 2.132495017234745 #E = -10meV
lbdi = 4
lbdf = 4
#lbdi = 2
#lbdf = 2

k_i = 2*np.pi/lbdi
k_f = 2*np.pi/lbdf
v_i = vce2.k2v(k_i)
v_f = vce2.k2v(k_f)

#v_rot = 2*12000
v_rot = 2*16000

# Sample:
#Sr, Sh = 4e7, 50e7 #5e7, 50e7 #Mn-ac
#Sr, Sh = 6e7, 60e7 #Vanadium
Sr, Sh = 0.1e7, 0.1e7

# Instrument:
M = 3.5
thetacrit = M*np.arcsin(lbdi/10*np.sqrt(nb/np.pi))

thetaCP, thetaBP, wP = 9.0, 8.5, 12e7
thetaCM, thetaBM, wM = 3.25, 3.0, 6e7
Eyh, Ezh, Lpe, Lme, Les = 7e7, 27e7, 9064.7e7, 1114.3e7, 170e7
Dr, Hdet, Dw = 4000e7, 3000e7, 26e7
Hcrit, Hmax = Ezh - Les*np.tan(thetacrit), Ezh + Les*np.tan(thetacrit)
VarDr = np.divide(np.square(Dw), 12 )
VarDtheta = np.square(0.0065)#( (1 - np.divide(1, np.sqrt(1 - np.divide(np.square(Sr), np.square(Dr)))))*(2*np.divide(np.square(Dr), np.square(Sr)) - np.divide(np.square(Dw), 6*np.square(Sr)))
            #+ np.divide(2, np.sqrt(1 - np.divide(np.square(Sr), np.square(Dr)))) -1 )
print(np.sqrt(VarDtheta), 'Hcrit/max =', Hcrit/(1e7), Hmax/(1e7))

Vartp, Vartm, Vartd = np.divide( np.square(thetaCP) + np.square(thetaBP), 12*np.square(6*v_rot) ), np.divide( np.square(thetaCM) + np.square(thetaBM), 12*np.square(6*v_rot) ), np.divide( VarDr, np.square(v_f) )
VarPx = np.square( v_i*np.sqrt(Vartp) - wP )
VarPy = np.divide(1,12)*( 3*np.square(Sr) + (4*np.square(Lpe + Les) + np.square(Sr))*np.square(np.tan(thetacrit)) )
VarPz = 1
if Sh/2 <= Hcrit:
    print("Simple 1")
    VarPz = np.divide(1,12) * ( np.square(Sh) + (4*np.square(Lpe + Les) + np.square(Sr))*np.square(np.tan(thetacrit)) )
elif Hcrit < Sh/2 < Hmax:
    print("un peu grand 1")
    VarPz = np.divide(1, 36*Sh) * ( np.divide(1, np.square(Sr)*np.sqrt(np.square(Les) - np.square(Sr)))*(Sh - 2*Hcrit)*(
        (3*np.square(Hcrit) + np.square(Hcrit + Sh))*(2*Lpe*(np.square(Les) - np.square(Sr) + Lpe*Les) - np.sqrt(np.square(Les) - np.square(Sr))*(2*np.square(Lpe) - np.square(Sr) + 2*Lpe*Les))
        + 3*Ezh*(Sh + 2*Hcrit)*(2*Lpe*(np.square(Les) - np.square(Sr) - 2*Lpe*Les) + np.sqrt(np.square(Les) - np.square(Sr))*(4*np.square(Lpe) + np.square(Sr) - 2*Lpe*Les))
        + 12*np.square(Ezh)*(2*Lpe*(2*np.square(Sr) - 2*np.square(Les) + Lpe*Les) + np.sqrt(np.square(Les) - np.square(Sr))*(np.square(Sr) - 2*np.square(Lpe) + 4*Lpe*Les))
        - 3*np.tan(thetacrit)*(Sh + 2*Hcrit)*(2*np.square(Lpe)*(np.square(Les) - np.square(Sr)) + np.sqrt(np.square(Les) - np.square(Sr))*(2*Les*np.square(Sr) + Lpe*np.square(Sr) - 2*np.square(Lpe)*Les))
        - 12*np.tan(thetacrit)*Ezh*(2*np.square(Lpe)*(np.square(Sr)-np.square(Les)) + np.sqrt(np.square(Les) - np.square(Sr))*(2*np.square(Lpe)*Les + Les*np.square(Sr) + 2*Lpe*np.square(Sr)))
        + 3*np.square(Sr)*np.sqrt(np.square(Les) - np.square(Sr))*(np.square(2*Lpe + 2*Les)+np.square(Sr))*np.square(np.tan(thetacrit)))
    + 6*(4*np.power(Hcrit, 3) + Hcrit*(4*np.square(Lpe + Les) + np.square(Sr))*np.square(np.tan(thetacrit))))
else:
    print("trop grand 1")
    VarPz = np.divide(1, 18*Sh) * ( np.divide(1, np.square(Sr)*np.sqrt(np.square(Les) - np.square(Sr)))*(Hmax - Hcrit)*(
        (3*np.square(Hcrit) + np.square(Hcrit + 2*Hmax))*(2*Lpe*(np.square(Les) - np.square(Sr) + Lpe*Les) - np.sqrt(np.square(Les) - np.square(Sr))*(2*np.square(Lpe) - np.square(Sr) + 2*Lpe*Les))
        + 3*Ezh*(2*Hmax + 2*Hcrit)*(2*Lpe*(np.square(Les) - np.square(Sr) - 2*Lpe*Les) + np.sqrt(np.square(Les) - np.square(Sr))*(4*np.square(Lpe) + np.square(Sr) - 2*Lpe*Les))
        + 12*np.square(Ezh)*(2*Lpe*(2*np.square(Sr) - 2*np.square(Les) + Lpe*Les) + np.sqrt(np.square(Les) - np.square(Sr))*(np.square(Sr) - 2*np.square(Lpe) + 4*Lpe*Les))
        - 3*np.tan(thetacrit)*(2*Hmax + 2*Hcrit)*(2*np.square(Lpe)*(np.square(Les) - np.square(Sr)) + np.sqrt(np.square(Les) - np.square(Sr))*(2*Les*np.square(Sr) + Lpe*np.square(Sr) - 2*np.square(Lpe)*Les))
        - 12*np.tan(thetacrit)*Ezh*(2*np.square(Lpe)*(np.square(Sr)-np.square(Les)) + np.sqrt(np.square(Les) - np.square(Sr))*(2*np.square(Lpe)*Les + Les*np.square(Sr) + 2*Lpe*np.square(Sr)))
        + 3*np.square(Sr)*np.sqrt(np.square(Les) - np.square(Sr))*(np.square(2*Lpe + 2*Les)+np.square(Sr))*np.square(np.tan(thetacrit)))
    + 3*(4*np.power(Hcrit, 3) + Hcrit*(4*np.square(Lpe + Les) + np.square(Sr))*np.square(np.tan(thetacrit))))
VarMx = np.square( v_i*np.sqrt(Vartm) - wM )
VarMy = np.divide(1,12)*( 3*np.square(Sr) + (4*np.square(Lme + Les) + np.square(Sr))*np.square(np.tan(thetacrit)) )
VarMz = 1
if Sh/2 <= Hcrit:
    VarMz = np.divide(1,12) * ( np.square(Sh) + (4*np.square(Lme + Les) + np.square(Sr))*np.square(np.tan(thetacrit)) )
    print("Simple 2")
elif Hcrit < Sh/2 < Hmax:
    print("un peu grand 2")
    VarMz = np.divide(1, 36*Sh) * ( np.divide(1, np.square(Sr)*np.sqrt(np.square(Les) - np.square(Sr)))*(Sh - 2*Hcrit)*(
        (3*np.square(Hcrit) + np.square(Hcrit + Sh))*(2*Lme*(np.square(Les) - np.square(Sr) + Lme*Les) - np.sqrt(np.square(Les) - np.square(Sr))*(2*np.square(Lme) - np.square(Sr) + 2*Lme*Les))
        + 3*Ezh*(Sh + 2*Hcrit)*(2*Lme*(np.square(Les) - np.square(Sr) - 2*Lme*Les) + np.sqrt(np.square(Les) - np.square(Sr))*(4*np.square(Lme) + np.square(Sr) - 2*Lme*Les))
        + 12*np.square(Ezh)*(2*Lme*(2*np.square(Sr) - 2*np.square(Les) + Lme*Les) + np.sqrt(np.square(Les) - np.square(Sr))*(np.square(Sr) - 2*np.square(Lme) + 4*Lme*Les))
        - 3*np.tan(thetacrit)*(Sh + 2*Hcrit)*(2*np.square(Lme)*(np.square(Les) - np.square(Sr)) + np.sqrt(np.square(Les) - np.square(Sr))*(2*Les*np.square(Sr) + Lme*np.square(Sr) - 2*np.square(Lme)*Les))
        - 12*np.tan(thetacrit)*Ezh*(2*np.square(Lme)*(np.square(Sr)-np.square(Les)) + np.sqrt(np.square(Les) - np.square(Sr))*(2*np.square(Lme)*Les + Les*np.square(Sr) + 2*Lme*np.square(Sr)))
        + 3*np.square(Sr)*np.sqrt(np.square(Les) - np.square(Sr))*(np.square(2*Lme + 2*Les)+np.square(Sr))*np.square(np.tan(thetacrit)))
    + 6*(4*np.power(Hcrit, 3) + Hcrit*(4*np.square(Lme + Les) + np.square(Sr))*np.square(np.tan(thetacrit))))
else:
    print("trop grand 2")
    VarMz = np.divide(1, 18*Sh) * ( np.divide(1, np.square(Sr)*np.sqrt(np.square(Les) - np.square(Sr)))*(Hmax - Hcrit)*(
        (3*np.square(Hcrit) + np.square(Hcrit + 2*Hmax))*(2*Lme*(np.square(Les) - np.square(Sr) + Lme*Les) - np.sqrt(np.square(Les) - np.square(Sr))*(2*np.square(Lme) - np.square(Sr) + 2*Lme*Les))
        + 3*Ezh*(2*Hmax + 2*Hcrit)*(2*Lme*(np.square(Les) - np.square(Sr) - 2*Lme*Les) + np.sqrt(np.square(Les) - np.square(Sr))*(4*np.square(Lme) + np.square(Sr) - 2*Lme*Les))
        + 12*np.square(Ezh)*(2*Lme*(2*np.square(Sr) - 2*np.square(Les) + Lme*Les) + np.sqrt(np.square(Les) - np.square(Sr))*(np.square(Sr) - 2*np.square(Lme) + 4*Lme*Les))
        - 3*np.tan(thetacrit)*(2*Hmax + 2*Hcrit)*(2*np.square(Lme)*(np.square(Les) - np.square(Sr)) + np.sqrt(np.square(Les) - np.square(Sr))*(2*Les*np.square(Sr) + Lme*np.square(Sr) - 2*np.square(Lme)*Les))
        - 12*np.tan(thetacrit)*Ezh*(2*np.square(Lme)*(np.square(Sr)-np.square(Les)) + np.sqrt(np.square(Les) - np.square(Sr))*(2*np.square(Lme)*Les + Les*np.square(Sr) + 2*Lme*np.square(Sr)))
        + 3*np.square(Sr)*np.sqrt(np.square(Les) - np.square(Sr))*(np.square(2*Lme + 2*Les)+np.square(Sr))*np.square(np.tan(thetacrit)))
    + 3*(4*np.power(Hcrit, 3) + Hcrit*(4*np.square(Lme + Les) + np.square(Sr))*np.square(np.tan(thetacrit))))
VarSx, VarSy, VarSz = np.divide(np.square(Sr), 4), np.divide(np.square(Sr), 4), np.divide(np.square(Sh), 12)
VarDz = np.divide(1, 1) * np.square(np.divide(Hdet, 100))
Var = [VarPx, VarPy, VarPz, VarMx, VarMy, VarMz, VarSx, VarSy, VarSz, 1, 1, VarDz, Vartp, Vartm, Vartd]

covInstr = np.eye(15)
for i in range(15):
    covInstr[i][i] = Var[i]

print(VarPx, VarMx)

sigPx = np.sqrt(VarPx)
sigMx = np.sqrt(VarMx)
LPM, LPMx, LPMy, LPMz, LMS, LMSx, LMSy, LMSz = vce2.length(Sr, Sh, Lpe, Lme, Les, Eyh, Ezh, -(Lpe+Les), sigPx, -(Lme+Les), sigMx)
LSD, LSDz = Dr, 0

#l_Qc = [1.64, 1.66, 1.68, 1.70, 1.72, 1.74, 1.76, 1.78, 1.80, 1.82, 1.84, 1.86, 1.88, 1.90, 1.92, 1.94, 1.96, 1.98, 2.00, 2.02, 2.04, 2.06, 2.08, 2.10, 2.12, 2.14, 2.16, 2.18, 2.20, 2.22, 2.24, 2.26, 2.28, 2.30, 2.32, 2.34, 2.36]
l_Qc = [0]
for Qc in l_Qc:
    print()
    Q = np.sqrt(0 + 1 + np.square(Qc))
    print("Qc = ", Qc, " ; Q = ", Q)
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
    reso.plot_ellipses(ellipses, verbose = True)

