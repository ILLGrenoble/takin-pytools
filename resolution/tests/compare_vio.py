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
import libs.reso as reso

l_vio_coh_mod1, l_vio_coh_mod2, l_vio_coh_mod3, l_vio_coh_mod4, l_vio_coh_mod5 = [], [], [], [], []
l_vioext_coh_mod1, l_vioext_coh_mod2, l_vioext_coh_mod3, l_vioext_coh_mod4 = [], [], [], []
l_vioext2_PS_coh_mod1, l_vioext2_PS_coh_mod2, l_vioext2_PS_coh_mod3 = [], [], []
l_vioext2_VS_coh_mod1, l_vioext2_VS_coh_mod2, l_vioext2_VS_coh_mod3 = [], [], []

l_vio_inc_mod1, l_vio_inc_mod2, l_vio_inc_mod3, l_vio_inc_mod4, l_vio_inc_mod5 = [], [], [], [], []
l_vioext_inc_mod1, l_vioext_inc_mod2, l_vioext_inc_mod3, l_vioext_inc_mod4 = [], [], [], []
l_vioext2_PS_inc_mod1, l_vioext2_PS_inc_mod2, l_vioext2_PS_inc_mod3 = [], [], []
l_vioext2_VS_inc_mod1, l_vioext2_VS_inc_mod2, l_vioext2_VS_inc_mod3 = [], [], []


l_lambda = [1, 3, 5, 10, 12, 20]
lbdi = 6
lbdf = lbdi
l_Q = [0.5, 1]
printEllipse = False
for Q in l_Q:
    v_rot = 12000
    a = 1  # 2*np.sqrt(3)

    k_i = 2*np.pi/lbdi
    k_f = 2*np.pi/lbdf
    v_i = vc.k2v(k_i)
    v_f = vc.k2v(k_f)
    theta_f = tas.get_scattering_angle(k_i, k_f, Q)
    Q_ki = tas.get_psi(k_i, k_f, Q, 1)
    rot = helpers.rotation_matrix_nd(-Q_ki, 4)
    det_shape = 'VCYL'


    print('---------------------------------------- vio.py ----------------------------------------')
    # ---------------------------------------- vio.py ----------------------------------------
    LP12, LP2M1, LM12, LM2S, Rd = 149.8e7, 7800.6e7, 54.8e7, 1229.5e7, 4000.e7
    aP, aM, v_rotmin, v_rotmax = 9., 3.25, 7000, 17000
    dRd, dthetaf, dDz = 26e7, 0.0065, 30e7
    # 1er modele:
    print('\n', 'PREMIER MODELE', '\n')
    delta_tp, delta_tm, delta_td = np.divide(aP, 2*6*v_rot), np.divide(aM, 2*6*v_rot), 0
    dPM, dMS= 0, 0
    d_geo = {"dist_PM":[LP12, 0., LP2M1, dPM, LM12, 0.], "dist_MS":[LM2S, dMS], "dist_SD":[Rd, dRd, 0, dDz], "angles":[0, 0, 0, 0, theta_f, dthetaf], "delta_time_detector":delta_td}
    d_choppers = {"chopperP":[np.deg2rad(aP), np.divide(v_rotmin*np.pi, 30), np.divide(v_rotmax*np.pi, 30), np.divide(v_rot*np.pi, 30), delta_tp],
                "chopperM":[np.deg2rad(aM), np.divide(v_rotmin*np.pi, 30), np.divide(v_rotmax*np.pi, 30), np.divide(v_rot*np.pi, 30), delta_tm]}
    covQhwVio = vc.cov(d_geo, d_choppers, v_i, v_f, det_shape, False)
    covQhwInvVio = la.inv(np.divide(covQhwVio, helpers.sig2fwhm))
    covQhwInvVio = np.dot(rot.T, np.dot(covQhwInvVio, rot))

    ellipses = reso.calc_ellipses(covQhwInvVio, verbose = True)
    if print(ellipses):
        reso.plot_ellipses(ellipses, verbose = True)
    l_vio_coh_mod1.append( reso.calc_coh_fwhms(covQhwInvVio) )
    l_vio_inc_mod1.append( reso.calc_incoh_fwhms(covQhwInvVio) )

    # 2e modele:
    print('\n', 'DEUXIEME MODELE', '\n')
    a = 0.761
    delta_tp, delta_tm, delta_td = a*np.divide(aP, 2*6*v_rot), a*np.divide(aM, 2*6*v_rot), 0
    dPM, dMS= 0, 0
    d_geo = {"dist_PM":[LP12, 0., LP2M1, dPM, LM12, 0.], "dist_MS":[LM2S, dMS], "dist_SD":[Rd, dRd, 0, dDz], "angles":[0, 0, 0, 0, theta_f, dthetaf], "delta_time_detector":delta_td}
    d_choppers = {"chopperP":[np.deg2rad(aP), np.divide(v_rotmin*np.pi, 30), np.divide(v_rotmax*np.pi, 30), np.divide(v_rot*np.pi, 30), delta_tp],
                "chopperM":[np.deg2rad(aM), np.divide(v_rotmin*np.pi, 30), np.divide(v_rotmax*np.pi, 30), np.divide(v_rot*np.pi, 30), delta_tm]}
    covQhwVio = vc.cov(d_geo, d_choppers, v_i, v_f, det_shape, False)
    covQhwInvVio = la.inv(np.divide(covQhwVio, helpers.sig2fwhm))
    covQhwInvVio = np.dot(rot.T, np.dot(covQhwInvVio, rot))

    ellipses = reso.calc_ellipses(covQhwInvVio, verbose = True)
    if print(ellipses):
        reso.plot_ellipses(ellipses, verbose = True)
    l_vio_coh_mod2.append( reso.calc_coh_fwhms(covQhwInvVio) )
    l_vio_inc_mod2.append( reso.calc_incoh_fwhms(covQhwInvVio) )

    # 3e modele: (Only for vio.py)
    print('\n', 'TROISIEME MODELE', '\n')
    a = 0.761
    delta_tp, delta_tm, delta_td = a*np.divide(aP, 2*6*v_rot), a*np.divide(aM, 2*6*v_rot), 0
    dPM, dMS= np.sqrt(12**2 + 6**2)*1e7, 6e7
    d_geo = {"dist_PM":[LP12, 0., LP2M1, dPM, LM12, 0.], "dist_MS":[LM2S, dMS], "dist_SD":[Rd, dRd, 0, dDz], "angles":[0, 0, 0, 0, theta_f, dthetaf], "delta_time_detector":delta_td}
    d_choppers = {"chopperP":[np.deg2rad(aP), np.divide(v_rotmin*np.pi, 30), np.divide(v_rotmax*np.pi, 30), np.divide(v_rot*np.pi, 30), delta_tp],
                "chopperM":[np.deg2rad(aM), np.divide(v_rotmin*np.pi, 30), np.divide(v_rotmax*np.pi, 30), np.divide(v_rot*np.pi, 30), delta_tm]}
    covQhwVio = vc.cov(d_geo, d_choppers, v_i, v_f, det_shape, False)
    covQhwInvVio = la.inv(np.divide(covQhwVio, helpers.sig2fwhm))
    covQhwInvVio = np.dot(rot.T, np.dot(covQhwInvVio, rot))

    ellipses = reso.calc_ellipses(covQhwInvVio, verbose = True)
    if print(ellipses):
        reso.plot_ellipses(ellipses, verbose = True)
    l_vio_coh_mod3.append( reso.calc_coh_fwhms(covQhwInvVio) )
    l_vio_inc_mod3.append( reso.calc_incoh_fwhms(covQhwInvVio) )

    # 4e modele:
    print('\n', 'QUATRIEME MODELE', '\n')
    a = 0.761
    delta_tp, delta_tm, delta_td = a*np.divide(aP, 2*6*v_rot), a*np.divide(aM, 2*6*v_rot), np.divide(dRd, v_f)
    dPM, dMS= np.sqrt( np.square(v_i*delta_tp - 12e7) + np.square(v_i*delta_tm - 6e7) ), v_i*delta_tm - 6e7
    d_geo = {"dist_PM":[LP12, 0., LP2M1, dPM, LM12, 0.], "dist_MS":[LM2S, dMS], "dist_SD":[Rd, dRd, 0, dDz], "angles":[0, 0, 0, 0, theta_f, dthetaf], "delta_time_detector":delta_td}
    d_choppers = {"chopperP":[np.deg2rad(aP), np.divide(v_rotmin*np.pi, 30), np.divide(v_rotmax*np.pi, 30), np.divide(v_rot*np.pi, 30), delta_tp],
                "chopperM":[np.deg2rad(aM), np.divide(v_rotmin*np.pi, 30), np.divide(v_rotmax*np.pi, 30), np.divide(v_rot*np.pi, 30), delta_tm]}
    covQhwVio = vc.cov(d_geo, d_choppers, v_i, v_f, det_shape, False)
    covQhwInvVio = la.inv(np.divide(covQhwVio, helpers.sig2fwhm))
    covQhwInvVio = np.dot(rot.T, np.dot(covQhwInvVio, rot))

    ellipses = reso.calc_ellipses(covQhwInvVio, verbose = True)
    if print(ellipses):
        reso.plot_ellipses(ellipses, verbose = True)
    l_vio_coh_mod4.append( reso.calc_coh_fwhms(covQhwInvVio) )
    l_vio_inc_mod4.append( reso.calc_incoh_fwhms(covQhwInvVio) )

    # 5e modele:
    print('\n', 'CINQUIEME MODELE', '\n')
    a= np.divide(1, 2*np.sqrt(3))
    delta_tp, delta_tm, delta_td = a*np.divide(aP, 2*6*v_rot), a*np.divide(aM, 2*6*v_rot), np.divide(dRd, v_f)
    dPM, dMS= np.sqrt( np.square(v_i*delta_tp - 12e7) + np.square(v_i*delta_tm - 6e7) ), v_i*delta_tm - 6e7
    d_geo = {"dist_PM":[LP12, 0., LP2M1, dPM, LM12, 0.], "dist_MS":[LM2S, dMS], "dist_SD":[Rd, dRd, 0, dDz], "angles":[0, 0, 0, 0, theta_f, dthetaf], "delta_time_detector":delta_td}
    d_choppers = {"chopperP":[np.deg2rad(aP), np.divide(v_rotmin*np.pi, 30), np.divide(v_rotmax*np.pi, 30), np.divide(v_rot*np.pi, 30), delta_tp],
                "chopperM":[np.deg2rad(aM), np.divide(v_rotmin*np.pi, 30), np.divide(v_rotmax*np.pi, 30), np.divide(v_rot*np.pi, 30), delta_tm]}
    covQhwVio = vc.cov(d_geo, d_choppers, v_i, v_f, det_shape, False)
    covQhwInvVio = la.inv(covQhwVio)
    covQhwInvVio = np.dot(rot.T, np.dot(covQhwInvVio, rot))

    ellipses = reso.calc_ellipses(covQhwInvVio, verbose = True)
    if print(ellipses):
        reso.plot_ellipses(ellipses, verbose = True)
    l_vio_coh_mod5.append( reso.calc_coh_fwhms(covQhwInvVio) )
    l_vio_inc_mod5.append( reso.calc_incoh_fwhms(covQhwInvVio) )


    print('---------------------------------------- vio_ext.py ----------------------------------------')
    # ---------------------------------------- vio_ext.py ----------------------------------------
    LPM, LMS, rad, z, Hdet = 8005.2e7, 1229.5e7, 4000.e7, 0, 3000e7
    thetaCP, thetaCM, wP, wM = 9.0, 3.25, 12e7, 6e7
    dDrad, dthetaf = 26e7, 0.0065
    # 1er modele:
    print('\n', 'PREMIER MODELE', '\n')
    dtp, dtm, dtd = np.divide(thetaCP, 2*6*v_rot), np.divide(thetaCM, 2*6*v_rot), 0
    dPrad, dMrad, dDz = 0, 0, np.divide(Hdet, 100)
    d_la = {"L_PM":LPM, "L_MS":LMS, "rad":rad, "z":z, "theta_i":0, "phi_i":0, "theta_f":theta_f, "phi_f":0}
    d_delta = {"dlt_Prad":dPrad, "dlt_Pz":0, "dlt_Mrad":dMrad, "dlt_Mz":0, "dlt_Srad":0, "dlt_Sz":0, "dlt_Drad":dDrad, "dlt_Dz":dDz, "dlt_tP":dtp, "dlt_tM":dtm, "dlt_tD":dtd, "dlt_theta_i":0, "dlt_theta_f":dthetaf}

    covQhwVioExt = vce.cov(v_i, v_f, d_la, d_delta, verbose=False)
    covQhwInvVioExt = la.inv(np.divide(covQhwVioExt, helpers.sig2fwhm))
    covQhwInvVioExt = np.dot(rot.T, np.dot(covQhwInvVioExt, rot))

    ellipses = reso.calc_ellipses(covQhwInvVioExt, verbose = True)
    if print(ellipses):
        reso.plot_ellipses(ellipses, verbose = True)
    l_vioext_coh_mod1.append( reso.calc_coh_fwhms(covQhwInvVioExt) )
    l_vioext_inc_mod1.append( reso.calc_incoh_fwhms(covQhwInvVioExt) )

    # 2e modele:
    print('\n', 'DEUXIEME MODELE', '\n')
    a = 0.761
    dtp, dtm, dtd = a*np.divide(thetaCP, 2*6*v_rot), a*np.divide(thetaCM, 2*6*v_rot), 0
    dPrad, dMrad, dDz = 0, 0, np.divide(Hdet, 100)
    d_la = {"L_PM":LPM, "L_MS":LMS, "rad":rad, "z":z, "theta_i":0, "phi_i":0, "theta_f":theta_f, "phi_f":0}
    d_delta = {"dlt_Prad":dPrad, "dlt_Pz":0, "dlt_Mrad":dMrad, "dlt_Mz":0, "dlt_Srad":0, "dlt_Sz":0, "dlt_Drad":dDrad, "dlt_Dz":dDz, "dlt_tP":dtp, "dlt_tM":dtm, "dlt_tD":dtd, "dlt_theta_i":0, "dlt_theta_f":dthetaf}

    covQhwVioExt = vce.cov(v_i, v_f, d_la, d_delta, verbose=False)
    covQhwInvVioExt = la.inv(np.divide(covQhwVioExt, helpers.sig2fwhm))
    covQhwInvVioExt = np.dot(rot.T, np.dot(covQhwInvVioExt, rot))

    ellipses = reso.calc_ellipses(covQhwInvVioExt, verbose = True)
    if print(ellipses):
        reso.plot_ellipses(ellipses, verbose = True)
    l_vioext_coh_mod2.append( reso.calc_coh_fwhms(covQhwInvVioExt) )
    l_vioext_inc_mod2.append( reso.calc_incoh_fwhms(covQhwInvVioExt) )

    # 3e modele:
    print('\n', 'TROISIEME MODELE', '\n')
    a = 0.761
    dtp, dtm, dtd = a*np.divide(thetaCP, 2*6*v_rot), a*np.divide(thetaCM, 2*6*v_rot), np.divide(dDrad, v_f)
    dPrad, dMrad, dDz = v_i*dtp - wP, v_i*dtm - wM, np.divide(Hdet, 100)
    d_la = {"L_PM":LPM, "L_MS":LMS, "rad":rad, "z":z, "theta_i":0, "phi_i":0, "theta_f":theta_f, "phi_f":0}
    d_delta = {"dlt_Prad":dPrad, "dlt_Pz":0, "dlt_Mrad":dMrad, "dlt_Mz":0, "dlt_Srad":0, "dlt_Sz":0, "dlt_Drad":dDrad, "dlt_Dz":dDz, "dlt_tP":dtp, "dlt_tM":dtm, "dlt_tD":dtd, "dlt_theta_i":0, "dlt_theta_f":dthetaf}

    covQhwVioExt = vce.cov(v_i, v_f, d_la, d_delta, verbose=False)
    covQhwInvVioExt = la.inv(np.divide(covQhwVioExt, helpers.sig2fwhm))
    covQhwInvVioExt = np.dot(rot.T, np.dot(covQhwInvVioExt, rot))

    ellipses = reso.calc_ellipses(covQhwInvVioExt, verbose = True)
    if print(ellipses):
        reso.plot_ellipses(ellipses, verbose = True)
    l_vioext_coh_mod3.append( reso.calc_coh_fwhms(covQhwInvVioExt) )
    l_vioext_inc_mod3.append( reso.calc_incoh_fwhms(covQhwInvVioExt) )

    # 4e modele:
    print('\n', 'QUATRIEME MODELE', '\n')
    a= np.divide(1, 2*np.sqrt(3))
    dtp, dtm, dtd = a*np.divide(thetaCP, 2*6*v_rot), a*np.divide(thetaCM, 2*6*v_rot), np.divide(dDrad, v_f)
    dPrad, dMrad, dDz = v_i*dtp - wP, v_i*dtm - wM, np.divide(Hdet, 100)
    d_la = {"L_PM":LPM, "L_MS":LMS, "rad":rad, "z":z, "theta_i":0, "phi_i":0, "theta_f":theta_f, "phi_f":0}
    d_delta = {"dlt_Prad":dPrad, "dlt_Pz":0, "dlt_Mrad":dMrad, "dlt_Mz":0, "dlt_Srad":0, "dlt_Sz":0, "dlt_Drad":dDrad, "dlt_Dz":dDz, "dlt_tP":dtp, "dlt_tM":dtm, "dlt_tD":dtd, "dlt_theta_i":0, "dlt_theta_f":dthetaf}

    covQhwVioExt = vce.cov(v_i, v_f, d_la, d_delta, verbose=False)
    covQhwInvVioExt = la.inv(covQhwVioExt)
    covQhwInvVioExt = np.dot(rot.T, np.dot(covQhwInvVioExt, rot))

    ellipses = reso.calc_ellipses(covQhwInvVioExt, verbose = True)
    if print(ellipses):
        reso.plot_ellipses(ellipses, verbose = True)
    l_vioext_coh_mod4.append( reso.calc_coh_fwhms(covQhwInvVioExt) )
    l_vioext_inc_mod4.append( reso.calc_incoh_fwhms(covQhwInvVioExt) )

    print('---------------------------------------- vio_ext2.py ----------------------------------------')
    # ---------------------------------------- vio_ext2.py ----------------------------------------
    thetaCP, thetaBP, wP = 9.0, 8.5, 12e7
    thetaCM, thetaBM, wM = 3.25, 3.0, 6e7
    Eyh, Ezh, Lpe, Lme, Les = 7e7, 27e7, 9064.7e7, 1114.3e7, 170e7
    Dr, Hdet, Wr = 4000e7, 3000e7, 26e7

    LSD, LSDz = Dr, 0
    LSDx, LSDy = LSD*np.cos(theta_f), LSD*np.sin(theta_f)
    Vartpref = np.divide( np.square(thetaCP) + np.square(thetaBP), np.square(6*2*v_rot) )
    Vartmref = np.divide( np.square(thetaCM) + np.square(thetaBM), np.square(6*2*v_rot) )
    VarDz = np.divide(1, 12) * np.square(np.divide(Hdet, 100))

    # POINT SAMPLE
    print('------ POINT SAMPLE ------')
    Sr, Sh = 1e3, 1e3
    VarDr = np.divide( np.square(2*Sr) + np.square(Wr), 12 )
    VarDtheta = np.square(0.0065)
    VarDx = np.square(np.cos(theta_f))*VarDr + np.square(Dr)*np.square(np.sin(theta_f))*VarDtheta
    VarDy = np.square(np.sin(theta_f))*VarDr + np.square(Dr)*np.square(np.cos(theta_f))*VarDtheta
    CovDxDy = np.cos(theta_f)*np.sin(theta_f) * ( VarDr - np.square(Dr)*VarDtheta )
    VarSx, VarSy, VarSz = 0, 0, 0
    Vartdref = np.divide( VarDr, np.square(v_f) )
    VarPyref = ( np.divide(np.square(Eyh), 3)
            + np.square(Lpe)*( 2*np.divide(np.square(Les), np.square(Sr))*(1 - np.sqrt(1 - np.divide(np.square(Sr), np.square(Les)))) - 1 )
            + np.divide(4*Lpe*Les, 3*np.square(Sr))*np.square(Eyh)*(1 - np.sqrt(1 - np.divide(np.square(Sr), np.square(Les))))
            + np.divide(2*np.square(Lpe), 3*np.square(Sr))*np.square(Eyh)*(np.divide(1, np.sqrt(1 - np.divide(np.square(Sr), np.square(Les)))) - 1) )
    VarPzref = ( np.divide(np.square(Ezh), 3)
            + np.divide(np.square(Lpe), 3*np.square(Sr))*(np.divide(np.square(Sh), 2) + 2*np.square(Ezh))*(np.divide(1, np.sqrt(1 - np.divide(np.square(Sr), np.square(Les)))) - 1) 
            + np.divide(4*Lpe*Les, 3*np.square(Sr))*np.square(Ezh)*(1 - np.sqrt(1 - np.divide(np.square(Sr), np.square(Les)))) )
    VarMyref = ( np.divide(np.square(Eyh), 3)
            + np.square(Lme)*( 2*np.divide(np.square(Les), np.square(Sr))*(1 - np.sqrt(1 - np.divide(np.square(Sr), np.square(Les)))) - 1 )
            + np.divide(4*Lme*Les, 3*np.square(Sr))*np.square(Eyh)*(1 - np.sqrt(1 - np.divide(np.square(Sr), np.square(Les))))
            + np.divide(2*np.square(Lme), 3*np.square(Sr))*np.square(Eyh)*(np.divide(1, np.sqrt(1 - np.divide(np.square(Sr), np.square(Les)))) - 1) )
    VarMzref = ( np.divide(np.square(Ezh), 3)
            + np.divide(np.square(Lme), 3*np.square(Sr))*(np.divide(np.square(Sh), 2) + 2*np.square(Ezh))*(np.divide(1, np.sqrt(1 - np.divide(np.square(Sr), np.square(Les)))) - 1) 
            + np.divide(4*Lme*Les, 3*np.square(Sr))*np.square(Ezh)*(1 - np.sqrt(1 - np.divide(np.square(Sr), np.square(Les)))) )

    # 1er modele :
    print('\n', 'PREMIER MODELE', '\n')
    a2 = np.square(0.761)
    Vartp, Vartm, Vartd = a2*Vartpref, a2*Vartmref, 0
    VarPx, VarPy, VarPz = 0, helpers.sig2fwhm * VarPyref, helpers.sig2fwhm * VarPzref
    VarMx, VarMy, VarMz = 0, helpers.sig2fwhm * VarMyref, helpers.sig2fwhm * VarMzref
    Var = [VarPx, VarPy, VarPz, VarMx, VarMy, VarMz, VarSx, VarSy, VarSz, VarDx, VarDy, VarDz, Vartp, Vartm, Vartd]
    covInstr = np.eye(15)
    for i in range(15):
        covInstr[i][i] = Var[i]
    covInstr[9][10], covInstr[10][9] = CovDxDy, CovDxDy

    sigPx = 1
    sigMx = 1
    LPM, LPMx, LPMy, LPMz, LMS, LMSx, LMSy, LMSz = vce2.length(Sr, Sh, Lpe, Lme, Les, Eyh, Ezh, -(Lpe+Les), sigPx, -(Lme+Les), sigMx)
    dict_la = {"L_PM":LPM, "L_PMx":LPMx, "L_PMy":LPMy, "L_PMz":LPMz, "L_MS":LMS, "L_MSx":LMSx, "L_MSy":LMSy, "L_MSz":LMSz, "L_SD":LSD, "L_SDx":LSDx, "L_SDy":LSDy, "L_SDz":LSDz}
    shape = 'VCYL'

    covQhwVioExt2 = vce2.cov(v_i, v_f, dict_la, covInstr, shape, False)
    covQhwInvVioExt2 = la.inv(np.divide(covQhwVioExt2, helpers.sig2fwhm))
    covQhwInvVioExt2 = np.dot(rot.T, np.dot(covQhwInvVioExt2, rot))

    ellipses = reso.calc_ellipses(covQhwInvVioExt2, verbose = True)
    if print(ellipses):
        reso.plot_ellipses(ellipses, verbose = True)
    l_vioext2_PS_coh_mod1.append( reso.calc_coh_fwhms(covQhwInvVioExt2) )
    l_vioext2_PS_inc_mod1.append( reso.calc_incoh_fwhms(covQhwInvVioExt2) )

    # 2e modele :
    print('\n', 'DEUXIEME MODELE', '\n')
    a2 = np.square(0.761)
    Vartp, Vartm, Vartd = a2*Vartpref, a2*Vartmref, Vartdref
    VarPx, VarPy, VarPz = np.square( v_i*np.sqrt(Vartp) - wP ), helpers.sig2fwhm * VarPyref, helpers.sig2fwhm * VarPzref
    VarMx, VarMy, VarMz = np.square( v_i*np.sqrt(Vartm) - wM ), helpers.sig2fwhm * VarMyref, helpers.sig2fwhm * VarMzref
    Var = [VarPx, VarPy, VarPz, VarMx, VarMy, VarMz, VarSx, VarSy, VarSz, VarDx, VarDy, VarDz, Vartp, Vartm, Vartd]
    covInstr = np.eye(15)
    for i in range(15):
        covInstr[i][i] = Var[i]
    covInstr[9][10], covInstr[10][9] = CovDxDy, CovDxDy

    sigPx = np.sqrt(VarPx)
    sigMx = np.sqrt(VarMx)
    LPM, LPMx, LPMy, LPMz, LMS, LMSx, LMSy, LMSz = vce2.length(Sr, Sh, Lpe, Lme, Les, Eyh, Ezh, -(Lpe+Les), sigPx, -(Lme+Les), sigMx)
    dict_la = {"L_PM":LPM, "L_PMx":LPMx, "L_PMy":LPMy, "L_PMz":LPMz, "L_MS":LMS, "L_MSx":LMSx, "L_MSy":LMSy, "L_MSz":LMSz, "L_SD":LSD, "L_SDx":LSDx, "L_SDy":LSDy, "L_SDz":LSDz}
    shape = 'VCYL'

    covQhwVioExt2 = vce2.cov(v_i, v_f, dict_la, covInstr, shape, False)
    covQhwInvVioExt2 = la.inv(np.divide(covQhwVioExt2, helpers.sig2fwhm))
    covQhwInvVioExt2 = np.dot(rot.T, np.dot(covQhwInvVioExt2, rot))

    ellipses = reso.calc_ellipses(covQhwInvVioExt2, verbose = True)
    if print(ellipses):
        reso.plot_ellipses(ellipses, verbose = True)
    l_vioext2_PS_coh_mod2.append( reso.calc_coh_fwhms(covQhwInvVioExt2) )
    l_vioext2_PS_inc_mod2.append( reso.calc_incoh_fwhms(covQhwInvVioExt2) )

    # 3e modele :
    a2 = np.divide(1, 12)
    Vartp, Vartm, Vartd = a2*Vartpref, a2*Vartmref, Vartdref
    VarPx, VarPy, VarPz = np.square( v_i*np.sqrt(Vartp) - wP ), VarPyref, VarPzref
    VarMx, VarMy, VarMz = np.square( v_i*np.sqrt(Vartm) - wM ), VarMyref, VarMzref
    Var = [VarPx, VarPy, VarPz, VarMx, VarMy, VarMz, VarSx, VarSy, VarSz, VarDx, VarDy, VarDz, Vartp, Vartm, Vartd]
    covInstr = np.eye(15)
    for i in range(15):
        covInstr[i][i] = Var[i]
    covInstr[9][10], covInstr[10][9] = CovDxDy, CovDxDy

    sigPx = np.sqrt(VarPx)
    sigMx = np.sqrt(VarMx)
    LPM, LPMx, LPMy, LPMz, LMS, LMSx, LMSy, LMSz = vce2.length(Sr, Sh, Lpe, Lme, Les, Eyh, Ezh, -(Lpe+Les), sigPx, -(Lme+Les), sigMx)
    dict_la = {"L_PM":LPM, "L_PMx":LPMx, "L_PMy":LPMy, "L_PMz":LPMz, "L_MS":LMS, "L_MSx":LMSx, "L_MSy":LMSy, "L_MSz":LMSz, "L_SD":LSD, "L_SDx":LSDx, "L_SDy":LSDy, "L_SDz":LSDz}
    shape = 'VCYL'

    covQhwVioExt2 = vce2.cov(v_i, v_f, dict_la, covInstr, shape, False)
    covQhwInvVioExt2 = la.inv(covQhwVioExt2)
    covQhwInvVioExt2 = np.dot(rot.T, np.dot(covQhwInvVioExt2, rot))

    ellipses = reso.calc_ellipses(covQhwInvVioExt2, verbose = True)
    if print(ellipses):
        reso.plot_ellipses(ellipses, verbose = True)
    l_vioext2_PS_coh_mod3.append( reso.calc_coh_fwhms(covQhwInvVioExt2) )
    l_vioext2_PS_inc_mod3.append( reso.calc_incoh_fwhms(covQhwInvVioExt2) )

    # VOLUME SAMPLE
    print('------ VOLUME SAMPLE ------')
    Sr, Sh = 6e7, 60e7
    VarDr = np.divide( np.square(2*Sr) + np.square(Wr), 12 )
    VarDtheta = np.square(0.0065)
    VarDx = np.square(np.cos(theta_f))*VarDr + np.square(Dr)*np.square(np.sin(theta_f))*VarDtheta
    VarDy = np.square(np.sin(theta_f))*VarDr + np.square(Dr)*np.square(np.cos(theta_f))*VarDtheta
    CovDxDy = np.cos(theta_f)*np.sin(theta_f) * ( VarDr - np.square(Dr)*VarDtheta )
    VarSxref, VarSyref, VarSzref = np.divide(np.square(Sr), 4), np.divide(np.square(Sr), 4), np.divide(np.square(Sh), 12)
    Vartdref = np.divide( VarDr, np.square(v_f) )
    VarPyref = ( np.divide(np.square(Eyh), 3)
            + np.square(Lpe)*( 2*np.divide(np.square(Les), np.square(Sr))*(1 - np.sqrt(1 - np.divide(np.square(Sr), np.square(Les)))) - 1 )
            + np.divide(4*Lpe*Les, 3*np.square(Sr))*np.square(Eyh)*(1 - np.sqrt(1 - np.divide(np.square(Sr), np.square(Les))))
            + np.divide(2*np.square(Lpe), 3*np.square(Sr))*np.square(Eyh)*(np.divide(1, np.sqrt(1 - np.divide(np.square(Sr), np.square(Les)))) - 1) )
    VarPzref = ( np.divide(np.square(Ezh), 3)
            + np.divide(np.square(Lpe), 3*np.square(Sr))*(np.divide(np.square(Sh), 2) + 2*np.square(Ezh))*(np.divide(1, np.sqrt(1 - np.divide(np.square(Sr), np.square(Les)))) - 1) 
            + np.divide(4*Lpe*Les, 3*np.square(Sr))*np.square(Ezh)*(1 - np.sqrt(1 - np.divide(np.square(Sr), np.square(Les)))) )
    VarMyref = ( np.divide(np.square(Eyh), 3)
            + np.square(Lme)*( 2*np.divide(np.square(Les), np.square(Sr))*(1 - np.sqrt(1 - np.divide(np.square(Sr), np.square(Les)))) - 1 )
            + np.divide(4*Lme*Les, 3*np.square(Sr))*np.square(Eyh)*(1 - np.sqrt(1 - np.divide(np.square(Sr), np.square(Les))))
            + np.divide(2*np.square(Lme), 3*np.square(Sr))*np.square(Eyh)*(np.divide(1, np.sqrt(1 - np.divide(np.square(Sr), np.square(Les)))) - 1) )
    VarMzref = ( np.divide(np.square(Ezh), 3)
            + np.divide(np.square(Lme), 3*np.square(Sr))*(np.divide(np.square(Sh), 2) + 2*np.square(Ezh))*(np.divide(1, np.sqrt(1 - np.divide(np.square(Sr), np.square(Les)))) - 1) 
            + np.divide(4*Lme*Les, 3*np.square(Sr))*np.square(Ezh)*(1 - np.sqrt(1 - np.divide(np.square(Sr), np.square(Les)))) )

    # 1er modele :
    print('\n', 'PREMIER MODELE', '\n')
    a2 = np.square(0.761)
    Vartp, Vartm, Vartd = a2*Vartpref, a2*Vartmref, 0
    VarPx, VarPy, VarPz = 0, helpers.sig2fwhm * VarPyref, helpers.sig2fwhm * VarPzref
    VarMx, VarMy, VarMz = 0, helpers.sig2fwhm * VarMyref, helpers.sig2fwhm * VarMzref
    VarSx, VarSy, VarSz = helpers.sig2fwhm * VarSxref, helpers.sig2fwhm * VarSyref, helpers.sig2fwhm * VarSzref
    Var = [VarPx, VarPy, VarPz, VarMx, VarMy, VarMz, VarSx, VarSy, VarSz, VarDx, VarDy, VarDz, Vartp, Vartm, Vartd]
    covInstr = np.eye(15)
    for i in range(15):
        covInstr[i][i] = Var[i]
    covInstr[9][10], covInstr[10][9] = CovDxDy, CovDxDy

    sigPx = 1
    sigMx = 1
    LPM, LPMx, LPMy, LPMz, LMS, LMSx, LMSy, LMSz = vce2.length(Sr, Sh, Lpe, Lme, Les, Eyh, Ezh, -(Lpe+Les), sigPx, -(Lme+Les), sigMx)
    dict_la = {"L_PM":LPM, "L_PMx":LPMx, "L_PMy":LPMy, "L_PMz":LPMz, "L_MS":LMS, "L_MSx":LMSx, "L_MSy":LMSy, "L_MSz":LMSz, "L_SD":LSD, "L_SDx":LSDx, "L_SDy":LSDy, "L_SDz":LSDz}
    shape = 'VCYL'

    covQhwVioExt2 = vce2.cov(v_i, v_f, dict_la, covInstr, shape, False)
    covQhwInvVioExt2 = la.inv(np.divide(covQhwVioExt2, helpers.sig2fwhm))
    covQhwInvVioExt2 = np.dot(rot.T, np.dot(covQhwInvVioExt2, rot))

    ellipses = reso.calc_ellipses(covQhwInvVioExt2, verbose = True)
    if print(ellipses):
        reso.plot_ellipses(ellipses, verbose = True)
    l_vioext2_VS_coh_mod1.append( reso.calc_coh_fwhms(covQhwInvVioExt2) )
    l_vioext2_VS_inc_mod1.append( reso.calc_incoh_fwhms(covQhwInvVioExt2) )

    # 2e modele :
    print('\n', 'DEUXIEME MODELE', '\n')
    a2 = np.square(0.761)
    Vartp, Vartm, Vartd = a2*Vartpref, a2*Vartmref, Vartdref
    VarPx, VarPy, VarPz = np.square( v_i*np.sqrt(Vartp) - wP ), helpers.sig2fwhm * VarPyref, helpers.sig2fwhm * VarPzref
    VarMx, VarMy, VarMz = np.square( v_i*np.sqrt(Vartm) - wM ), helpers.sig2fwhm * VarMyref, helpers.sig2fwhm * VarMzref
    VarSx, VarSy, VarSz = helpers.sig2fwhm * VarSxref, helpers.sig2fwhm * VarSyref, helpers.sig2fwhm * VarSzref
    Var = [VarPx, VarPy, VarPz, VarMx, VarMy, VarMz, VarSx, VarSy, VarSz, VarDx, VarDy, VarDz, Vartp, Vartm, Vartd]
    covInstr = np.eye(15)
    for i in range(15):
        covInstr[i][i] = Var[i]
    covInstr[9][10], covInstr[10][9] = CovDxDy, CovDxDy

    sigPx = np.sqrt(VarPx)
    sigMx = np.sqrt(VarMx)
    LPM, LPMx, LPMy, LPMz, LMS, LMSx, LMSy, LMSz = vce2.length(Sr, Sh, Lpe, Lme, Les, Eyh, Ezh, -(Lpe+Les), sigPx, -(Lme+Les), sigMx)
    dict_la = {"L_PM":LPM, "L_PMx":LPMx, "L_PMy":LPMy, "L_PMz":LPMz, "L_MS":LMS, "L_MSx":LMSx, "L_MSy":LMSy, "L_MSz":LMSz, "L_SD":LSD, "L_SDx":LSDx, "L_SDy":LSDy, "L_SDz":LSDz}
    shape = 'VCYL'

    covQhwVioExt2 = vce2.cov(v_i, v_f, dict_la, covInstr, shape, False)
    covQhwInvVioExt2 = la.inv(np.divide(covQhwVioExt2, helpers.sig2fwhm))
    covQhwInvVioExt2 = np.dot(rot.T, np.dot(covQhwInvVioExt2, rot))

    ellipses = reso.calc_ellipses(covQhwInvVioExt2, verbose = True)
    if print(ellipses):
        reso.plot_ellipses(ellipses, verbose = True)
    l_vioext2_VS_coh_mod2.append( reso.calc_coh_fwhms(covQhwInvVioExt2) )
    l_vioext2_VS_inc_mod2.append( reso.calc_incoh_fwhms(covQhwInvVioExt2) )

    # 3e modele :
    a2 = np.divide(1, 12)
    Vartp, Vartm, Vartd = a2*Vartpref, a2*Vartmref, Vartdref
    VarPx, VarPy, VarPz = np.square( v_i*np.sqrt(Vartp) - wP ), VarPyref, VarPzref
    VarMx, VarMy, VarMz = np.square( v_i*np.sqrt(Vartm) - wM ), VarMyref, VarMzref
    VarSx, VarSy, VarSz = VarSxref, VarSyref, VarSzref
    Var = [VarPx, VarPy, VarPz, VarMx, VarMy, VarMz, VarSx, VarSy, VarSz, VarDx, VarDy, VarDz, Vartp, Vartm, Vartd]
    covInstr = np.eye(15)
    for i in range(15):
        covInstr[i][i] = Var[i]
    covInstr[9][10], covInstr[10][9] = CovDxDy, CovDxDy

    sigPx = np.sqrt(VarPx)
    sigMx = np.sqrt(VarMx)
    LPM, LPMx, LPMy, LPMz, LMS, LMSx, LMSy, LMSz = vce2.length(Sr, Sh, Lpe, Lme, Les, Eyh, Ezh, -(Lpe+Les), sigPx, -(Lme+Les), sigMx)
    dict_la = {"L_PM":LPM, "L_PMx":LPMx, "L_PMy":LPMy, "L_PMz":LPMz, "L_MS":LMS, "L_MSx":LMSx, "L_MSy":LMSy, "L_MSz":LMSz, "L_SD":LSD, "L_SDx":LSDx, "L_SDy":LSDy, "L_SDz":LSDz}
    shape = 'VCYL'

    covQhwVioExt2 = vce2.cov(v_i, v_f, dict_la, covInstr, shape, False)
    covQhwInvVioExt2 = la.inv(covQhwVioExt2)
    covQhwInvVioExt2 = np.dot(rot.T, np.dot(covQhwInvVioExt2, rot))

    ellipses = reso.calc_ellipses(covQhwInvVioExt2, verbose = True)
    if print(ellipses):
        reso.plot_ellipses(ellipses, verbose = True)
    l_vioext2_VS_coh_mod3.append( reso.calc_coh_fwhms(covQhwInvVioExt2) )
    l_vioext2_VS_inc_mod3.append( reso.calc_incoh_fwhms(covQhwInvVioExt2) )

print('\n', 'VIO')
print('modele 1 :', '  inc =', l_vio_inc_mod1, '        coh =', l_vio_coh_mod1)
print('modele 2 :', '  inc =', l_vio_inc_mod2, '        coh =', l_vio_coh_mod2)
print('modele 3 :', '  inc =', l_vio_inc_mod3, '        coh =', l_vio_coh_mod3)
print('modele 4 :', '  inc =', l_vio_inc_mod4, '        coh =', l_vio_coh_mod4)
print('modele 5 :', '  inc =', l_vio_inc_mod5, '        coh =', l_vio_coh_mod5)

print('\n', 'VIOEXT')
print('modele 1 :', '  inc =', l_vioext_inc_mod1, '        coh =', l_vioext_coh_mod1)
print('modele 2 :', '  inc =', l_vioext_inc_mod2, '        coh =', l_vioext_coh_mod2)
print('modele 3 :', '  inc =', l_vioext_inc_mod3, '        coh =', l_vioext_coh_mod3)
print('modele 4 :', '  inc =', l_vioext_inc_mod4, '        coh =', l_vioext_coh_mod4)

print('\n', 'VIOEXT2 - POINT SAMPLE')
print('modele 1 :', '  inc =', l_vioext2_PS_inc_mod1, '        coh =', l_vioext2_PS_coh_mod1)
print('modele 2 :', '  inc =', l_vioext2_PS_inc_mod2, '        coh =', l_vioext2_PS_coh_mod2)
print('modele 3 :', '  inc =', l_vioext2_PS_inc_mod3, '        coh =', l_vioext2_PS_coh_mod3)

print('\n', 'VIOEXT2 - POINT SAMPLE')
print('modele 1 :', '  inc =', l_vioext2_VS_inc_mod1, '        coh =', l_vioext2_VS_coh_mod1)
print('modele 2 :', '  inc =', l_vioext2_VS_inc_mod2, '        coh =', l_vioext2_VS_coh_mod2)
print('modele 3 :', '  inc =', l_vioext2_VS_inc_mod3, '        coh =', l_vioext2_VS_coh_mod3)

def savemodel(nomf, lQ, model):
    dltQpara, dltQperp, dltQup, dltE = [], [], [], []
    nbQ = len(lQ)
    for i in range(nbQ):
        dltQpara.append(model[i][0])
        dltQperp.append(model[i][1])
        dltQup.append(model[i][2])
        dltE.append(model[i][3])
    with open(nomf, 'w') as fichier:
        fichier.write("Q (1/A)\tQpara (1/A)\tQperp (1/A)\tQup (1/A)\tE (meV)\n")
        for Q, Qpara, Qperp, Qup, E in zip(lQ, dltQpara, dltQperp, dltQup, dltE):
            fichier.write(f"{Q}\t{Qpara}\t{Qperp}\t{Qup}\t{E}\n")
        fichier.close()
    return 0

savemodel('test.txt', l_Q, l_vio_coh_mod5)