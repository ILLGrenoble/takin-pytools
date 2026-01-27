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
from scipy.stats import rv_continuous

# Conversion
meV2J = 1.602176634e-22
m2A = 1e10
#Constant
m_n = 1.67492749804e-27
h = 6.62607015e-34
hbar = np.divide(h, 2*np.pi)

#Initialisation of the covariance Matrix
covQhw = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])

class trapeze_distribution(rv_continuous):
    def __init__(self, x0, x1, x2, x3):
        super().__init__(a=x0, b=x3)
        self.x0, self.x1, self.x2, self.x3, self.J = x0, x1, x2, x3, np.divide(2, x3 + x2 - x1 - x0)

    def _pdf(self, x):
        if x < self.x0 or x >= self.x3:
            return 0
        elif self.x0 <= x < self.x1:
            return np.divide(self.J * (x - self.x0), (self.x1 - self.x0))
        elif self.x1 <= x < self.x2:
            return self.J
        elif self.x2 <= x < self.x3:
            return np.divide(self.J*(self.x3 - x), (self.x3 - self.x2))
        
    def _cdf(self, x):
        if x < self.x0:
            return 0
        elif self.x0 <= x < self.x1:
            return np.divide(self.J*np.square(x - self.x0), 2*(self.x1 - self.x0))
        elif self.x1 <= x < self.x2:
            return np.divide(self.J*(2*x - self.x0 - self.x1), 2)
        elif self.x2 <= x <= self.x3:
            return 1 - np.divide(self.J*np.square(self.x3 - x), 2*(self.x3 - self.x2))
        elif self.x3 < x:
            return 1
        
    def _ppf(self, q):
        if q < 0 or q > 1:
            return np.nan
        elif 0 <= q <= np.divide(self.J*(self.x1 - self.x0), 2):
            return self.x0 + np.sqrt(np.divide(2*q*(self.x1 - self.x0), self.J))
        elif np.divide(self.J*(self.x1 - self.x0), 2) < q <= 1 - np.divide(self.J*(self.x3 - self.x2), 2):
            return np.divide(q, self.J) + np.divide(self.x0 + self.x1, 2)
        elif 1 - np.divide(self.J*(self.x3 - self.x2), 2) < q <= 1:
            return self.x3 - np.sqrt(np.divide(2*(1 - q)*(self.x3 - self.x2), self.J))

class triangle_distribution(rv_continuous):
    def __init__(self, x0, x3):
        super().__init__(a=x0, b=x3)
        self.x0, self.x3, self.xm, self.J = x0, x3, np.divide(x0 + x3, 2), np.divide(2, x3 - x0)

    def _pdf(self, x):
        if x < self.x0 or x > self.x3:
            return 0
        elif self.x0 <= x < self.xm:
            return np.divide(self.J * (x - self.x0), (self.xm - self.x0))
        elif self.xm <= x <= self.x3:
            return np.divide(self.J * (x - self.x3), (self.xm - self.x3))
        
    def _cdf(self, x):
        if x < self.x0:
            return 0
        elif self.x0 <= x < self.xm:
            return np.divide(self.J*np.square(x - self.x0), 2*(self.xm - self.x0))
        elif self.xm <= x <= self.x3:
            return 1 - np.divide(self.J*np.square(self.x3 - x), 2*(self.x3 - self.xm))
        elif self.x3 < x:
            return 1
        
    def _ppf(self, q):
        if q < 0 or q > 1:
            return np.nan
        elif 0 <= q <= np.divide(self.J*(self.xm - self.x0), 2):
            return self.x0 + np.sqrt(np.divide(2*q*(self.xm - self.x0), self.J))
        elif 1 - np.divide(self.J*(self.x3 - self.xm), 2) < q <= 1:
            return self.x3 - np.sqrt(np.divide(2*(1 - q)*(self.x3 - self.xm), self.J))

def getx_trapeze(moy, var, slope):
    x0 = moy - np.sqrt(np.divide(1 + 6*slope*var, 2*slope))
    x1 = moy - np.sqrt(np.divide(np.square(moy)*slope - 2*moy*slope*x0 + slope*np.square(x0) - 1, slope))
    x2, x3 = 2*moy - x1, 2*moy - x0
    return (x0, x1, x2, x3)

def getx_triangle(moy, var):
    x0, x3 = moy - np.sqrt(6*var), moy + np.sqrt(6*var)
    return (x0, x3)

def k2v(k:float):
    return np.divide(k*np.square(m2A)*hbar, m_n)

def length(radS:float, heiS:float, L_PE:float, L_ME:float, L_ES:float, wEy:float, wEz:float, moyPx:float, varPx:float, slopePx:float, distrib_type_Px:str, moyMx:float, varMx:float, slopeMx:float, distrib_type_Mx:str):
    LPS, LPSx, LPSy, LPSz = 0, 0, 0, 0
    LMS, LMSx, LMSy, LMSz = 0, 0, 0, 0
    nb_pts = 100000
    Px, Mx = moyPx, moyMx
    if distrib_type_Px == 'Trapeze':
        x0P, x1P, x2P, x3P = getx_trapeze(moyPx, varPx, slopePx)
        distrib_Px = trapeze_distribution(x0P, x1P, x2P, x3P)
    elif distrib_type_Px == 'Triangle':
        x0P, x3P = getx_triangle(moyPx, varPx)
        distrib_Px = triangle_distribution(x0P, x3P)
    if distrib_type_Mx == 'Trapeze':
        x0M, x1M, x2M, x3M = getx_trapeze(moyMx, varMx, slopeMx)
        distrib_Mx = trapeze_distribution(x0M, x1M, x2M, x3M)
    elif distrib_type_Mx == 'Triangle':
        x0M, x3M = getx_triangle(moyMx, varMx)
        distrib_Mx = triangle_distribution(x0M, x3M)
    for i in range(nb_pts):
        Sr, theta, Sz = radS*np.sqrt(np.random.uniform(0,1)), 2*np.pi*np.random.uniform(0,1), np.random.uniform(-heiS/2, heiS/2)
        Sx, Sy = Sr*np.cos(theta), Sr*np.sin(theta)
        Pymin, Pymax = Sy + np.divide(L_PE + L_ES + Sx, L_ES + Sx)*(-wEy - Sy), Sy + np.divide(L_PE + L_ES + Sx, L_ES + Sx)*(wEy - Sy)
        Pzmin, Pzmax = Sz + np.divide(L_PE + L_ES + Sx, L_ES + Sx)*(-wEz - Sz), Sz + np.divide(L_PE + L_ES + Sx, L_ES + Sx)*(wEz - Sz)
        Mymin, Mymax = Sy + np.divide(L_ME + L_ES + Sx, L_ES + Sx)*(-wEy - Sy), Sy + np.divide(L_ME + L_ES + Sx, L_ES + Sx)*(wEy - Sy)
        Mzmin, Mzmax = Sz + np.divide(L_ME + L_ES + Sx, L_ES + Sx)*(-wEz - Sz), Sz + np.divide(L_ME + L_ES + Sx, L_ES + Sx)*(wEz - Sz)
        Px, Py, Pz = distrib_Px.rvs(), np.random.uniform(Pymin, Pymax), np.random.uniform(Pzmin, Pzmax)
        Mx, My, Mz = distrib_Mx.rvs(), np.random.uniform(Mymin, Mymax), np.random.uniform(Mzmin, Mzmax)
        LPS += np.sqrt( np.square(Sx-Px) + np.square(Sy-Py) + np.square(Sz-Pz) )
        LPSx += Sx - Px
        LPSy += Sy - Py
        LPSz += Sz - Pz
        LMS += np.sqrt( np.square(Sx-Mx) + np.square(Sy-My) + np.square(Sz-Mz) )
        LMSx += Sx - Mx
        LMSy += Sy - My
        LMSz += Sz - Mz
    LPS /= nb_pts
    LPSx /= nb_pts
    LPSy /= nb_pts
    LPSz /= nb_pts
    LMS /= nb_pts
    LMSx /= nb_pts
    LMSy /= nb_pts
    LMSz /= nb_pts
    LPM, LPMx, LPMy, LPMz = LPS - LMS, LPSx - LMSx, LPSy - LMSy, LPSz - LMSz 
    return(LPM, LPMx, LPMy, LPMz, LMS, LMSx, LMSy, LMSz)


def covInstrument(VarPx:float, VarPy:float, VarPz:float, VarMx:float, VarMy:float, VarMz:float, VarSx:float, VarSy:float, VarSz:float,
             VarDx:float, VarDy:float, VarDz:float, Vartp:float, Vartm:float, Vartd:float, CovDxDy:float) -> np.array:
    covI = np.eye(15)
    Var = [VarPx, VarPy, VarPz, VarMx, VarMy, VarMz, VarSx, VarSy, VarSz, VarDx, VarDy, VarDz, Vartp, Vartm, Vartd]
    for i in range(15):
        covI[i][i] = Var[i]
    covI[9][10], covI[10][9] = CovDxDy, CovDxDy
    return covI


# Calcul of Jacobian's terms for every shape
def jacobTerms(v_i:float, v_f:float, dict_L:dict, shape, verbose=False) -> np.array:
    L_PM, L_PMx, L_PMy, L_PMz = dict_L["L_PM"], dict_L["L_PMx"], dict_L["L_PMy"], dict_L["L_PMz"]
    L_MS, L_MSx, L_MSy, L_MSz = dict_L["L_MS"], dict_L["L_MSx"], dict_L["L_MSy"], dict_L["L_MSz"]
    L_SD, L_SDx, L_SDy, L_SDz = dict_L["L_SD"], dict_L["L_SDx"], dict_L["L_SDy"], dict_L["L_SDz"]
    #Energie
    dEdPx = -np.divide(m_n, L_PM) * ( np.square(v_i) + np.divide(np.power(v_f, 3), v_i)*np.divide(L_MS, L_SD) ) * np.divide(L_PMx, L_PM)   *np.divide(1, np.square(m2A)*meV2J)
    dEdPy = -np.divide(m_n, L_PM) * ( np.square(v_i) + np.divide(np.power(v_f, 3), v_i)*np.divide(L_MS, L_SD) ) * np.divide(L_PMy, L_PM)   *np.divide(1, np.square(m2A)*meV2J)
    dEdPz = -np.divide(m_n, L_PM) * ( np.square(v_i) + np.divide(np.power(v_f, 3), v_i)*np.divide(L_MS, L_SD) ) * np.divide(L_PMz, L_PM)   *np.divide(1, np.square(m2A)*meV2J)
    dEdMx = np.divide(m_n, L_PM) * ( ( np.square(v_i) + np.divide(np.power(v_f, 3), v_i)*np.divide(L_MS, L_SD) )*np.divide(L_PMx, L_PM)
                                    + np.divide(np.power(v_f, 3), v_i)*np.divide(L_PM, L_SD)*np.divide(L_MSx, L_MS) )   *np.divide(1, np.square(m2A)*meV2J)
    dEdMy = np.divide(m_n, L_PM) * ( ( np.square(v_i) + np.divide(np.power(v_f, 3), v_i)*np.divide(L_MS, L_SD) )*np.divide(L_PMy, L_PM)
                                    + np.divide(np.power(v_f, 3), v_i)*np.divide(L_PM, L_SD)*np.divide(L_MSy, L_MS) )   *np.divide(1, np.square(m2A)*meV2J)
    dEdMz = np.divide(m_n, L_PM) * ( ( np.square(v_i) + np.divide(np.power(v_f, 3), v_i)*np.divide(L_MS, L_SD) )*np.divide(L_PMz, L_PM)
                                    + np.divide(np.power(v_f, 3), v_i)*np.divide(L_PM, L_SD)*np.divide(L_MSz, L_MS) )   *np.divide(1, np.square(m2A)*meV2J)
    dEdSx = np.divide(m_n, L_SD) * ( np.square(v_f)*np.divide(L_SDx, L_SD) - np.divide(np.power(v_f, 3), v_i)*np.divide(L_MSx, L_MS) )   *np.divide(1, np.square(m2A)*meV2J)
    dEdSy = np.divide(m_n, L_SD) * ( np.square(v_f)*np.divide(L_SDy, L_SD) - np.divide(np.power(v_f, 3), v_i)*np.divide(L_MSy, L_MS) )   *np.divide(1, np.square(m2A)*meV2J)
    dEdSz = np.divide(m_n, L_SD) * ( np.square(v_f)*np.divide(L_SDz, L_SD) - np.divide(np.power(v_f, 3), v_i)*np.divide(L_MSz, L_MS) )   *np.divide(1, np.square(m2A)*meV2J)
    dEdDx = -np.divide(m_n, L_SD) * np.square(v_f)*np.divide(L_SDx, L_SD)   *np.divide(1, np.square(m2A)*meV2J)
    dEdDy = -np.divide(m_n, L_SD) * np.square(v_f)*np.divide(L_SDy, L_SD)   *np.divide(1, np.square(m2A)*meV2J)
    dEdDz = -np.divide(m_n, L_SD) * np.square(v_f)*np.divide(L_SDz, L_SD)   *np.divide(1, np.square(m2A)*meV2J)
    dEdtp = np.divide(m_n, L_PM) * ( np.power(v_i,3) + np.power(v_f, 3)*np.divide(L_MS, L_SD) )   *np.divide(1, np.square(m2A)*meV2J)
    dEdtm = -np.divide(m_n, L_PM) * ( np.power(v_i,3) + np.power(v_f, 3)*( np.divide(L_PM, L_SD) + np.divide(L_MS, L_SD) ) )   *np.divide(1, np.square(m2A)*meV2J)
    dEdtd = np.divide(m_n, L_SD) * np.power(v_f, 3)   *np.divide(1, np.square(m2A)*meV2J)
    dE = np.array([dEdPx, dEdPy, dEdPz, dEdMx, dEdMy, dEdMz, dEdSx, dEdSy, dEdSz, dEdDx, dEdDy, dEdDz, dEdtp, dEdtm, dEdtd])

    #Qx
    dQxdPx = -np.divide(m_n, hbar*L_PM) * ( v_i + np.divide(np.square(v_f), v_i)*np.divide(L_MS, L_SD)*np.divide(L_SDx, L_SD)*np.divide(L_PMx, L_PM) )   *np.divide(1, np.square(m2A))
    dQxdPy = -np.divide(m_n, hbar*L_PM) * ( np.divide(np.square(v_f), v_i)*np.divide(L_MS, L_SD)*np.divide(L_SDx, L_SD)*np.divide(L_PMy, L_PM) )   *np.divide(1, np.square(m2A))
    dQxdPz = -np.divide(m_n, hbar*L_PM) * ( np.divide(np.square(v_f), v_i)*np.divide(L_MS, L_SD)*np.divide(L_SDx, L_SD)*np.divide(L_PMz, L_PM) )   *np.divide(1, np.square(m2A))
    dQxdMx = np.divide(m_n, hbar*L_PM) * ( v_i + np.divide(np.square(v_f), v_i)*( np.divide(L_PM, L_SD)*np.divide(L_MSx, L_MS) + np.divide(L_MS, L_SD)*np.divide(L_PMx, L_PM) )*np.divide(L_SDx, L_SD) )   *np.divide(1, np.square(m2A))
    dQxdMy = np.divide(m_n, hbar*L_PM) * ( np.divide(np.square(v_f), v_i)*( np.divide(L_PM, L_SD)*np.divide(L_MSy, L_MS) + np.divide(L_MS, L_SD)*np.divide(L_PMy, L_PM) )*np.divide(L_SDx, L_SD) )   *np.divide(1, np.square(m2A))
    dQxdMz = np.divide(m_n, hbar*L_PM) * ( np.divide(np.square(v_f), v_i)*( np.divide(L_PM, L_SD)*np.divide(L_MSz, L_MS) + np.divide(L_MS, L_SD)*np.divide(L_PMz, L_PM) )*np.divide(L_SDx, L_SD) )   *np.divide(1, np.square(m2A))
    dQxdSx = np.divide(m_n, hbar*L_SD) * ( v_f - np.divide(np.square(v_f), v_i)*np.divide(L_SDx, L_SD)*np.divide(L_MSx, L_MS) )   *np.divide(1, np.square(m2A))
    dQxdSy = -np.divide(m_n, hbar*L_SD) * ( np.divide(np.square(v_f), v_i)*np.divide(L_SDx, L_SD)*np.divide(L_MSy, L_MS) )   *np.divide(1, np.square(m2A))
    dQxdSz = -np.divide(m_n, hbar*L_SD) * ( np.divide(np.square(v_f), v_i)*np.divide(L_SDx, L_SD)*np.divide(L_MSz, L_MS) )   *np.divide(1, np.square(m2A))
    dQxdDx = -np.divide(m_n, hbar*L_SD) * v_f   *np.divide(1, np.square(m2A))
    dQxdDy = 0
    dQxdDz = 0
    dQxdtp = np.divide(m_n, hbar*L_PM) * ( np.square(v_i)*np.divide(L_PMx, L_PM) + np.square(v_f)*np.divide(L_MS, L_SD)*np.divide(L_SDx, L_SD) )   *np.divide(1, np.square(m2A))
    dQxdtm = -np.divide(m_n, hbar*L_PM) * ( np.square(v_i)*np.divide(L_PMx, L_PM) + np.square(v_f)*( np.divide(L_PM, L_SD) + np.divide(L_MS, L_SD) )*np.divide(L_SDx, L_SD) )   *np.divide(1, np.square(m2A))
    dQxdtd = np.divide(m_n, hbar*L_SD) * np.square(v_f)*np.divide(L_SDx, L_SD)   *np.divide(1, np.square(m2A))
    dQx = np.array([dQxdPx, dQxdPy, dQxdPz, dQxdMx, dQxdMy, dQxdMz, dQxdSx, dQxdSy, dQxdSz, dQxdDx, dQxdDy, dQxdDz, dQxdtp, dQxdtm, dQxdtd])

    #Qy
    dQydPx = -np.divide(m_n, hbar*L_PM) * ( np.divide(np.square(v_f), v_i)*np.divide(L_MS, L_SD)*np.divide(L_SDy, L_SD)*np.divide(L_PMx, L_PM) )   *np.divide(1, np.square(m2A))
    dQydPy = -np.divide(m_n, hbar*L_PM) * ( v_i + np.divide(np.square(v_f), v_i)*np.divide(L_MS, L_SD)*np.divide(L_SDy, L_SD)*np.divide(L_PMy, L_PM) )   *np.divide(1, np.square(m2A))
    dQydPz = -np.divide(m_n, hbar*L_PM) * ( np.divide(np.square(v_f), v_i)*np.divide(L_MS, L_SD)*np.divide(L_SDy, L_SD)*np.divide(L_PMz, L_PM) )   *np.divide(1, np.square(m2A))
    dQydMx = np.divide(m_n, hbar*L_PM) * ( np.divide(np.square(v_f), v_i)*( np.divide(L_PM, L_SD)*np.divide(L_MSx, L_MS) + np.divide(L_MS, L_SD)*np.divide(L_PMx, L_PM) )*np.divide(L_SDy, L_SD) )   *np.divide(1, np.square(m2A))
    dQydMy = np.divide(m_n, hbar*L_PM) * ( v_i + np.divide(np.square(v_f), v_i)*( np.divide(L_PM, L_SD)*np.divide(L_MSy, L_MS) + np.divide(L_MS, L_SD)*np.divide(L_PMy, L_PM) )*np.divide(L_SDy, L_SD) )   *np.divide(1, np.square(m2A))
    dQydMz = np.divide(m_n, hbar*L_PM) * ( np.divide(np.square(v_f), v_i)*( np.divide(L_PM, L_SD)*np.divide(L_MSz, L_MS) + np.divide(L_MS, L_SD)*np.divide(L_PMz, L_PM) )*np.divide(L_SDy, L_SD) )   *np.divide(1, np.square(m2A))
    dQydSx = -np.divide(m_n, hbar*L_SD) * ( np.divide(np.square(v_f), v_i)*np.divide(L_SDy, L_SD)*np.divide(L_MSx, L_MS) )   *np.divide(1, np.square(m2A))
    dQydSy = np.divide(m_n, hbar*L_SD) * ( v_f - np.divide(np.square(v_f), v_i)*np.divide(L_SDy, L_SD)*np.divide(L_MSy, L_MS) )   *np.divide(1, np.square(m2A))
    dQydSz = -np.divide(m_n, hbar*L_SD) * ( np.divide(np.square(v_f), v_i)*np.divide(L_SDy, L_SD)*np.divide(L_MSz, L_MS) )   *np.divide(1, np.square(m2A))
    dQydDx = 0
    dQydDy = -np.divide(m_n, hbar*L_SD) * v_f   *np.divide(1, np.square(m2A))
    dQydDz = 0
    dQydtp = np.divide(m_n, hbar*L_PM) * ( np.square(v_i)*np.divide(L_PMy, L_PM) + np.square(v_f)*np.divide(L_MS, L_SD)*np.divide(L_SDy, L_SD) )   *np.divide(1, np.square(m2A))
    dQydtm = -np.divide(m_n, hbar*L_PM) * ( np.square(v_i)*np.divide(L_PMy, L_PM) + np.square(v_f)*( np.divide(L_PM, L_SD) + np.divide(L_MS, L_SD) )*np.divide(L_SDy, L_SD) )   *np.divide(1, np.square(m2A))
    dQydtd = np.divide(m_n, hbar*L_SD) * np.square(v_f)*np.divide(L_SDy, L_SD)   *np.divide(1, np.square(m2A))
    dQy = np.array([dQydPx, dQydPy, dQydPz, dQydMx, dQydMy, dQydMz, dQydSx, dQydSy, dQydSz, dQydDx, dQydDy, dQydDz, dQydtp, dQydtm, dQydtd])

    #Qz
    dQzdPx = -np.divide(m_n, hbar*L_PM) * ( np.divide(np.square(v_f), v_i)*np.divide(L_MS, L_SD)*np.divide(L_SDz, L_SD)*np.divide(L_PMx, L_PM) )   *np.divide(1, np.square(m2A))
    dQzdPy = -np.divide(m_n, hbar*L_PM) * ( np.divide(np.square(v_f), v_i)*np.divide(L_MS, L_SD)*np.divide(L_SDz, L_SD)*np.divide(L_PMy, L_PM) )   *np.divide(1, np.square(m2A))
    dQzdPz = -np.divide(m_n, hbar*L_PM) * ( v_i + np.divide(np.square(v_f), v_i)*np.divide(L_MS, L_SD)*np.divide(L_SDz, L_SD)*np.divide(L_PMz, L_PM) )   *np.divide(1, np.square(m2A))
    dQzdMx = np.divide(m_n, hbar*L_PM) * ( np.divide(np.square(v_f), v_i)*( np.divide(L_PM, L_SD)*np.divide(L_MSx, L_MS) + np.divide(L_MS, L_SD)*np.divide(L_PMx, L_PM) )*np.divide(L_SDz, L_SD) )   *np.divide(1, np.square(m2A))
    dQzdMy = np.divide(m_n, hbar*L_PM) * ( np.divide(np.square(v_f), v_i)*( np.divide(L_PM, L_SD)*np.divide(L_MSy, L_MS) + np.divide(L_MS, L_SD)*np.divide(L_PMy, L_PM) )*np.divide(L_SDz, L_SD) )   *np.divide(1, np.square(m2A))
    dQzdMz = np.divide(m_n, hbar*L_PM) * ( v_i + np.divide(np.square(v_f), v_i)*( np.divide(L_PM, L_SD)*np.divide(L_MSz, L_MS) + np.divide(L_MS, L_SD)*np.divide(L_PMz, L_PM) )*np.divide(L_SDz, L_SD) )   *np.divide(1, np.square(m2A))
    dQzdSx = -np.divide(m_n, hbar*L_SD) * ( np.divide(np.square(v_f), v_i)*np.divide(L_SDz, L_SD)*np.divide(L_MSx, L_MS) )   *np.divide(1, np.square(m2A))
    dQzdSy = -np.divide(m_n, hbar*L_SD) * ( np.divide(np.square(v_f), v_i)*np.divide(L_SDz, L_SD)*np.divide(L_MSy, L_MS) )   *np.divide(1, np.square(m2A))
    dQzdSz = np.divide(m_n, hbar*L_SD) * ( v_f - np.divide(np.square(v_f), v_i)*np.divide(L_SDz, L_SD)*np.divide(L_MSz, L_MS) )   *np.divide(1, np.square(m2A))
    dQzdDx = 0
    dQzdDy = 0
    dQzdDz = -np.divide(m_n, hbar*L_SD) * v_f   *np.divide(1, np.square(m2A))
    dQzdtp = np.divide(m_n, hbar*L_PM) * ( np.square(v_i)*np.divide(L_PMz, L_PM) + np.square(v_f)*np.divide(L_MS, L_SD)*np.divide(L_SDz, L_SD) )   *np.divide(1, np.square(m2A))
    dQzdtm = -np.divide(m_n, hbar*L_PM) * ( np.square(v_i)*np.divide(L_PMz, L_PM) + np.square(v_f)*( np.divide(L_PM, L_SD) + np.divide(L_MS, L_SD) )*np.divide(L_SDz, L_SD) )   *np.divide(1, np.square(m2A))
    dQzdtd = np.divide(m_n, hbar*L_SD) * np.square(v_f)*np.divide(L_SDz, L_SD)   *np.divide(1, np.square(m2A))
    dQz = np.array([dQzdPx, dQzdPy, dQzdPz, dQzdMx, dQzdMy, dQzdMz, dQzdSx, dQzdSy, dQzdSz, dQzdDx, dQzdDy, dQzdDz, dQzdtp, dQzdtm, dQzdtd])
    return(dQx, dQy, dQz, dE)

# Only for shape = VCYL
def cov(v_i:float, v_f:float, dict_L:dict, cov_instr:np.array, shape='VCYL', verbose=False):
    dQx, dQy, dQz, dE = jacobTerms(v_i, v_f, dict_L, shape, verbose)
    jacobian = np.array([dQx, dQy, dQz, dE])
    covQhw = np.dot(jacobian, np.dot(cov_instr, jacobian.T))
    ###########################################################################################
    if(verbose):
        print('dict_L_A =', dict_L)
        print('v_i =', v_i, '; v_f =', v_f)
        print('det_shape =', shape, '\n')
        print('L_PM =', dict_L["L_PM"], '; L_MS =', dict_L["L_MS"], '; L_SD =', dict_L["L_SD"])
        print('dQx =', dQx, '\ndQy =', dQy, '\ndQz =', dQz, '\ndE =', dE, '\n')
        print('Jacobian =', jacobian, '\n')
        print('covxi = ', cov_instr, '\n')
        print('covQhw =', covQhw, '\n')

    return covQhw


#a, b, c, d = 0, 2, 6, 8
#ma_distribution = trapeze_distribution(a, b, c, d)
#print(ma_distribution.rvs())

#moy = 0
#nb = 10000
#mini, maxi = (a+d)/2, (a+d)/2
#for i in range(nb):
#    add = ma_distribution.rvs()
#    if add < mini:
#        mini = add
#    if add > maxi:
#        maxi = add
#    moy += add
#print(moy/nb, mini, maxi)

#distrib = triangle_distribution(a,d)
#print(distrib.rvs())
#moy = 0
#mini, maxi = (a+d)/2, (a+d)/2
#for i in range(nb):
#    add = distrib.rvs()
#    if add < mini:
#        mini = add
#    if add > maxi:
#        maxi = add
#    moy += add
#print(moy/nb, mini, maxi)