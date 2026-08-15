#!/usr/bin/env python3
#
# simple fourier trafo tests
# @author Tobias Weber <tweber@ill.fr>
# @date 15-aug-2026
# @license see 'LICENSE' file
#

import numpy as np
import scipy as sp
import scipy.constants as co
import matplotlib
import matplotlib.pyplot as plt


# calculate constants
hbar_in_meVps = co.Planck/co.elementary_charge*1e15/2./np.pi


# fourier trafo of a gaussian
# see: https://mathworld.wolfram.com/FourierTransformGaussian.html
def gauss_ft(E0, sigma, amp, t):
    return amp * sigma * np.sqrt(2.*np.pi) \
        * np.exp(-0.5 * (t*sigma/hbar_in_meVps)**2.) \
        * np.exp(-1j * t*E0/hbar_in_meVps)



ts = np.logspace(0.1, 5., 128)
Cs_0 = np.real(gauss_ft(0., 1e-4, 100., ts))
Cs_1 = np.real(gauss_ft(1e-3, 1e-3, 10., ts))

Cs_total = Cs_0 + Cs_1
Cs_total /= Cs_total[0]

plt.xlabel("t (ps)")
plt.ylabel("C")

plt.semilogx(ts, Cs_total)
plt.show()
