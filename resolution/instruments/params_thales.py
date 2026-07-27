#
# thales parameters
#
# @author Tobias Weber <tweber@ill.fr>
# @date jul-2024
# @license see 'LICENSE' file
#
# ----------------------------------------------------------------------------
# Takin (inelastic neutron scattering software package)
# Copyright (C) 2017-2026  Tobias WEBER (Institut Laue-Langevin (ILL),
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

import libs.tas as tas
import libs.helpers as helpers

import numpy as np


#
# pre-defined parameters for thales
#
params = {
    # options
    "verbose" : True,

    # resolution method, "eck", "pop", or "cn"
    "reso_method" : "pop",

    # scattering triangle
    "ki" : 1.4,
    "kf" : 1.4,
    "E"  : tas.get_E(1.4, 1.4),
    "Q"  : 2.,

    # d spacings
    "mono_xtal_d" : 3.1355,   # SI
    "ana_xtal_d"  : 3.3550,   # PG

     # scattering senses
    "mono_sense"   : 1.,
    "sample_sense" : -1.,
    "ana_sense"    : 1.,
    "mirror_Qperp" : False,

    # distances
    "dist_vsrc_mono"   : 200. * helpers.cm2A,
    "dist_hsrc_mono"   : 200. * helpers.cm2A,
    "dist_mono_sample" : 200. * helpers.cm2A,
    "dist_sample_ana"  : 126.63 * helpers.cm2A,
    "dist_ana_det"     : 86.0  * helpers.cm2A,

    # shapes
    "src_shape"    : "rectangular",  # "rectangular" or "circular"
    "sample_shape" : "cylindrical",  # "cuboid" or "cylindrical"
    "det_shape"    : "rectangular",  # "rectangular" or "circular"

    # component sizes
    "src_w"    : 4.   * helpers.cm2A,
    "src_h"    : 12.  * helpers.cm2A,
    "mono_d"   : 0.2  * helpers.cm2A,
    "mono_w"   : 15. * helpers.cm2A,
    "mono_h"   : 19.5  * helpers.cm2A,
    "sample_d" : 1.   * helpers.cm2A,
    "sample_w" : 1.   * helpers.cm2A,
    "sample_h" : 3.   * helpers.cm2A,
    "ana_d"    : 0.2  * helpers.cm2A,
    "ana_w"    : 17.  * helpers.cm2A,
    "ana_h"    : 13.  * helpers.cm2A,
    "det_w"    : 5.  * helpers.cm2A,
    "det_h"    : 8.  * helpers.cm2A,

    # horizontal collimation
    "coll_h_pre_mono"    : 9999. * helpers.min2rad,
    "coll_h_pre_sample"  : 9999. * helpers.min2rad,
    "coll_h_post_sample" : 9999. * helpers.min2rad,
    "coll_h_post_ana"    : 9999. * helpers.min2rad,

    # vertical collimation
    "coll_v_pre_mono"    : 9999. * helpers.min2rad,
    "coll_v_pre_sample"  : 9999. * helpers.min2rad,
    "coll_v_post_sample" : 9999. * helpers.min2rad,
    "coll_v_post_ana"    : 9999. * helpers.min2rad,

    # horizontal focusing
    "mono_curv_h" : 0.,
    "ana_curv_h"  : 0.,
    "mono_is_curved_h" : True,
    "ana_is_curved_h"  : True,
    "mono_is_optimally_curved_h" : True,
    "ana_is_optimally_curved_h"  : True,
    "mono_curv_h_formula" : None,
    "ana_curv_h_formula" : None,

    # vertical focusing
    "mono_curv_v" : 0.,
    "ana_curv_v"  : 37. * helpers.cm2A,
    "mono_is_curved_v" : True,
    "ana_is_curved_v"  : True,
    "mono_is_optimally_curved_v" : True,
    "ana_is_optimally_curved_v"  : True,
    "mono_curv_v_formula" : None,
    "ana_curv_v_formula" : None,

    # guide before monochromator
    "use_guide"   : True,
    "guide_div_h" : 30. * helpers.min2rad,
    "guide_div_v" : 30. * helpers.min2rad,

    # horizontal mosaics
    "mono_mosaic"   : 1. * helpers.min2rad,
    "sample_mosaic" : 15. * helpers.min2rad,
    "ana_mosaic"    : 40. * helpers.min2rad,

    # vertical mosaics
    "mono_mosaic_v"   : 1. * helpers.min2rad,
    "sample_mosaic_v" : 15. * helpers.min2rad,
    "ana_mosaic_v"    : 40. * helpers.min2rad,

    # crystal reflectivities; TODO, so far always 1
    "dmono_refl" : 1.,
    "dana_effic" : 1.,

    # off-center scattering
    # WARNING: while this is calculated, it is not yet considered in the ellipse plots
    "pos_x" : 0. * helpers.cm2A,
    "pos_y" : 0. * helpers.cm2A,
    "pos_z" : 0. * helpers.cm2A,

    # vertical scattering in kf, keep "False" for normal TAS
    "kf_vert" : False,

    # sample integration method (for eck_ext): "gaussian" or "analytical"
    "sample_int" : "gaussian",
}
