#!/usr/bin/env python3
#
# convert takin taz files
#
# @author Tobias Weber <tweber@ill.fr>
# @date 30-aug-2026
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

import sys
sys.path.append(".")

import libs.tas as tas
#import libs.helpers as helpers

#import numpy as np
#np.set_printoptions(floatmode = "fixed",  precision = 4)


# -----------------------------------------------------------------------------
# get command-line arguments
# -----------------------------------------------------------------------------
import argparse
argparser = argparse.ArgumentParser(
    description = "Converts Takin taz files.")

argparser.add_argument("in_file", type = str, help = "input taz file")
argparser.add_argument("-o", "--out_file", default = "", type = str, help = "output file")

parsedargs = argparser.parse_args()

# get parsed command-line arguments
in_file = parsedargs.in_file
out_file = parsedargs.out_file
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# load takin file
# -----------------------------------------------------------------------------
import xml.etree.ElementTree as xml
taz = xml.parse(in_file).getroot()

sample_a = float(taz.find("./sample/a").text)
sample_b = float(taz.find("./sample/b").text)
sample_c = float(taz.find("./sample/c").text)
sample_alpha = float(taz.find("./sample/alpha").text)
sample_beta = float(taz.find("./sample/beta").text)
sample_gamma = float(taz.find("./sample/gamma").text)
sample_mosaic = float(taz.find("./reso/sample_mosaic").text)
sample_h = float(taz.find("./reso/pop_sample_h").text)
sample_wq = float(taz.find("./reso/pop_sample_wq").text)
sample_wperpq = float(taz.find("./reso/pop_sample_wperpq").text)
sample_geo = int(taz.find("./reso/pop_sample_cuboid").text)
if sample_geo > 0:
    sample_geo_factor_w = 12.
else:
    sample_geo_factor_w = 16.
sample_geo_factor_h = 12.
sample_sense = int(taz.find("./reso/sample_scatter_sense").text)
if sample_sense == 1:
    sample_sense = 1.
else:
    sample_sense = -1.

plane_x0 = float(taz.find("./plane/x0").text)
plane_x1 = float(taz.find("./plane/x1").text)
plane_x2 = float(taz.find("./plane/x2").text)
plane_y0 = float(taz.find("./plane/y0").text)
plane_y1 = float(taz.find("./plane/y1").text)
plane_y2 = float(taz.find("./plane/y2").text)

src_w = float(taz.find("./reso/pop_src_w").text)
src_h = float(taz.find("./reso/pop_src_h").text)
src_geo = int(taz.find("./reso/pop_source_rect").text)
if src_geo > 0:
    src_geo_factor = 12.
else:
    src_geo_factor = 16.

mono_d = float(taz.find("./reso/mono_d").text)
mono_w = float(taz.find("./reso/pop_mono_w").text)
mono_h = float(taz.find("./reso/pop_mono_h").text)
mono_t = float(taz.find("./reso/pop_mono_thick").text)
mono_mosaic = float(taz.find("./reso/mono_mosaic").text)
mono_geo_factor = 12.
mono_sense = int(taz.find("./reso/mono_scatter_sense").text)
if mono_sense == 1:
    mono_sense = 1.
else:
    mono_sense = -1.

ana_d = float(taz.find("./reso/ana_d").text)
ana_w = float(taz.find("./reso/pop_ana_w").text)
ana_h = float(taz.find("./reso/pop_ana_h").text)
ana_t = float(taz.find("./reso/pop_ana_thick").text)
ana_mosaic = float(taz.find("./reso/ana_mosaic").text)
ana_geo_factor = 12.
ana_sense = float(taz.find("./reso/ana_scatter_sense").text)
if ana_sense == 1:
    ana_sense = 1.
else:
    ana_sense = -1.

det_w = float(taz.find("./reso/pop_det_w").text)
det_h = float(taz.find("./reso/pop_det_h").text)
det_geo = int(taz.find("./reso/pop_det_rect").text)
if det_geo > 0:
    det_geo_factor = 12.
else:
    det_geo_factor = 16.

monitor_w = float(taz.find("./reso/pop_monitor_w").text)
monitor_h = float(taz.find("./reso/pop_monitor_h").text)
monitor_geo = int(taz.find("./reso/pop_monitor_rect").text)
if monitor_geo > 0:
    monitor_geo_factor = 12.
else:
    monitor_geo_factor = 16.

dist_src_mono = float(taz.find("./reso/pop_dist_src_mono").text)
dist_mono_sample = float(taz.find("./reso/pop_dist_mono_sample").text)
dist_mono_monitor = float(taz.find("./reso/pop_dist_mono_monitor").text)
dist_sample_ana = float(taz.find("./reso/pop_dist_sample_ana").text)
dist_ana_det = float(taz.find("./reso/pop_dist_ana_det").text)

coll_h_mono = float(taz.find("./reso/h_coll_mono").text)
coll_h_sample1 = float(taz.find("./reso/h_coll_before_sample").text)
coll_h_sample2 = float(taz.find("./reso/h_coll_after_sample").text)
coll_h_ana = float(taz.find("./reso/h_coll_ana").text)
coll_v_mono = float(taz.find("./reso/v_coll_mono").text)
coll_v_sample1 = float(taz.find("./reso/v_coll_before_sample").text)
coll_v_sample2 = float(taz.find("./reso/v_coll_after_sample").text)
coll_v_ana = float(taz.find("./reso/v_coll_ana").text)

ki = float(taz.find("./reso/ki").text)
kf = float(taz.find("./reso/kf").text)
Ef = tas.k2_to_E * kf**2.
E = float(taz.find("./reso/E").text)
Q = float(taz.find("./reso/Q").text)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# output reslib script
# -----------------------------------------------------------------------------
reslib = """using Pkg
Pkg.activate(".")

using ResLib
using LinearAlgebra

exp = Experiment(
    setup = Popovici(
        monochromator = Monochromator(tau = 2*pi/{mono_d}, mosaic = {mono_mosaic}),
        mono_width = {mono_w}/sqrt({mono_geo_factor}),
        mono_height = {mono_h}/sqrt({mono_geo_factor}),
        mono_depth = {mono_t}/sqrt({mono_geo_factor}),
        mono_rh = 9999, mono_rv = 9999,

        analyzer = Analyzer(tau = 2*pi/{ana_d}, mosaic = {ana_mosaic}),
        ana_width = {ana_w}/sqrt({ana_geo_factor}),
        ana_height = {ana_h}/sqrt({ana_geo_factor}),
        ana_depth = {ana_t}/sqrt({ana_geo_factor}),
        ana_rh = 9999, ana_rv = 9999,

        beam = (;
            width = {src_w}/sqrt({src_geo_factor}),
            height = {src_h}/sqrt({src_geo_factor})),

        detector = (;
            width = {det_w}/sqrt({det_geo_factor}),
            height = {det_h}/sqrt({det_geo_factor})),

        sample_shape = Diagonal([
            {sample_wq}^2/{sample_geo_factor_w},
            {sample_wperpq}^2/{sample_geo_factor_w},
            {sample_h}^2/{sample_geo_factor_h} ]),

        hcol = ({coll_h_mono}, {coll_h_sample1}, {coll_h_sample2}, {coll_h_ana}),
        vcol = ({coll_v_mono}, {coll_v_sample1}, {coll_v_sample2}, {coll_v_ana}),

        arms = ({dist_src_mono}, {dist_mono_sample}, {dist_sample_ana}, {dist_ana_det}),

        monitor = true,
        monitor_geometry = (;
            width = {monitor_w}/sqrt({monitor_geo_factor}),
            height = {monitor_h}/sqrt({monitor_geo_factor}),
            distance = {dist_mono_monitor}),
    ),

    scattering_directions = ({mono_sense}, {sample_sense}, {ana_sense}),
    efixed = ({E_f}, :final),

    sample = Sample(
        lattice = Lattice(
            a = {sample_a}, b = {sample_b}, c = {sample_c},
            alpha = {sample_alpha}, beta = {sample_beta}, gamma = {sample_gamma}),
        mosaic = {sample_mosaic},
    ),

    orient1 = ({plane_x0}, {plane_x1}, {plane_x2}),
    orient2 = ({plane_y0}, {plane_y1}, {plane_x2}),
)

results = ResLib.resolution_matrix(exp, {Q}, {E})
display(results[1])
display(results[2])
"""


print(reslib.format(
    sample_a = sample_a, sample_b = sample_b, sample_c = sample_c,
    sample_alpha = sample_alpha, sample_beta = sample_beta, sample_gamma = sample_gamma,
    sample_mosaic = sample_mosaic, sample_sense = sample_sense,
    sample_wq = sample_wq, sample_wperpq = sample_wperpq, sample_h = sample_h,
    sample_geo_factor_w = sample_geo_factor_w,
    sample_geo_factor_h = sample_geo_factor_h,

    plane_x0 = plane_x0, plane_x1 = plane_x1, plane_x2 = plane_x2,
    plane_y0 = plane_y0, plane_y1 = plane_y1, plane_y2 = plane_y2,

    src_w = src_w, src_h = src_h, src_geo_factor = src_geo_factor,
    det_w = det_w, det_h = det_h, det_geo_factor = det_geo_factor,

    mono_w = mono_w, mono_h = mono_h, mono_t = mono_t,
    mono_d = mono_d, mono_mosaic = mono_mosaic, mono_sense = mono_sense,
    mono_geo_factor = mono_geo_factor,

    monitor_w = monitor_w, monitor_h = monitor_h,
    monitor_geo_factor = monitor_geo_factor,

    ana_w = ana_w, ana_h = ana_h, ana_t = ana_t,
    ana_d = ana_d, ana_mosaic = ana_mosaic, ana_sense = ana_sense,
    ana_geo_factor = ana_geo_factor,

    coll_h_mono = coll_h_mono, coll_h_sample1 = coll_h_sample1,
    coll_h_sample2 = coll_h_sample2, coll_h_ana = coll_h_ana,
    coll_v_mono = coll_v_mono, coll_v_sample1 = coll_v_sample1,
    coll_v_sample2 = coll_v_sample2, coll_v_ana = coll_v_ana,

    dist_src_mono = dist_src_mono,
    dist_mono_sample = dist_mono_sample,
    dist_mono_monitor = dist_mono_monitor,
    dist_sample_ana = dist_sample_ana,
    dist_ana_det = dist_ana_det,

    Q = Q, E = E, E_f = Ef,
))
# -----------------------------------------------------------------------------
