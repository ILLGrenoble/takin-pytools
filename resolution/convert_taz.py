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
import libs.helpers as helpers

import numpy as np
np.set_printoptions(floatmode = "fixed",  precision = 4)


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
sample_sense = float(taz.find("./reso/sample_scatter_sense").text)
sample_h = float(taz.find("./reso/pop_sample_h").text)
sample_wq = float(taz.find("./reso/pop_sample_wq").text)
sample_wperpq = float(taz.find("./reso/pop_sample_wperpq").text)
sample_geo = int(taz.find("./reso/pop_sample_cuboid").text)

plane_x0 = float(taz.find("./plane/x0").text)
plane_x1 = float(taz.find("./plane/x1").text)
plane_x2 = float(taz.find("./plane/x2").text)
plane_y0 = float(taz.find("./plane/y0").text)
plane_y1 = float(taz.find("./plane/y1").text)
plane_y2 = float(taz.find("./plane/y2").text)

src_w = float(taz.find("./reso/pop_src_w").text)
src_h = float(taz.find("./reso/pop_src_h").text)
src_geo = int(taz.find("./reso/pop_source_rect").text)

mono_d = float(taz.find("./reso/mono_d").text)
mono_w = float(taz.find("./reso/pop_mono_w").text)
mono_h = float(taz.find("./reso/pop_mono_h").text)
mono_t = float(taz.find("./reso/pop_mono_thick").text)
mono_mosaic = float(taz.find("./reso/mono_mosaic").text)
mono_sense = float(taz.find("./reso/mono_scatter_sense").text)

ana_d = float(taz.find("./reso/ana_d").text)
ana_w = float(taz.find("./reso/pop_ana_w").text)
ana_h = float(taz.find("./reso/pop_ana_h").text)
ana_t = float(taz.find("./reso/pop_ana_thick").text)
ana_mosaic = float(taz.find("./reso/ana_mosaic").text)
ana_sense = float(taz.find("./reso/ana_scatter_sense").text)

det_w = float(taz.find("./reso/pop_det_w").text)
det_h = float(taz.find("./reso/pop_det_h").text)
det_geo = int(taz.find("./reso/pop_det_rect").text)

dist_src_mono = float(taz.find("./reso/pop_dist_src_mono").text)
dist_mono_sample = float(taz.find("./reso/pop_dist_mono_sample").text)
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
E = float(taz.find("./reso/E").text)
Q = float(taz.find("./reso/Q").text)
# -----------------------------------------------------------------------------


# TODO
