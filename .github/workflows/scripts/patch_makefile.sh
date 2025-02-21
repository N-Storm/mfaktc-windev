#!/bin/bash
#
# Script to patch Makefile.win to fix windowss builds with
# Github runners. This adds a lot of confusion with quotes
# but for now this seems to be required. Invoking PowerShell
# or cmd.exe as a SHELL fails with ACCESS_VIOLATION on nvcc.
# Running cmd.exe from wrapper encounters a lot of issues
# on paths with spaces and quotes again. So we're going
# to run PowerShell from a wrapper script.
# This script modifies Makefile.win to make it work with this
# wrapper.
# TODO: Find a better and less confusing way to make it work.
#
# SPDX-License-Identifier: GPL-3.0-or-later
# 
# This file is part of mfaktc.
# Copyright (C) 2009, 2010, 2011  Oliver Weihe (o.weihe@t-online.de)
# 
# mfaktc is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# mfaktc is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#                                 
# You should have received a copy of the GNU General Public License
# along with mfaktc.  If not, see <http://www.gnu.org/licenses/>.

echo "Removing CC arch entries from the Makefile.win and adding all, which current nvcc claims to support..."
sed -i '/^NVCCFLAGS += --generate-code arch=compute.*/d' src/Makefile.win
nvcc --list-gpu-arch | grep -Eoe 'compute_[0-9]+' | cut -d '_' -f2 | /usr/bin/sort -un | xargs -i bash -c "sed -i '/^CUFLAGS = -DWIN64.*$/a NVCCFLAGS += --generate-code arch=compute_{},code=sm_{}' src/Makefile.win"

echo "Patching Makefile.win to support build..."
sed -i -e 's/^CC = cl/CC = .\/vcwrapper.sh cl/' -e 's/^LINK = link/LINK = .\/vcwrapper.sh link/' -E -e 's/(.*?)nvcc (.*)/\1.\/vcwrapper.sh \"nvcc \2\"/' src/Makefile.win
sed -i -e 's/^CUDA_DIR = .*/CUDA_DIR = ${CUDA_PATH}/' -e 's/\/I$(CUDA_DIR)\\include/\\\"\/I"$(CUDA_DIR)\\include"\\\"/' -e 's/\/I$(CUDA_DIR)\\include\\cudart/\\\"\/I"$(CUDA_DIR)\\include\\cudart"\\\"/' src/Makefile.win
sed -i -e 's/\/EHsc \/W3 \/nologo \/O/\/EHsc,\/W3,\/nologo,\/Ox/' src/Makefile.win
sed -i -e 's/$(LIBS) \/out:$@/\\\"\"$(LIBS)\"\\\" \"\/out:$@\"/' src/Makefile.win
