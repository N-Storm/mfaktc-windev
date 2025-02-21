#!/bin/bash
#
# Wrapper script to call Microsoft Visual Studio compiler & tools
# from GNU Make. It's used by the Github Actions build workflow
# for Windows targets as a workaround.
# A more proper way would be to add MSVC environment initialiazion
# by the means of GNU Make itself or switch to other build system.
# But for now it does it's job.
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
#

# Default location (within Github runner) of the MSVC env setup script
MSVC_SHELL_DEFAULT='C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\Tools\Launch-VsDevShell.ps1'
# If it's not found on the path above, try to find it and save path to
# this file to avoid searching for it on every call.
MSVC_SHELL_LOC_CACHE='vcwrapper.cache'

if [ -f "$MSVC_SHELL_DEFAULT" ]; then
  MSVC_SHELL="'$MSVC_SHELL_DEFAULT'"
elif [ -f "$MSVC_SHELL_LOC_CACHE" ]; then
  MSVC_SHELL="$(cat $MSVC_SHELL_LOC_CACHE)"
else
  MSVC_SHELL="'$(find 'C:/Program Files/Microsoft Visual Studio' -type f -name 'Launch-VsDevShell.ps1' -print -quit)'"
  echo "$MSVC_SHELL" > "$MSVC_SHELL_LOC_CACHE"
fi

powershell "& $MSVC_SHELL -Arch amd64 -HostArch amd64; cd $GITHUB_WORKSPACE\\src; $*"
