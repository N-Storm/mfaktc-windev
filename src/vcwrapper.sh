#!/bin/bash

powershell "& 'C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\Tools\Launch-VsDevShell.ps1' -Arch amd64 -HostArch amd64; cd $GITHUB_WORKSPACE\\src; $*"
