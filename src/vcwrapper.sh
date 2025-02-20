#!/bin/bash

# powershell & "C:\Progra~1\Micros~2\2022\Enterp~1\Common7\Tools\Launch-VsDevShell.ps1 -Arch amd64 -HostArch amd64 & $*"
powershell "& 'C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\Tools\Launch-VsDevShell.ps1' -Arch amd64 -HostArch amd64; $*"
# powershell "& 'D:\Tools\Dev\msys64\tmp\path with space\test.ps1' -Arch amd64 -HostArch amd64; $*"
