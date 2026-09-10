$ErrorActionPreference = "Stop"
$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $root
try {
    cmd /c "`"$vcvars`" >nul && cl /nologo /O2 /LD rollout_planner.cpp /Fe:rollout_planner.dll"
    if ($LASTEXITCODE -ne 0) {
        throw "cl failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
