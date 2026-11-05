$ErrorActionPreference = "Stop"

$RepoUrl = "https://github.com/chocolate-doom/chocolate-doom.git"
$Tag = "chocolate-doom-3.1.1"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ScriptDir
$RefDir = Join-Path $RootDir "reference"

if (Test-Path (Join-Path $RefDir ".git")) {
    Write-Host "Reference repository already exists at $RefDir. Verifying tag $Tag..."
    Push-Location $RefDir
    try {
        $CurrentTag = git describe --tags --exact-match 2>$null
        if ($CurrentTag -ne $Tag) {
            Write-Host "Fetching and checking out $Tag..."
            git fetch --depth 1 origin tag $Tag
            git checkout $Tag
        } else {
            Write-Host "Already at tag $Tag."
        }
    } finally {
        Pop-Location
    }
} else {
    if (Test-Path $RefDir) {
        Remove-Item -Recurse -Force $RefDir
    }
    Write-Host "Cloning $RepoUrl at tag $Tag into $RefDir..."
    git clone --depth 1 --branch $Tag -c advice.detachedHead=false $RepoUrl $RefDir
}

$ExpectedFile = Join-Path $RefDir "src\hexen\p_map.c"
if (-not (Test-Path $ExpectedFile)) {
    throw "Expected reference file not found: $ExpectedFile"
}
Write-Host "Reference source ready at $RefDir ($Tag)."
