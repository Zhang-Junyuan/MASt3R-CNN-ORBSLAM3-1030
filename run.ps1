param([switch]$Build,[switch]$Train,[int]$Frames=120,[int]$Stride=3,[ValidateSet('dynamic','baseline','both')][string]$Mode='dynamic')
$ErrorActionPreference='Stop'
$projectPath=$PSScriptRoot.Replace('\','/')
$linuxPath=(& wsl -d Ubuntu-22.04 -- wslpath -a $projectPath).Trim()
$py='/home/yuan/mast3r-slam-local/env/bin/python'
if($Build){
  & wsl -d Ubuntu-22.04 -- bash "$linuxPath/build.sh"
  if($LASTEXITCODE -ne 0){throw 'Native build failed'}
  & wsl -d Ubuntu-22.04 -- $py "$linuxPath/build_curope.py" build_ext --inplace
  if($LASTEXITCODE -ne 0){throw 'CUDA extension build failed'}
}
if($Train){& wsl -d Ubuntu-22.04 -- $py "$linuxPath/train_gate.py";if($LASTEXITCODE -ne 0){throw 'CNN training failed'}}
$modes=if($Mode -eq 'both'){@('baseline','dynamic')}else{@($Mode)}
foreach($m in $modes){
  & wsl -d Ubuntu-22.04 -- $py -u "$linuxPath/run_pipeline.py" --frames $Frames --stride $Stride --mode $m --cache --out "$linuxPath/results/visual_v2/$m"
  if($LASTEXITCODE -ne 0){throw "Pipeline failed: $m"}
}
& wsl -d Ubuntu-22.04 -- $py "$linuxPath/visual_report.py"
if($LASTEXITCODE -ne 0){throw 'Report failed'}
