param([switch]$Build,[switch]$Train,[switch]$Verify,[int]$Frames=120,[int]$Stride=3,[ValidateSet('dynamic','baseline','both')][string]$Mode='dynamic')
$ErrorActionPreference='Stop'
$projectPath=$PSScriptRoot.Replace('\','/')
$linuxPath=(& wsl -d Ubuntu-22.04 -- wslpath -a $projectPath).Trim()
$py='/home/yuan/mast3r-slam-local/env/bin/python'
function CheckExit([string]$message){if($LASTEXITCODE -ne 0){throw $message}}
if($Verify -and ($Frames -ne 120 -or $Stride -ne 3 -or $Mode -eq 'baseline')){throw 'Controlled real-feature verification requires dynamic output with Frames=120 and Stride=3'}
if($Build){
 & wsl -d Ubuntu-22.04 -- bash "$linuxPath/build.sh"
 CheckExit 'Native build failed'
 & wsl -d Ubuntu-22.04 -- $py "$linuxPath/build_curope.py" build_ext --inplace
 CheckExit 'CUDA extension failed'
}
if($Train){& wsl -d Ubuntu-22.04 -- $py "$linuxPath/train_gate.py";CheckExit 'CNN training failed'}
$modes=if($Mode -eq 'both'){@('baseline','dynamic')}else{@($Mode)}
foreach($m in $modes){
 & wsl -d Ubuntu-22.04 -- $py -u "$linuxPath/run_pipeline.py" --frames $Frames --stride $Stride --mode $m --cache --out "$linuxPath/results/visual_v2/$m"
 CheckExit "Visual pipeline failed: $m"
}
if($Verify){
 foreach($script in @('tests.py','test_visual_chain.py','replay_relocalization.py','replay_loop.py')){
  & wsl -d Ubuntu-22.04 -- $py -u "$linuxPath/$script"
  CheckExit "Verification failed: $script"
 }
}
& wsl -d Ubuntu-22.04 -- $py "$linuxPath/visual_report.py"
CheckExit 'Visual report failed'
