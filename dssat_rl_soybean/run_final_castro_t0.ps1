# Final Castro run: five PPO seeds, seed summary and paired comparison on the frozen 2021-2024 block.
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = "python"
$cfg = "dssat_rl_soybean/configs/experiment_castro_t0.yaml"
$log = "dssat_rl_soybean/outputs/ppo_castro_t0_pipeline.log"

& $py -u -m dssat_rl_soybean.src.run_multiseed --config $cfg --run-prefix ppo_castro_t0 --generated-dir _generated_ppo_castro_t0 --seeds 42,43,44,45,46 --log-file "dssat_rl_soybean/outputs/ppo_castro_t0_run.log" *>> $log
if ($LASTEXITCODE -ne 0) { throw "multi-seed run failed" }
& $py -u -m dssat_rl_soybean.src.summarize_multiseed --run-prefix ppo_castro_t0 *>> $log
if ($LASTEXITCODE -ne 0) { throw "seed summary failed" }
& $py -u -m dssat_rl_soybean.src.final_comparators --config $cfg --target test --ppo-evaluation dssat_rl_soybean/outputs/ppo_castro_t0_summary/policy_evaluation_all_seeds.csv *>> $log
if ($LASTEXITCODE -ne 0) { throw "test comparators failed" }
"PIPELINE FINISHED $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 $log
