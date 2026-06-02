param(
  [string]$Task = "demo"
)

switch ($Task) {
  "demo" { python .\scripts\run_baseline.py }
  "eval" { python .\scripts\run_eval.py }
  default { Write-Host "Unknown task: $Task" }
}
