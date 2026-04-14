param(
  [string]$Task = "demo"
)

switch ($Task) {
  "demo" { python .\scripts\demo_query.py }
  "eval" { python .\scripts\run_eval.py }
  default { Write-Host "Unknown task: $Task" }
}
