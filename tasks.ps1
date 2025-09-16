Param(
  [Parameter(Mandatory=$false)][string]$Task = "run"
)

switch ($Task) {
  "run" { python app.py }
  "lint" { ruff check . }
  "format" { ruff format . }
  default { Write-Host "Usage: .\tasks.ps1 -Task [run|lint|format]" }
}
