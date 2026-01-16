# Repository Cleanup Plan

**Date**: 2026-01-15
**Purpose**: Clean up before implementing Fix #3

---

## Current State

### Models (3.5 MB)
- `scripts/RL_training_&_testing/solar_batt_agent_weekly_lagged.zip` (3.5 MB)
  - This is Fix #2 model (worst performing: €623/year)
  - Should be DELETED

### TensorBoard Logs (264 KB)
- `scripts/RL_training_&_testing/solar_batt_tensorboard/` (264 KB)
  - Training logs from previous runs
  - Should be DELETED (will regenerate with Fix #3)

### Output CSVs (4.3 MB total)
- `outputs/agent_step_data.csv` (2.3 MB) - Original agent (€591/year)
  - KEEP - needed for baseline comparison
- `outputs/agent_step_data_new.csv` (2.0 MB) - Fix #2 agent (€623/year)
  - DELETE - worst performing, no longer needed

### Diagnostic CSVs
- `outputs/diagnostics/daily_summary.csv`
- `outputs/diagnostics/full_analysis.csv`
- `outputs/diagnostics/hourly_summary.csv`
  - KEEP - useful diagnostic data

### Visualization PNGs
- `outputs/cumulative_rewards_comparison.png`
- `outputs/diagnostics/agent_diagnostic.png`
- `outputs/diagnostics/why_agent_fails_analysis.png`
  - KEEP - useful for documentation

---

## Cleanup Actions

### Files to DELETE (5.5 MB total):
1. `scripts/RL_training_&_testing/solar_batt_agent_weekly_lagged.zip` (3.5 MB)
2. `scripts/RL_training_&_testing/solar_batt_tensorboard/` (264 KB)
3. `outputs/agent_step_data_new.csv` (2.0 MB)

### Files to KEEP:
1. `outputs/agent_step_data.csv` - Original baseline
2. `outputs/diagnostics/*.csv` - Diagnostic data
3. `outputs/*.png` - Visualizations
4. All analysis scripts
5. All documentation

---

## Post-Cleanup State

After cleanup:
- Clean slate for Fix #3 training
- Original baseline preserved for comparison
- All diagnostic data preserved
- Documentation intact

Ready for Fix #3 implementation!
