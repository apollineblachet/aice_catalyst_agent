from __future__ import annotations
from typing import Tuple

def recompute_phase_estimate(phase) -> float:
    total = round(sum(t.estimate_days for t in phase.tasks), 2)
    phase.estimate_days = total
    return total

def recompute_plan_estimate(plan) -> Tuple[float, list[tuple[str, float]]]:
    phase_breakdown = []
    for ph in plan.phases:
        ph_total = recompute_phase_estimate(ph)
        phase_breakdown.append((ph.name, ph_total))
    plan.total_estimate_days = round(sum(v for _, v in phase_breakdown), 2)
    return plan.total_estimate_days, phase_breakdown
