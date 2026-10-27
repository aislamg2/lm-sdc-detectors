"""Breakdowns for RQ3 (fault site, bit group, layer) and RQ4 ablations."""

import pandas as pd

STRATA = ["fault_site", "bit_group", "layer"]

ABLATIONS = {
    "reasoning": ("lm_generative", "lm_single_pass"),
    "decision_head": ("lm_probe", "learned"),
    "access": ("blackbox", "whitebox"),
    "context": ("zero_shot", "reference"),
}


def stratified_metrics(scored_runs: pd.DataFrame, by: str) -> pd.DataFrame:
    """AUROC and recall@threshold (with CIs) per detector per stratum. TODO: implement."""
    raise NotImplementedError


def ablation_table(scored_runs: pd.DataFrame, ablation: str) -> pd.DataFrame:
    """Side-by-side comparison for one entry of ABLATIONS. TODO: implement."""
    raise NotImplementedError
