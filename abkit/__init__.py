"""
abkit — Production-grade A/B testing framework
================================================
pip install abkit

Quickstart
----------
    from abkit import Experiment, ExperimentConfig
    import pandas as pd

    config = ExperimentConfig(
        name="My Experiment",
        metric="revenue",
        covariate="pre_revenue",
        guardrail_metrics=["latency_ms"],
        alpha=0.05,
    )

    exp = Experiment(config)
    results = exp.run(pd.read_csv("data/experiment.csv"))
    exp.print_summary(results)
"""

from abkit.experiment import Experiment, ExperimentConfig, ExperimentResults
from abkit.compare import ExperimentComparison
from abkit.logger import configure_logging, get_logger

__version__ = "0.2.0"
__author__  = "Sudiksha Acharya"
__email__   = "sudiksha.acharyaa7@gmail.com"
__license__ = "MIT"

__all__ = [
    "Experiment",
    "ExperimentConfig",
    "ExperimentResults",
    "ExperimentComparison",
    "configure_logging",
    "get_logger",
]
