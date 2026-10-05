"""Compatibility imports for the modular intent-routing model and requests."""

from abcd_benchmark.intent.agents import make_agent
from abcd_benchmark.intent.requests import dialogue_prefix
from abcd_benchmark.intent.schema import Intent, IntentPrediction

__all__ = ['Intent', 'IntentPrediction', 'dialogue_prefix', 'make_agent']
