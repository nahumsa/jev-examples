"""Pydantic AI agent construction, isolated from dataset preparation."""

from pydantic_ai import Agent

from .schema import IntentPrediction


def make_agent(model: str = 'typesafe:jev-latest') -> Agent:
    return Agent(model, output_type=IntentPrediction)
