"""Pure, leakage-safe intent-routing request preparation."""

from pathlib import Path

from abcd_benchmark.common.data import DATA_REVISION

from .config import EvaluationConfig


def dialogue_prefix(conversation: dict, customer_turns: int) -> list[dict[str, str]]:
    """Whitelist observed speech; stop immediately after N customer messages.

    Zero means all speech (a retrospective diagnostic, not online routing).
    Never include scenarios, targets, candidates, or action text.
    """
    dialogue = []
    customers = 0
    for turn in conversation['delexed']:
        if turn['speaker'] not in {'customer', 'agent'}:
            continue
        dialogue.append({'speaker': turn['speaker'], 'text': turn['text']})
        if turn['speaker'] == 'customer':
            customers += 1
            if customer_turns and customers >= customer_turns:
                break
    if not customers:
        raise ValueError(f"Conversation {conversation['convo_id']} has no customer messages")
    return dialogue


def prepare_row(conversation: dict, config: EvaluationConfig, dataset: Path,
                intent_to_flow: dict[str, str]) -> dict:
    gold = conversation['delexed'][0]['targets'][0]
    if gold not in intent_to_flow:
        raise ValueError(f'Unknown gold intent: {gold}')
    return {
        'convo_id': conversation['convo_id'], 'gold': gold,
        'scenario_subflow': conversation['scenario']['subflow'],
        'state': {'dialogue': dialogue_prefix(conversation, config.customer_turns)},
        'data_revision': DATA_REVISION, 'dataset': str(dataset), 'split': config.split,
        'seed': config.seed, 'customer_turns': config.customer_turns,
        'requested_model': config.model, 'dry_run': config.dry_run,
    }
