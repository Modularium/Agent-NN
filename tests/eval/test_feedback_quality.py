import json
from pathlib import Path

def test_feedback_quality():
    data = [json.loads(line) for line in Path('evaluation/examples/feedback_examples.jsonl').read_text().splitlines()]
    ratings = {}
    for entry in data:
        ratings.setdefault(entry['agent'], []).append(entry['rating'])
    avg_a = sum(ratings['agent_a']) / len(ratings['agent_a'])
    avg_b = sum(ratings['agent_b']) / len(ratings['agent_b'])
    assert avg_b >= avg_a
