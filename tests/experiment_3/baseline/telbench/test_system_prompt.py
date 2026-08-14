from __future__ import annotations

from baseline.telbench.system_prompt import (
    all_at_once_parser,
    step_by_step_parser,
)


def test_all_at_once_parser_extracts_span_id():
    parsed = all_at_once_parser.parse('{"Span ID": "s004"}')
    assert parsed.span_id == "s004"


def test_step_by_step_parser_extracts_error_found():
    parsed = step_by_step_parser.parse('{"Error Found": true}')
    assert parsed.error_found is True
