from __future__ import annotations


def first_error_span(span_ids_in_order: list[str], gold_spans: list[str]) -> str:
    gold_set = set(gold_spans)
    for span_id in span_ids_in_order:
        if span_id in gold_set:
            return span_id
    raise ValueError("No gold span found in span_ids_in_order")


def compute_metrics(
    pred_span: str | None, gold_spans: list[str], gt_first_error: str
) -> dict:
    hit = pred_span is not None and pred_span in gold_spans
    fea = float(pred_span == gt_first_error)

    if not hit:
        return {"fea": fea, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = 1.0
    recall = 1.0 / len(gold_spans)
    f1 = 2 * precision * recall / (precision + recall)
    return {"fea": fea, "precision": precision, "recall": recall, "f1": f1}
