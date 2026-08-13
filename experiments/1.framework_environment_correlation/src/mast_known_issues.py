"""Known data-quality issues in ``data/error_categorization/mast/``.

Shared by the analysis/plot scripts in this package so the exclusion list
lives in exactly one place.
"""

# AppWorld/HyperAgent/OpenManus: mast_annotation is index-aligned duplicated
# across the 3 frameworks (same 14-code pattern at every matching
# trace_index) despite raw_trajectory being 100% distinct (SHA1 hash) and
# each framework running on a different benchmark + model. Confirmed data
# pipeline bug, not real framework behavior — see
# papers/notes/findings/mast_data_finding.md. Excluded from every
# mas_name-level analysis below.
DUPLICATE_ANNOTATION_BUG = frozenset({"AppWorld", "HyperAgent", "OpenManus"})
