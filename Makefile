PYTHON ?= python
WW_DIR := data/error_localization/single_fault

.PHONY: load_whowhen load_whowhen_algo load_whowhen_hand

load_whowhen: load_whowhen_algo load_whowhen_hand

load_whowhen_algo:
	rm -rf "$(WW_DIR)/who_and_when__algorithm-generated"
	$(PYTHON) -m data.error_localization.single_fault.ww_algorithm_generated

load_whowhen_hand:
	rm -rf "$(WW_DIR)/who_and_when__hand-crafted"
	$(PYTHON) -m data.error_localization.single_fault.ww_hand_crafted
