PYTHON ?= python
WW_DIR := data/error_localization/single_fault
MAST_DIR := data/error_categorization/mast
TRAIL_DIR := data/error_localization/multi_fault/trail
AEGIS_DIR := data/error_localization/multi_fault/aegis
AEB_DIR := data/error_localization/single_fault/agent_error_bench

# Shell-agnostic recursive delete (make picks cmd.exe on Windows, sh elsewhere).
RM_DIR = $(PYTHON) -c "import shutil,sys; shutil.rmtree(sys.argv[1], ignore_errors=True)"

.PHONY: load_whowhen load_whowhen_algo load_whowhen_hand load_mast load_trail load_aegis load_agent_error_bench
.PHONY: length_all length_whowhen_algo length_whowhen_hand length_mast length_trail length_aegis length_agent_error_bench

load_whowhen: load_whowhen_algo load_whowhen_hand

load_whowhen_algo:
	$(RM_DIR) "$(WW_DIR)/who_and_when__algorithm-generated"
	$(PYTHON) -m data.error_localization.single_fault.ww_algorithm_generated

load_whowhen_hand:
	$(RM_DIR) "$(WW_DIR)/who_and_when__hand-crafted"
	$(PYTHON) -m data.error_localization.single_fault.ww_hand_crafted

load_mast:
	$(RM_DIR) "$(MAST_DIR)"
	$(PYTHON) -m data.error_categorization.mast

load_trail:
	$(RM_DIR) "$(TRAIL_DIR)"
	$(PYTHON) -m data.error_localization.multi_fault.trail

load_aegis:
	$(RM_DIR) "$(AEGIS_DIR)"
	$(PYTHON) -m data.error_localization.multi_fault.aegis

load_agent_error_bench:
	$(RM_DIR) "$(AEB_DIR)"
	$(PYTHON) -m data.error_localization.single_fault.agent_error_bench

# Trace length distribution (tokens + steps); figures are overwritten in place.
length_all: length_whowhen_algo length_whowhen_hand length_mast length_trail length_aegis length_agent_error_bench

length_whowhen_algo:
	$(PYTHON) -m data.error_localization.single_fault.ww_algorithm_generated_length

length_whowhen_hand:
	$(PYTHON) -m data.error_localization.single_fault.ww_hand_crafted_length

length_mast:
	$(PYTHON) -m data.error_categorization.mast_length

length_trail:
	$(PYTHON) -m data.error_localization.multi_fault.trail_length

length_aegis:
	$(PYTHON) -m data.error_localization.multi_fault.aegis_length

length_agent_error_bench:
	$(PYTHON) -m data.error_localization.single_fault.agent_error_bench_length
