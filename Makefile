PYTHON ?= python
WW_DIR := data/error_localization/single_fault
MAST_DIR := data/error_categorization/mast
TRAIL_DIR := data/error_localization/multi_fault/trail
AEGIS_DIR := data/error_localization/multi_fault/aegis

# Shell-agnostic recursive delete (make picks cmd.exe on Windows, sh elsewhere).
RM_DIR = $(PYTHON) -c "import shutil,sys; shutil.rmtree(sys.argv[1], ignore_errors=True)"

.PHONY: load_whowhen load_whowhen_algo load_whowhen_hand load_mast load_trail load_aegis

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
