# who&when: schema tầng project + Makefile `load_whowhen`

Ngày: 2026-08-04

## Mục tiêu

Tiền xử lý dữ liệu who&when với hai thay đổi:

1. Tách schema pydantic ra khỏi `data/error_localization/single_fault/utils.py`, đưa lên package `schemas/` ở tầng project để cả `data/` (ghi) lẫn `experiments/` (đọc) cùng import một nguồn.
2. Thêm `Makefile` ở root với target `load_whowhen`: xoá thư mục JSON cũ rồi tải lại từ HuggingFace.

Quyết định đã chốt:

- **Mỗi dataset một schema riêng.** Không thống nhất who&when / MAST / TRAIL vào một model chung. Lý do: ba nguồn có cấu trúc nhãn khác hẳn nhau (who&when single-fault step+agent, TRAIL multi-fault theo span, MAST failure-mode cấp trace); ép chung sẽ tạo model toàn field optional.
- **Vị trí thì thống nhất.** Mọi schema nằm dưới `schemas/`, đặt tên theo dataset. Lần này chỉ làm `schemas/who_and_when.py`; `schemas/mast.py`, `schemas/trail.py` migrate sau, ngoài phạm vi spec này.
- Giữ nguyên tên class và field hiện tại ⇒ 184 file JSON đã sinh vẫn parse được, `experiments/single_fault/` chỉ phải sửa dòng import (nếu có).

## Trạng thái hiện tại

```
data/error_localization/single_fault/
  utils.py                              # AgentBehavior, Data, dataset_name_to_filename
  ww_algorithm_generated.py             # -> who_and_when__algorithm-generated/ (126 json)
  ww_hand_crafted.py                    # -> who_and_when__hand-crafted/ (58 json)
experiments/single_fault/utils/datasets.py   # DATASET_DIRS trỏ vào 2 thư mục trên
```

Chưa có `Makefile`. Chưa có `pyproject.toml` / venv trong repo — chạy bằng conda env đã activate.

Vấn đề của code hiện tại:

- Schema chôn trong `data/`, experiment muốn dùng phải import xuyên qua data layer.
- `dataset_path.mkdir(...)` chạy ở module level ⇒ import module là tạo thư mục.
- `if os.path.exists(file_path): continue` khiến không có cách nào ép tải lại; dữ liệu cũ sai không tự sửa được.

## Layout đích

```
rs_who&when/
  Makefile                      (mới)
  schemas/                      (mới)
    __init__.py
    who_and_when.py             AgentBehavior, Data
  data/error_localization/single_fault/
    utils.py                    chỉ còn dataset_name_to_filename
    ww_algorithm_generated.py
    ww_hand_crafted.py
    who_and_when__algorithm-generated/*.json
    who_and_when__hand-crafted/*.json
```

## Thay đổi chi tiết

### 1. `schemas/who_and_when.py`

Chuyển nguyên văn từ `utils.py`, không đổi tên, không đổi field:

```python
class AgentBehavior(BaseModel):
    step: int
    agent_name: str
    content: str

class Data(BaseModel):
    question: str
    trajectory: List[AgentBehavior]
    mistake_step: int
    mistake_agent: str
```

Không thêm `ground_truth` / `mistake_reason` lần này dù HuggingFace có sẵn: chưa experiment nào dùng (YAGNI). Thêm sau là thay đổi additive, JSON đã sinh vẫn hợp lệ.

`dataset_name_to_filename()` là util của loader, không phải schema ⇒ ở lại `utils.py`.

### 2. Loader `ww_algorithm_generated.py` / `ww_hand_crafted.py`

Giữ cấu trúc hai file tách rời. Khác biệt `history_to_trajectory` (`item["name"]` với algorithm-generated, `item["role"]` với hand-crafted) là khác biệt thật của nguồn dữ liệu, giữ nguyên.

Sửa:

| Chỗ | Trước | Sau |
|---|---|---|
| import schema | `from data.error_localization.single_fault.utils import AgentBehavior, Data, dataset_name_to_filename` | `from schemas.who_and_when import AgentBehavior, Data` + `from data.error_localization.single_fault.utils import dataset_name_to_filename` |
| tạo thư mục | `dataset_path.mkdir(...)` ở module level | chuyển vào trong `load_data_path()` |
| ghi file | `if os.path.exists(file_path): continue` | giữ nguyên |

Việc xoá thư mục thuộc về Makefile, không thuộc loader — loader vẫn idempotent, gọi lại khi thư mục trống là tải đầy đủ. Không thêm cờ `--reset` vào Python (YAGNI: đã có `rm -rf` trong target).

### 3. `Makefile` (root)

```make
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
```

Chạy trong Git Bash (GNU Make 4.4.1 đã xác nhận có sẵn). `rm -rf` nằm trong recipe make, không nằm trong Python — xoá là hành động của người gọi target, không phải của loader.

`PYTHON ?= python` cho phép `make load_whowhen PYTHON=/c/Users/dinhd/miniconda3/envs/<env>/python.exe` khi chưa activate conda env.

**Cảnh báo:** `make load_whowhen` xoá 184 file JSON hiện có rồi tải lại từ HuggingFace. Cần mạng.

### 4. Cập nhật consumer

Grep `data.error_localization.single_fault.utils` toàn repo (loại trừ `.claude/worktrees/`) và đổi dòng import schema sang `schemas.who_and_when`. Hiện chỉ hai loader import; `experiments/single_fault/` đọc JSON qua `DATASET_DIRS` chứ không import schema của data layer, nên dự kiến không phải sửa.

## Kiểm chứng

1. `make load_whowhen` chạy hết, không lỗi.
2. `who_and_when__algorithm-generated/` có 126 file, `who_and_when__hand-crafted/` có 58 file.
3. `Data.model_validate_json()` parse được file `0.json` của cả hai thư mục.
4. `git diff --stat` trên các file JSON: nội dung sau khi tải lại trùng với bản cũ (chỉ chấp nhận khác biệt nếu upstream HuggingFace đã đổi).
5. Một script trong `experiments/single_fault/` vẫn import và chạy được.

## Ngoài phạm vi

- Migrate schema MAST / TRAIL sang `schemas/`.
- Thêm `pyproject.toml`, khoá dependency.
- Thêm field `ground_truth` / `mistake_reason`.
- Bất kỳ thay đổi nào trong `experiments/`.
