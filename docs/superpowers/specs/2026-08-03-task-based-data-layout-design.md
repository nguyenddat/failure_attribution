# Task-based data layout: who&when → error_localization, MAST → error_categorization

Ngày: 2026-08-03

## Mục tiêu

Đưa dataset theo **bài toán** thay vì theo tên cũ:

- who&when (`algorithm-generated` + `hand-crafted`) → `data/error_localization/single_fault/`
- MAST → `data/error_categorization/`

Đây là thay đổi thuần vị trí file + path constant. Không đổi schema JSON, không đổi nội dung file dữ liệu, không tạo experiment mới.

## Trạng thái hiện tại

```
data/
  single_fault/{utils.py, ww_algorithm_generated.py, ww_hand_crafted.py,
                json/who_and_when__algorithm-generated/ (126 file),
                json/who_and_when__hand-crafted/ (58 file)}
  fault_detection/{mast.py, build_agent_behaviors.py, patch_mas_name.py,
                   analyze_trajectory_length.py, split_by_length.py, figures/,
                   json/mast/ (1244 file: 1242 sample + metadata.json + length_split.json)}
  error_localization/     (rỗng)
  error_categorization/   (rỗng)
  multi_fault/            (không đụng tới)
```

## Layout đích

```
data/
  error_localization/single_fault/
    utils.py  ww_algorithm_generated.py  ww_hand_crafted.py
    who_and_when__algorithm-generated/*.json   (126)
    who_and_when__hand-crafted/*.json          (58)
  error_categorization/
    mast.py  build_agent_behaviors.py  patch_mas_name.py
    analyze_trajectory_length.py  split_by_length.py  figures/
    mast/*.json                                (1244)
  multi_fault/            (giữ nguyên)
```

Tầng `json/` trung gian bị bỏ ở cả hai task. `data/single_fault/` và `data/fault_detection/` biến mất hoàn toàn (kể cả `__pycache__`).

Quyết định đã chốt:

- Move hẳn, không copy — một nguồn sự thật duy nhất.
- Giữ 2 thư mục con cho who&when: 2 tập đều đánh số `0.json, 1.json, ...` nên đổ chung sẽ đụng tên.
- File `.py` đi theo file JSON.
- Giữ nguyên schema (`Data`/`AgentBehavior` bên error_localization, `Sample`/`Metadata` bên error_categorization). Không align 2 schema với nhau trong lần này.
- `experiments/fault_detection/` và `docs/fault-detection/` **giữ nguyên tên và vị trí**; chỉ sửa dòng import. Chấp nhận lệch tên (experiment tên `fault_detection` đọc data từ `error_categorization`), đổi sau nếu cần.
- Không tạo `experiments/error_localization/`.

## Thay đổi code

### 1. Path constant trong module data

| File (sau khi move) | Sửa |
|---|---|
| `data/error_localization/single_fault/ww_algorithm_generated.py` | `dataset_path = base_dir / "json" / ...` → bỏ `"json"`; xóa biến chết `output_dir` |
| `data/error_localization/single_fault/ww_hand_crafted.py` | như trên |
| `data/error_categorization/mast.py` | `json_dir = base_dir / "json" / "mast"` → `base_dir / "mast"` |
| `data/error_categorization/analyze_trajectory_length.py` | tự dựng `json_dir = base_dir / "json" / "mast"` → import `json_dir` từ `mast.py` cho nhất quán |

`split_by_length.py`, `build_agent_behaviors.py`, `patch_mas_name.py` đã import `json_dir` từ `mast.py`, chỉ cần sửa dòng import (mục 2).

### 2. Import (12 dòng)

`from data.single_fault.utils import ...` → `from data.error_localization.single_fault.utils import ...`
- `ww_algorithm_generated.py`, `ww_hand_crafted.py`

`from data.fault_detection.mast import ...` → `from data.error_categorization.mast import ...`
- `data/error_categorization/{build_agent_behaviors.py, patch_mas_name.py, split_by_length.py}`
- `experiments/fault_detection/baseline/run.py`, `baseline/methods/all_at_once.py`
- `experiments/fault_detection/fixed_size_segment/{run.py, methods/fixed_size_segment.py, split_by_length.py}`
- `experiments/fault_detection/overlapping_segment/{run.py, methods/overlapping_segment.py}`
- `experiments/fault_detection/analysis/compare_segmentation.py`
- `experiments/fault_detection/twin_comparison_segment/{analyze.py, inspect_clusters.py}`

### 3. Ref path phía experiments/single_fault (3 chỗ)

- `experiments/single_fault/utils/datasets.py` — `DATA_DIR` → `PROJECT_ROOT / "data" / "error_localization" / "single_fault"` (không còn `"json"`).
- `experiments/single_fault/experiments/dataset_analysis/dataset_characteristics.py` — `DATASET_DIRS` hardcode.
- `experiments/single_fault/experiments/dataset_analysis/trajectory_length_scatter.py` — `DATASET_DIRS` hardcode.

Key `ww_algorithm_generated` / `ww_hand_crafted` giữ nguyên, nên mọi file import `DATASET_DIRS` không phải sửa.

### 4. Doc

Sửa path và lệnh chạy (`python -m data.fault_detection.X` → `python -m data.error_categorization.X`):

- `CLAUDE.md`
- `docs/fault-detection/{data_process.md, length_split.md, fixed_step_segment.md, twin_comparison_segment.md}`
- `docs/single-fault/twin_comparison_segment.md`
- `docs/twin_comparison_cadence_analysis.md`

## Ngoài phạm vi

- Không đổi tên `experiments/fault_detection/` hay `docs/fault-detection/`.
- Không đụng `data/multi_fault/`, không đụng `experiments/*/output/`.
- Không align schema 2 task, không thêm `Metadata` cho who&when.
- Không tạo experiment/method mới.

## Kiểm chứng

Trước khi move: ghi lại số file JSON mỗi thư mục (126 / 58 / 1244).

Sau khi sửa xong, với `conda activate idea_segment`:

```bash
python -c "from data.error_categorization.mast import json_dir; print(json_dir.exists(), len(list(json_dir.glob('*.json'))))"
# -> True 1244

python -c "from experiments.single_fault.utils.datasets import DATASET_DIRS; print({k: (v.exists(), len(list(v.glob('*.json')))) for k, v in DATASET_DIRS.items()})"
# -> ww_algorithm_generated: (True, 126), ww_hand_crafted: (True, 58)

python -c "import data.error_localization.single_fault.ww_hand_crafted, data.error_localization.single_fault.ww_algorithm_generated"
python -c "import data.error_categorization.build_agent_behaviors, data.error_categorization.split_by_length"
python -c "import experiments.fault_detection.baseline.run, experiments.fault_detection.fixed_size_segment.run, experiments.fault_detection.overlapping_segment.run"
```

Import-only, không gọi LLM, không refetch HuggingFace.

Còn lại: `grep -r "data\.single_fault\|data/single_fault\|data\.fault_detection\|data/fault_detection" .` phải sạch (trừ log/lịch sử git).
