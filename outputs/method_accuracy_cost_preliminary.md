# So sanh so bo accuracy va cost cua 4 methods

Pham vi tai lieu nay gom 4 methods:

- `all_at_once`
- `step_by_step`
- `binary_search`
- `v1_multi_step`

`multi step v2` duoc bo qua theo yeu cau.

## Nguon so lieu

- Accuracy lay tu:
  - `outputs/accuracy/all_at_once.csv`
  - `outputs/accuracy/step_by_step.csv`
  - `outputs/accuracy/binary_search.csv`
  - `outputs/accuracy/v1_multi_step_accuracy.csv`
- Cost lay tu:
  - `outputs/cost/Algorithm-Generated_all_at_once_cost.csv`
  - `outputs/cost/Algorithm-Generated_step_by_step_cost.csv`
  - `outputs/cost/Algorithm-Generated_binary_search_cost.csv`
  - `outputs/cost/v1_multi_step_cost.csv`

Luu y: bo `cost` hien khong co gia tien USD thuc te trong CSV dau ra. Vi vay phan so sanh cost ben duoi dung cac proxy sau:

- `avg input tokens`
- `avg output tokens`
- `avg total tokens = input + output`
- `avg latency`

Tat ca so lieu duoi day dang tinh tren `126` samples.

## Bang tong hop

| Method | Agent accuracy | Step accuracy | Avg latency (s) | Avg input tokens | Avg output tokens | Avg total tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `all_at_once` | 52.38% | 11.11% | 2.33 | 3543.72 | 75.22 | 3618.94 |
| `step_by_step` | 31.75% | 19.05% | 4.64 | 2222.39 | 143.50 | 2365.89 |
| `binary_search` | 31.75% | 11.90% | 14.35 | 10871.85 | 92.60 | 10964.45 |
| `v1_multi_step` | 22.22% | 16.67% | 3.66 | 2790.87 | 127.17 | 2918.04 |

## Nhan xet nhanh

- Neu uu tien `agent accuracy`, `all_at_once` dang tot nhat kha ro: `52.38%`, cao hon dang ke so voi 3 method con lai.
- Neu uu tien `step accuracy`, `step_by_step` dang tot nhat: `19.05%`.
- Neu uu tien `cost` theo token, `step_by_step` dang re nhat voi `2365.89` total tokens/sample.
- Neu uu tien `speed`, `all_at_once` dang nhanh nhat voi `2.33s/sample`.
- `binary_search` hien la phuong an ton kem nhat ve input token va latency, nhung accuracy khong vuot troi.
- `v1_multi_step` co latency va token o muc trung gian, nhung accuracy hien chua noi bat hon `step_by_step`.

## Ket luan so bo

- `all_at_once`: hop ly neu uu tien bat dung `agent` va can toc do nhanh.
- `step_by_step`: hop ly nhat neu uu tien bat dung `step`, dong thoi cung la phuong an tiet kiem token nhat trong 4 methods.
- `binary_search`: hien chua co loi the ro rang ve accuracy so voi chi phi bo ra.
- `v1_multi_step`: co the van dang giu de tiep tuc tinh chinh, nhung o snapshot hien tai chua thang ro tren ca accuracy lan cost.

## Goi y cach chot narrative trong bao cao

Co the dien dat ngan gon nhu sau:

> Tren 126 mau, `all_at_once` cho `agent accuracy` cao nhat, trong khi `step_by_step` cho `step accuracy` tot nhat va dong thoi co chi phi token thap nhat. `binary_search` tieu ton chi phi lon nhat nhung khong mang lai cai thien accuracy tuong ung. `v1_multi_step` dang nam giua ve cost nhung chua cho thay loi the du ro ve do chinh xac.
