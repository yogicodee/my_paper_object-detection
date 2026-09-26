# Safety-PPE Detection Unified Dataset v1 (stratified re-split)

Bộ dữ liệu phát hiện thiết bị bảo hộ lao động (PPE), định dạng **YOLO**, **12 lớp**
gồm 6 cặp *tuân thủ / vi phạm*: `helmet`, `no_helmet`, `vest`, `no_vest`, `gloves`,
`no_gloves`, `goggles`, `no_goggles`, `boots`, `no_boots`, `mask`, `no_mask`.

- **17,951 ảnh** / **51,179 bounding box**
- Chia **train/val/test = 12,566 / 3,590 / 1,795** ảnh (70/20/10)
- **Toàn bộ ảnh là 640x640.** Ảnh nguồn `c` vốn là 416x416 đã được **upscale
  LANCZOS lên 640x640** trong bản re-split này; ảnh nguồn `a`/`b` vốn đã đúng
  640x640 nên được giữ nguyên bản, không re-encode.

> Vì ảnh nguồn đều **vuông** (416x416 → 640x640) nên đây là phép **scale đều**,
> `padding = 0`, do đó **nhãn YOLO đã chuẩn hoá không thay đổi**.

## Cấu trúc

```
data.yaml
manifest.csv                     # file, split, source, source_image_id, n_boxes
DATASET_CARD.md
distribution_report_final.csv
train/images/  train/labels/
val/images/    val/labels/
test/images/   test/labels/
```

`data.yaml` khai báo `names` theo đúng thứ tự chỉ số lớp 0..11.

## Nguồn dữ liệu và ghi công (attribution)

Gộp từ 3 bộ dữ liệu Roboflow công khai — **tất cả đều CC BY 4.0, bắt buộc ghi công**:

1. **Hard Hat Detection v2** — computer-vision-filqz (`head` → `no_helmet`, `helmet`; lớp `person` bị loại)
2. **PPE detection v1** — testcasque (các cặp vest/boots/gloves/goggles/helmet)
3. **PPE Detection v5** — mohamed-traore-2ekkp (các cặp gloves/goggles/helmet/mask)

Tên file mang tiền tố `a_`, `b_`, `c_` tương ứng 3 nguồn trên.

**License: CC BY 4.0** (kế thừa từ 3 nguồn — vui lòng giữ nguyên phần ghi công này).

## Quy trình dựng (tái lập được, seed 42)

1. Khử trùng lặp trong từng nguồn theo ảnh gốc (bỏ hậu tố Roboflow `.rf.<hash>`) — loại các bản augment.
2. Khử trùng lặp chính xác giữa các nguồn (MD5).
3. Ánh xạ nhãn về taxonomy 12 lớp thống nhất; loại lớp `person`.
4. Giới hạn ảnh nền (không còn box) ở mức 10%.
5. Lấy mẫu 50% ảnh chỉ-có-helmet của nguồn A để giảm áp đảo của lớp `helmet`.
6. Chia **theo nhóm ảnh gốc** — không ảnh gốc nào xuất hiện ở hai split (split gốc của nguồn 2 & 3 có rò rỉ chéo nên đã bị loại bỏ).
7. Chuẩn hoá 100% ảnh về 640x640.

## Stratified re-split

Bản v1 chia 70/20/10 có bảo toàn nhóm nhưng **không phân tầng**, nên các lớp hiếm
thiếu mẫu ở test (`no_vest` 70 box, `mask` 97 box) — không đủ để ước lượng AP đáng tin.

Bản này chia lại bằng **greedy iterative stratification theo nhóm ảnh gốc**
(`stratified_resplit.py`):

- **Ràng buộc cứng:** mọi ảnh cùng `source_image_id` luôn nằm cùng một split — ràng buộc chống rò rỉ không bao giờ bị phá vỡ.
- **Thứ tự tham lam:** lặp lại — chọn lớp còn ít box nhất, gán các nhóm chứa lớp đó vào split đang thiếu hạn ngạch của lớp đó nhất.
- **Seed 42, tất định:** mọi tập hợp đều được `sorted()` trước khi duyệt, nên kết quả **không phụ thuộc `PYTHONHASHSEED`** (đã kiểm chứng với 3 hash seed khác nhau cho ra phân chia giống hệt).
- **Hàm phạt:** thiếu box so với ngưỡng tối thiểu (trọng số áp đảo) ≫ lệch hạn ngạch ảnh > độ lệch tỉ lệ. Thành phần tỉ lệ có **số hạng mềm bậc hai `w·d²` áp dụng ở mọi nơi**, kể cả **trong dải ±5%** — nếu chỉ phạt phần vượt ngưỡng thì bên trong dải không có gradient và thuật toán dừng ở trạng thái "vừa đủ lọt ngưỡng" (lớp `helmet` từng bị đẩy xuống 6,1% ở test).
- **Ràng buộc ảnh nền:** các nhóm **không chứa box nào** (ảnh nền / cảnh trống) được cấp **hạn ngạch riêng 70/20/10** (dung sai ±2 điểm %). Nhóm ảnh nền không đóng góp vào bất kỳ lớp nào, nên nếu không ép riêng thì greedy dồn hết chúng vào split còn nhiều hạn ngạch nhất — lần chia trước ra **584/0/3**, khiến val không còn cảnh trống nào để đo false-positive. Hàm phạt vì vậy có thêm số hạng cho độ lệch tỉ lệ ảnh nền, đo bằng điểm phần trăm giống các lớp.
- **Repair:** hill-climbing có định hướng, chỉ chấp nhận bước làm **giảm** hàm phạt nên luôn hội tụ.

Kết quả: độ lệch tỉ lệ lớn nhất trên toàn bộ 12 lớp chỉ còn **1.08 điểm phần trăm**
(bản v1: 11,1 điểm), hạn ngạch ảnh khớp chính xác 70/20/10.

### Ngưỡng tối thiểu đã đạt

Mọi lớp (trừ `no_boots`, được miễn vì tổng chỉ có 37 box) đều có
**≥ 100 box ở test** và **≥ 150 box ở val**;
riêng `no_vest` test ≥ 100 và `mask` test ≥ 110.

### So sánh test: v1 → resplit (các lớp thay đổi đáng kể)

| Lớp | test (v1) | test (resplit) | Δ | ngưỡng | trạng thái |
|---|---:|---:|---:|---:|:---:|
| `no_vest` | 70 | **112** | +42 | 100 | đạt |
| `mask` | 97 | **118** | +21 | 110 | đạt |
| `boots` | 120 | **111** | -9 | 100 | đạt |
| `no_boots` | 3 | **4** | +1 | miễn | miễn trừ |

## Phân bố lớp (số box)

| Lớp | train | val | test | tổng | % train/val/test |
|---|---:|---:|---:|---:|:---:|
| helmet | 11,537 | 3,296 | 1,649 | 16,482 | 70.0/20.0/10.0 |
| no_helmet | 5,797 | 1,655 | 827 | 8,279 | 70.0/20.0/10.0 |
| vest | 2,033 | 581 | 290 | 2,904 | 70.0/20.0/10.0 |
| no_vest | 786 | 225 | 112 | 1,123 | 70.0/20.0/10.0 |
| gloves | 3,515 | 1,004 | 501 | 5,020 | 70.0/20.0/10.0 |
| no_gloves | 4,315 | 1,231 | 616 | 6,162 | 70.0/20.0/10.0 |
| goggles | 2,581 | 737 | 368 | 3,686 | 70.0/20.0/10.0 |
| no_goggles | 2,708 | 774 | 386 | 3,868 | 70.0/20.0/10.0 |
| boots | 777 | 221 | 111 | 1,109 | 70.1/19.9/10.0 |
| no_boots | 26 | 7 | 4 | 37 | 70.3/18.9/10.8 |
| mask | 830 | 237 | 118 | 1,185 | 70.0/20.0/10.0 |
| no_mask | 927 | 265 | 132 | 1,324 | 70.0/20.0/10.0 |
| **TỔNG** | **35,832** | **10,233** | **5,114** | **51,179** | **70.0/20.0/10.0** |

## Số ảnh chứa mỗi lớp

| Lớp | train | val | test | tổng |
|---|---:|---:|---:|---:|
| helmet | 5,077 | 1,322 | 569 | 6,968 |
| no_helmet | 1,316 | 731 | 499 | 2,546 |
| vest | 1,121 | 472 | 251 | 1,844 |
| no_vest | 371 | 180 | 105 | 656 |
| gloves | 1,401 | 576 | 337 | 2,314 |
| no_gloves | 1,993 | 533 | 296 | 2,822 |
| goggles | 2,279 | 660 | 342 | 3,281 |
| no_goggles | 1,970 | 675 | 343 | 2,988 |
| boots | 348 | 107 | 59 | 514 |
| no_boots | 14 | 4 | 3 | 21 |
| mask | 250 | 163 | 110 | 523 |
| no_mask | 755 | 221 | 129 | 1,105 |
| **Ảnh trong split** | **12,566** | **3,590** | **1,795** | **17,951** |

### Ảnh nền (background images, 0 box)

| | train | val | test | tổng |
|---|---:|---:|---:|---:|
| Số ảnh | 411 | 117 | 59 | 587 |
| Tỉ lệ | 70.0% | 19.9% | 10.1% | 100% |

Ảnh nền được **ràng buộc phân bổ theo tỉ lệ split 70/20/10** (dung sai ±2 điểm %),
để mỗi split — đặc biệt là val — đều có cảnh trống dùng đo **false-positive**.

## Kiểm định

`verify_dataset.py` chạy trên bản này đạt **7/7 PASS**: class ID nguyên trong [0,11]
và khớp thứ tự `names`; toạ độ box trong (0,1] và không box rỗng; ghép cặp 1-1
images/labels không file mồ côi; 100% ảnh 640x640; mỗi `source_image_id` chỉ ở
đúng 1 split; 20 ảnh nguồn `c` sau upscale giải nén được bằng PIL không lỗi;
đạt toàn bộ ngưỡng phân bổ box.

## Hạn chế đã biết

- **`no_boots` chỉ có 37 box trên toàn bộ dataset.** Lớp này được **giữ trong dataset nhưng PHẢI LOẠI khỏi metric tổng hợp** (mAP trung bình) khi đánh giá — chỉ báo cáo riêng và xem là *không đủ dữ liệu để kết luận*.
- Các lớp ít dữ liệu (`no_vest`, `boots`, `mask`, `no_mask` đều dưới 1.500 box) cần class-weighted loss / oversampling và **bắt buộc báo cáo per-class**, không chỉ nhìn mAP trung bình.
- Mất cân bằng lớp cực đại: `helmet` nhiều gấp ~445 lần `no_boots`.
- `vest`/`no_vest` và `boots`/`no_boots` chỉ đến từ **một nguồn duy nhất** (domain bias) — kết quả trên hai cặp này không suy rộng ra domain khác.
- Near-duplicate xuyên nguồn chỉ được loại khi **trùng byte** (MD5); vẫn có thể còn ảnh gần giống nhau nằm ở hai split khác nhau.
- Ảnh nguồn `c` được upscale 416→640, **không thêm thông tin mới**; vật thể nhỏ (`goggles`, `gloves`) ở nguồn này vốn đã mờ hơn nguồn khác.
