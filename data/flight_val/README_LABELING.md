# flight_val — val set 3-6m THẬT (chuyến bay 2026-07-21)

Val set thật đầu tiên cho section "Real-world Deployment" của paper. Nguồn: `2026-07-21 09-43-12.mkv`
(quay màn GCS ~6m, POSHOLD). Đã crop UI (top 66 / left 105 px), lấy 1 frame/giây, đoạn bay 18–150s.

- `images/` : 132 ảnh 1174×654 (đã crop sạch UI).
- `labels/` : YOLO txt (đã **pre-label bằng model deploy** — CẦN SỬA TAY).
- `classes.txt` : 1 class duy nhất = `person`.

## Class gán nhãn: CHỈ 1 class = person (id 0)

Định dạng mỗi dòng (YOLO, chuẩn hoá theo ảnh):
```
0 x_center y_center width height
```
(0 = person; toạ độ [0,1] theo 1174×654).

## Luật gán (đọc kỹ — val set này CHẤM CHÍNH model deploy)

1. **CHỈ box người.** Box ôm sát toàn thân nhìn thấy (đầu→chân, cả tay chân).
2. **XOÁ false positive** model đoán sai (bóng, vết bẩn nền, icon UI, chấm/vật dưới đất).
3. **THÊM người model bỏ sót** (người nhỏ/mờ ở rìa — đây chính là ca model yếu, phải có để đo đúng).
4. Người bị **cắt ở rìa khung** → vẫn box phần nhìn thấy.
5. Frame **không có người** → để file `.txt` **rỗng** (là negative hợp lệ, giúp đo false-positive).
6. KHÔNG box: chữ "FRAME CENTER", crosshair, icon góc phải, pill dưới đáy, bóng đổ, vật thể dưới đất.

> ⚠️ Vì set này dùng để ĐO chính model đã pre-label, **đừng tin mù** — soi từng frame, sửa cho đúng
> sự thật, nếu không số AP sẽ bị thiên vị (model tự chấm mình).

## Tool gợi ý
- **labelImg** (đơn giản, YOLO format): `pip install labelImg` → `labelImg data/flight_val/images data/flight_val/classes.txt`
  Bật "YOLO" mode, lưu vào `data/flight_val/labels`. Phím `w`=vẽ box, `d`=ảnh sau, `a`=ảnh trước.
- Hoặc **X-AnyLabeling** / Roboflow (web, có auto-label).

## Sau khi gán xong — chạy eval trên val THẬT
```bash
python run_eval_coco.py --weights runs/train_v17_deploy/best.pt \
  --val-images data/flight_val/images --val-labels data/flight_val/labels \
  --img-size 512 --profile light --device cuda:0
```
So với `runs/train_v17_light/best.pt` (bản KHÔNG COCO) trên cùng val này → nếu deploy (COCO) **cao hơn**
= bằng chứng "COCO có ích cho tầm thấp" → mở khoá full-pivot. Xem [[project_paper_hybrid]].
