"""
train_yolo_baselines.py — Train YOLOv8n / YOLO11n / YOLO26n trên VisDrone-person để so sánh paper.
==================================================================================================
Cùng split + img 640 như v17. Từ COCO-pretrained (chuẩn dùng YOLO — baseline mạnh nhất).
Resume-safe: crash thì chạy lại script, tự tiếp. Ghi params/FLOPs/mAP vào SUMMARY.

Chạy: python train_yolo_baselines.py   (tự lặp 3 model)
"""
import os
from ultralytics import YOLO, RTDETR
from ultralytics.utils.torch_utils import get_flops

DATA = "configs/visdrone_person_ultralytics.yaml"
SUMMARY = "runs/paper_queue/SUMMARY.txt"
# PHẢI tuyệt đối: ultralytics ghép project tương đối vào runs_dir của nó
# -> "runs/yolo" biến thành runs/detect/runs/yolo (lệch chỗ script đi tìm best.pt).
PROJ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "yolo")
EPOCHS = 150
# (tên, loại, batch) — ĐA DẠNG HỌ: CNN 1-stage / NMS-free / Transformer(DETR)
# (tên, loại, batch, kwargs riêng) — kwargs ghi đè mặc định patience=20
MODELS = [
    ("yolov8n",  "yolo",   16, {}),   # CNN 1-stage — baseline kinh điển
    ("yolo11n",  "yolo",   16, {}),   # CNN 1-stage — chuẩn hiện hành
    ("yolo26n",  "yolo",   16, {}),   # CNN NMS-free — mới nhất
    ("yolov10n", "yolo",   16, {}),   # NMS-free end-to-end (NeurIPS 2024)
    # RT-DETR = Transformer. 'optimizer=auto' chọn MuSGD(lr=0.01) -> SẬP ngay epoch 2-4
    # (mAP 0.47->0.07), rồi bò lại nhưng không kịp vượt ep1 -> early-stop ở ep21 = số RÁC.
    # DETR phải dùng AdamW lr~1e-4, và hội tụ CHẬM nên patience phải lớn.
    ("rtdetr-l", "rtdetr",  4, {"optimizer": "AdamW", "lr0": 1e-4, "patience": 50}),
]


def append(s):
    os.makedirs("runs/paper_queue", exist_ok=True)
    with open(SUMMARY, "a") as f:
        f.write(s + "\n")


def train_done(name):
    csv = f"{PROJ}/{name}/results.csv"
    if not os.path.exists(csv):
        return False
    with open(csv) as f:
        n = sum(1 for _ in f) - 1
    return n >= EPOCHS


def main():
    append("\n===== BASELINES (VisDrone-person @640, từ COCO-pretrained; đa dạng họ) =====")
    for name, typ, bs, extra in MODELS:
        donemark = f"{PROJ}/{name}/PAPER_DONE"
        if os.path.exists(donemark):
            print(f"[base] {name} đã xong, bỏ qua"); continue
        Cls = RTDETR if typ == "rtdetr" else YOLO
        last = f"{PROJ}/{name}/weights/last.pt"
        best = f"{PROJ}/{name}/weights/best.pt"
        try:
            if not train_done(name):
                if os.path.exists(last):
                    print(f"[base] {name}: RESUME"); Cls(last).train(resume=True)
                else:
                    kw = dict(patience=20); kw.update(extra)   # mặc định 20, model đặc biệt tự ghi đè
                    print(f"[base] {name}: train mới (batch {bs}, {kw})")
                    Cls(f"{name}.pt").train(data=DATA, imgsz=640, epochs=EPOCHS, batch=bs,
                                            workers=8, name=name, project=PROJ, exist_ok=True, **kw)
            model = Cls(best)
            # model.info(verbose=False) trả None ở ultralytics 8.4.x -> tính trực tiếp cho chắc
            params = sum(p.numel() for p in model.model.parameters()) / 1e6
            try:
                flops = get_flops(model.model, imgsz=640)
            except Exception:
                flops = 0.0
            metrics = model.val(data=DATA, imgsz=640, split="val", verbose=False)
            append(f"{name}: params={params:.2f}M  GFLOPs={flops:.1f}  "
                   f"mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}")
            open(donemark, "w").close()
            print(f"[base] {name} XONG")
        except Exception as e:
            # 1 model lỗi (vd RT-DETR OOM) KHÔNG chặn model khác
            print(f"[base] {name} LỖI: {str(e)[:80]}")
            append(f"{name}: LỖI ({str(e)[:60]}) — bỏ qua")
            open(donemark, "w").close()   # đánh dấu đã thử (khỏi retry vô hạn)

    open("runs/paper_queue/BASELINES_DONE", "w").close()   # xong toàn bộ (best-effort)
    print("[base] TẤT CẢ BASELINE HOÀN TẤT")

    print("[yolo] TẤT CẢ BASELINE HOÀN TẤT")


if __name__ == "__main__":
    main()
