"""
train_yolo_baselines_deploy.py — YOLO/RT-DETR baselines cho nhánh DEPLOY.
Giống train_yolo_baselines.py nhưng train trên COCO-MIX; val trên VisDrone-val (ĐỒNG NHẤT với deploy).
So công bằng với runs/train_v17_deploy (cùng data mix + cùng val). flight_val giữ làm held-out eval sau.
Resume-safe, 1 model lỗi không chặn model khác.
"""
import os
from ultralytics import YOLO, RTDETR
from ultralytics.utils.torch_utils import get_flops

DATA = "configs/visdrone_coco_ultralytics.yaml"
SUMMARY = "runs/paper_queue/SUMMARY_DEPLOY.txt"
# PHẢI tuyệt đối: ultralytics ghép project tương đối vào runs_dir của nó
# -> "runs/yolo_deploy" biến thành runs/detect/runs/yolo_deploy (lệch chỗ script đi tìm best.pt).
PROJ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "yolo_deploy")
EPOCHS = 150
# (tên, loại, batch, kwargs riêng) — kwargs ghi đè mặc định patience=20
MODELS = [
    ("yolov8n",  "yolo",   16, {}),
    ("yolo11n",  "yolo",   16, {}),
    ("yolo26n",  "yolo",   16, {}),
    ("yolov10n", "yolo",   16, {}),
    # RT-DETR: 'auto' chọn MuSGD(lr=0.01) -> sập ngay (xem phase-1). Transformer cần AdamW lr~1e-4 + patience lớn.
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
    append("\n===== BASELINES DEPLOY (COCO-mix train, VisDrone-val @640 đồng nhất deploy; đa dạng họ) =====")
    for name, typ, bs, extra in MODELS:
        donemark = f"{PROJ}/{name}/DEPLOY_DONE"
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
                    kw = dict(patience=20); kw.update(extra)
                    print(f"[base] {name}: train mới (batch {bs}, {kw})")
                    Cls(f"{name}.pt").train(data=DATA, imgsz=640, epochs=EPOCHS, batch=bs,
                                            workers=8, name=name, project=PROJ, exist_ok=True, **kw)
            model = Cls(best)
            # model.info(verbose=False) trả None ở ultralytics 8.4.x -> tính trực tiếp
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
            print(f"[base] {name} LỖI: {str(e)[:80]}")
            append(f"{name}: LỖI ({str(e)[:60]}) — bỏ qua")
            open(donemark, "w").close()

    open("runs/paper_queue/BASELINES_DEPLOY_DONE", "w").close()
    print("[base] TẤT CẢ BASELINE DEPLOY HOÀN TẤT")


if __name__ == "__main__":
    main()
