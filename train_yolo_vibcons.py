"""
train_yolo_vibcons.py — ÁP vibration-consistency lên detector CỦA NGƯỜI KHÁC (YOLO11n)
======================================================================================

Đây là thí nghiệm nâng claim của paper từ
    "một mẹo cho model của tôi"   ->   "một kỹ thuật cho detector nhỏ nói chung".
Generality hiện tại chỉ có v15+v17 — hai model CÙNG HỌ, cùng loss, cùng optimizer của
mình. Reviewer sẽ nói đó không phải generality. Chạy được file này thì có.

CÁCH LÀM: không sửa ultralytics, chỉ ghi đè `model.loss()` (điểm móc sạch của
`BaseModel.loss(batch, preds)`), giữ nguyên toàn bộ pipeline train của họ:

    loss_detect = criterion(f(x), y)                      <- Y HỆT bản gốc, trên ảnh SẠCH
    loss       += lambda_vib * consistency(f(T(x)), sg[f(x)])

Ở ultralytics 8.4.x, forward lúc train trả dict:
    boxes  (B, 4*reg_max, A)   logit DFL
    scores (B, nc, A)          logit lớp
    feats  list[3]             feature neck (đầu vào của head Detect)
-> khớp 1-1 với 3 thành phần consistency của bản pfdet (cls / box / feature).
Khác biệt duy nhất: box của YOLO là PHÂN PHỐI DFL (4 cạnh × reg_max bin), nên
consistency của box dùng KL trên các bin thay vì smooth-L1 trên toạ độ — đúng bản chất
đại lượng hơn, và vẫn gate theo độ tự tin của nhánh sạch.

Chạy (finetune từ baseline yolo11n ĐÃ train, giống hệt cách vib finetune từ v17_light):
    python train_yolo_vibcons.py --weights runs/yolo/yolo11n/weights/best.pt \
        --name yolo11n_vibcons --epochs 120 --batch 8

Xong -> thêm vào YOLO_MODELS trong paper_eval_unified.py rồi chạy lại để lấy retain%.
"""

import argparse
import os
import random

import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer

from vib_consistency import corrupt_batch

HERE = os.path.dirname(os.path.abspath(__file__))
EPS = 1e-6


# ============================================================
#  Consistency trên output dict của ultralytics
# ============================================================
def vib_consistency_dict(pred_vib, pred_clean, reg_max, T=2.0, conf_thr=0.1,
                         w_cls=1.0, w_box=1.0, w_feat=0.5):
    """
    pred_* : dict {boxes (B,4*reg_max,A), scores (B,nc,A), feats list}
    Nhánh sạch LUÔN detach (stop-gradient) — nếu không, cách rẻ nhất để giảm loss là
    làm hỏng nhánh sạch cho khớp nhánh rung.
    """
    if "scores" not in pred_vib:
        # yolo26n / yolov10n (end2end=True) tra ve {one2many, one2one} — CAU TRUC KHAC.
        # Ap vib-consistency len chung can quyet dinh THEM: ep nhat quan tren nhanh o2o,
        # nhanh o2m, hay ca hai? Do la lua chon thiet ke moi, khong phai chinh vai dong.
        raise NotImplementedError(
            f"Model tra ve {sorted(pred_vib)} thay vi ['boxes','feats','scores']. "
            "Day la detector end2end (yolo26n / yolov10n) voi 2 nhanh one2many + one2one. "
            "Dung yolo11n hoac yolov8n (Detect thuong, reg_max=16), hoac mo rong ham nay "
            "de xu ly ca 2 nhanh.")
    s_v = pred_vib["scores"]
    s_c = pred_clean["scores"].detach()

    # --- 1) classification consistency: soft target có nhiệt độ ---
    tgt = torch.sigmoid(s_c / T)
    l_cls = F.binary_cross_entropy_with_logits(s_v / T, tgt) * (T * T)

    # --- 2) box consistency: KL trên phân phối DFL, gate theo tự tin nhánh sạch ---
    B, C, A = pred_vib["boxes"].shape
    b_v = pred_vib["boxes"].view(B, 4, reg_max, A)
    b_c = pred_clean["boxes"].detach().view(B, 4, reg_max, A)
    logp_v = F.log_softmax(b_v, dim=2)
    p_c = F.softmax(b_c, dim=2)
    kl = (p_c * (p_c.clamp_min(EPS).log() - logp_v)).sum(dim=2).mean(dim=1)   # (B,A)
    mask = (torch.sigmoid(s_c).amax(dim=1) > conf_thr).float()                # (B,A)
    l_box = (kl * mask).sum() / (mask.sum() + EPS)

    # --- 3) feature consistency ---
    l_feat = s_v.new_zeros(())
    fv, fc = pred_vib.get("feats"), pred_clean.get("feats")
    if fv is not None and fc is not None:
        for a, b in zip(fv, fc):
            l_feat = l_feat + F.mse_loss(a, b.detach())
        l_feat = l_feat / max(1, len(fv))

    total = w_cls * l_cls + w_box * l_box + w_feat * l_feat
    logs = {"vib_cls": float(l_cls.detach()), "vib_box": float(l_box.detach()),
            "vib_feat": float(l_feat.detach()), "vib_total": float(total.detach())}
    return total, logs


# ============================================================
#  Ghi đè model.loss (không sửa thư viện)
# ============================================================
def install_vibcons(model, lam, warmup, epoch_fn, reg_max, **kw):
    """Bind lại phương thức loss của INSTANCE model. Giữ nguyên criterion gốc."""
    import types

    orig_init = model.init_criterion

    def loss(self, batch, preds=None):
        if getattr(self, "criterion", None) is None:
            self.criterion = orig_init()
        img = batch["img"]
        p_clean = self.forward(img) if preds is None else preds
        det_loss, det_items = self.criterion(p_clean, batch)      # loss detect: ảnh SẠCH

        # Consistency CHỈ tồn tại trong bước train. Bỏ qua khi:
        #  - đang validate: validator cũng gọi model.loss(batch, preds) nhưng preds là output
        #    dạng INFERENCE (tuple), không phải dict {boxes,scores,feats} của lúc train;
        #  - chưa hết warmup.
        # Đây cũng chính là lý do inference cost không đổi: nhánh rung không nằm trong graph eval.
        if (not self.training) or (not isinstance(p_clean, dict)) or epoch_fn() < warmup:
            return det_loss, det_items

        img_vib = corrupt_batch(img, severity=None).clamp_(0.0, 1.0)
        p_vib = self.forward(img_vib)
        cons, logs = vib_consistency_dict(p_vib, p_clean, reg_max, **kw)

        # In MỘT LẦN lúc nhánh rung lần đầu chạy thật. Không có dòng này trong log = term bị
        # guard bỏ qua âm thầm = cả lần train thành vô nghĩa. Phải grep được nó.
        if not getattr(self, "_vib_fired", False):
            self._vib_fired = True
            print(f"\n[VIB] >>> consistency ĐÃ KÍCH HOẠT ở epoch {epoch_fn()}: "
                  f"cls={logs['vib_cls']:.4f} box={logs['vib_box']:.4f} "
                  f"feat={logs['vib_feat']:.4f} total={logs['vib_total']:.4f} "
                  f"(cộng {lam * logs['vib_total'] * img.shape[0]:.3f} vào loss)", flush=True)

        # v8DetectionLoss trả loss * batch_size -> consistency phải cùng thang độ.
        total = det_loss.clone()
        total[1] = total[1] + lam * cons * img.shape[0]
        self._vib_logs = logs
        # loss_items giữ NGUYÊN bản gốc để cột box/cls/dfl trên bảng train vẫn đọc được
        return total, det_items

    model.loss = types.MethodType(loss, model)
    model._vib_logs = {}
    return model


class VibConsTrainer(DetectionTrainer):
    """DetectionTrainer + vibration. Mọi thứ khác giữ nguyên của ultralytics.

    mode="consistency" : 2 forward + ép nhất quán (loss detect trên ảnh SẠCH)
    mode="aug"         : 1 forward, rung ảnh TRƯỚC forward -> loss detect trên ảnh RUNG.
                         Trên SteadyDet, 'aug' THẮNG 'consistency' 36.2 vs 30.0 retain
                         (đo 2026-07-26), nên đây mới là phương pháp chính của bài.
    """

    vib_lam = 0.5
    vib_warmup = 3
    vib_kw = {}
    vib_mode = "consistency"
    vib_aug_prob = 0.5

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        if (self.vib_mode == "aug" and getattr(self, "epoch", 0) >= self.vib_warmup
                and random.random() < self.vib_aug_prob):
            batch["img"] = corrupt_batch(batch["img"], severity=None).clamp_(0.0, 1.0)
            if not getattr(self, "_aug_fired", False):
                self._aug_fired = True
                print(f"\n[VIB] >>> AUGMENTATION đã kích hoạt ở epoch {self.epoch} "
                      f"(p={self.vib_aug_prob}, KHÔNG có consistency)", flush=True)
        return batch

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights, verbose)
        if self.vib_mode == "none":
            print("[VIB] che do 'none': KHONG rung, KHONG consistency (baseline thuan)")
            return model
        if self.vib_mode == "aug":
            print(f"[VIB] chế độ 'aug': rung làm augmentation p={self.vib_aug_prob}, "
                  f"KHÔNG consistency | warmup={self.vib_warmup}")
            return model
        reg_max = model.model[-1].reg_max
        install_vibcons(model, self.vib_lam, self.vib_warmup,
                        lambda: getattr(self, "epoch", 0), reg_max, **self.vib_kw)
        print(f"[VIB] vibration-consistency BẬT trên {type(model).__name__} | "
              f"lambda={self.vib_lam} | warmup={self.vib_warmup} epoch | "
              f"reg_max={reg_max} | {self.vib_kw}")
        return model

    def progress_string(self):
        s = super().progress_string()
        v = getattr(self.model, "_vib_logs", {}) or {}
        if v:
            s += f"   [vib {v.get('vib_total', 0):.3f}]"
        return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/yolo/yolo11n/weights/best.pt",
                    help="finetune TỪ baseline đã train (giống vib finetune từ v17_light)")
    ap.add_argument("--data", default="configs/visdrone_person_ultralytics.yaml")
    ap.add_argument("--name", default="yolo11n_vibcons")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--lr0", type=float, default=0.002, help="1/5 lr thường, như bản pfdet")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--lambda-vib", type=float, default=0.5)
    ap.add_argument("--warmup-epochs", type=int, default=3)
    ap.add_argument("--conf-thr", type=float, default=0.1)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--w-feat", type=float, default=0.5)
    ap.add_argument("--mode", choices=["consistency", "aug", "none"], default="consistency",
                    help="aug = rung lam augmentation (phuong phap chinh sau 2026-07-26); "
                         "none = train binh thuong, khong rung (dung tao baseline)")
    ap.add_argument("--fraction", type=float, default=1.0,
                    help="dùng % dataset — chỉ để smoke-test (vd 0.03)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    proj = os.path.join(HERE, "runs", "yolo")        # PHẢI tuyệt đối (bài học PROJECT.md)
    VibConsTrainer.vib_mode = args.mode
    VibConsTrainer.vib_aug_prob = 0.5
    VibConsTrainer.vib_lam = args.lambda_vib
    VibConsTrainer.vib_warmup = args.warmup_epochs
    VibConsTrainer.vib_kw = dict(T=args.temperature, conf_thr=args.conf_thr,
                                 w_cls=1.0, w_box=1.0, w_feat=args.w_feat)

    last = os.path.join(proj, args.name, "weights", "last.pt")
    if args.resume and os.path.exists(last):
        print(f"[VIB] RESUME từ {last}")
        YOLO(last).train(trainer=VibConsTrainer, resume=True)
        return

    # ".yaml" -> dung kien truc, KHOI TAO NGAU NHIEN (from scratch, khong COCO-pretrain)
    w = args.weights
    if not w.endswith(".yaml") and not os.path.isabs(w):
        w = os.path.join(HERE, w)
    print(f"[VIB] khoi tao tu: {w}  ({'FROM SCRATCH' if w.endswith('.yaml') else 'pretrained'})")
    model = YOLO(w)
    model.train(
        trainer=VibConsTrainer,
        data=args.data, imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
        workers=args.workers, name=args.name, project=proj, exist_ok=True,
        optimizer="SGD", lr0=args.lr0, patience=0,          # patience=0 -> chạy đủ epoch
        val=True, plots=True, fraction=args.fraction,
    )
    print(f"[VIB] XONG -> {os.path.join(proj, args.name, 'weights', 'best.pt')}")
    print("[VIB] Bước tiếp: thêm ('yolo11n-vibcons', 'runs/yolo/%s/weights/best.pt') vào "
          "YOLO_MODELS trong paper_eval_unified.py rồi chạy lại để lấy retain%%." % args.name)


if __name__ == "__main__":
    main()
