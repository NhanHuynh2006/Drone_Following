"""
losses_v17_e2e.py — Dual-assignment loss cho v17 NMS-free (end-to-end).
=======================================================================

Ý tưởng (giống YOLOv10/YOLO26 consistent dual assignment), bản THỰC DỤNG tái dùng code đã test:
  - Head o2m (one-to-many): loss = PFDetLossV15 với k bình thường (k_tiny=6...) -> giám sát giàu.
  - Head o2o (one-to-one): loss = PFDetLossV15 với k=1 (top-1/GT) -> học dự đoán ĐƠN-ĐỈNH,
    ít box trùng -> inference KHÔNG cần NMS.
  total = loss_o2m + lambda_o2o * loss_o2o

Train: criterion((o2m_preds, o2o_preds), labels, epoch).
Deploy: chỉ dùng head o2o -> top-k theo score, bỏ NMS. (o2m chỉ tồn tại lúc train.)
"""


class DualLossV17:
    def __init__(self, o2m_criterion, o2o_criterion, lambda_o2o=1.0):
        """o2m/o2o_criterion: 2 instance PFDetLossV15 (o2o khởi tạo với k_tiny=1,k_normal=1,stal_k_min=1)."""
        self.o2m = o2m_criterion
        self.o2o = o2o_criterion
        self.lambda_o2o = float(lambda_o2o)

    # multiscale: train loop set criterion.img_size -> truyền cho cả hai
    @property
    def img_size(self):
        return self.o2m.img_size

    @img_size.setter
    def img_size(self, v):
        self.o2m.img_size = v
        self.o2o.img_size = v

    def __call__(self, dual_preds, labels, epoch=0):
        o2m_preds, o2o_preds = dual_preds
        loss_m, d_m = self.o2m(o2m_preds, labels, epoch=epoch)
        loss_o, d_o = self.o2o(o2o_preds, labels, epoch=epoch)
        loss = loss_m + self.lambda_o2o * loss_o

        d = dict(d_m)                                   # giữ obj/box/n_pos... của o2m cho log
        d['o2o'] = float(loss_o.detach())
        d['o2o_box'] = float(d_o.get('box', 0.0))
        d['total'] = float(loss.detach())
        return loss, d
