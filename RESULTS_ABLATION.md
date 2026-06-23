# Kết quả — PFDet-Nano v15 light @640, VisDrone val (548 ảnh)

> **Cấu trúc paper (chốt):** đóng góp CHÍNH = **vibration-consistency** (kháng rung/nhoè khi bay).
> density head + crowd-loc (từ run C) + KD = các dòng **ablation phụ**. Đo bằng 2 thước:
> (1) bảng AP sạch dưới đây; (2) **đường cong robustness** (`eval_robustness.py`, file `RESULTS_ROBUSTNESS.md`).

## Robustness — bảng chính của paper (clean → rung)
| run | mô tả | AP clean | mCP (rung tb) | retain_% |
|---|---|---|---|---|
| A | baseline | ? | ? | ? |
| E ⭐ | + vibration-consistency (`train_v15_vib_E`) | ? | ? | ? |

→ chạy `eval_robustness.py` cho A và E; kỳ vọng E có retain_% **cao hơn** (tụt chậm hơn khi severity tăng).

---

## Bảng AP sạch (ablation phụ) — VisDrone val 548 ảnh

Đo bằng `run_eval_coco.py` (pycocotools chuẩn). EMA weights, conf_thr=0.01.

## Bảng chính (overall)
| # | run | density | crowd α | KD | AP@.5:.95 | AP75 | AP_s | AR_s |
|---|---|---|---|---|---|---|---|---|
| A | baseline = **musgd_v3** (BEST, AP@0.5 0.5932) | ✗ | 0 | ✗ | **0.235** | **0.131** | 0.232 | 0.359 |
| B | density-only (`train_v15_dgs_B`) | ✓ | 0 | ✗ | ? | ? | ? | ? |
| C | DGS đầy đủ (`train_v15_dgs_C`) | ✓ | 1.0 | ✗ | ? | ? | ? | ? |
| D | + KD (`train_v15_dgs_D`) | ✓ | 1.0 | ✓ | ? | ? | ? | ? |

## Phân tầng theo mật độ (AP / AR_s)
| nhóm | A baseline | B | C | D |
|---|---|---|---|---|
| 0–10 người (n=255) | AP 0.238 / AR_s 0.401 | | | |
| 10–30 người (n=230) | AP 0.243 / AR_s 0.383 | | | |
| **30+ người (n=63)** | **AP 0.188 / AR_s 0.285** | | | |

## Kỳ vọng (BEST_PLAN mục 5)
- **B vs A:** AR_s nhóm 30+ tăng (>0.285) — density head vá recall cảnh đông.
- **C vs B:** AP75 & AP nhóm 30+ tăng — crowd-loc chữa localization đúng chỗ sụp.
- **D vs C:** AP overall tăng tiếp nhờ KD.

## Ghi chú
- Baseline plan ghi 0.235; đo lại bằng eval_coco mới ra **0.220** overall → dùng 0.220 làm mốc chính thức.
- Nút thắt xác nhận: nhóm 30+ người AP 0.188 (thấp nhất), AR_s 0.285 (thấp nhất), AP75 0.105.
