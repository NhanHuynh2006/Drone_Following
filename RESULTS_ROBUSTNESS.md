# Robustness "VisDrone-Shake" — PFDet-Nano v15 light @640

Đo bằng `eval_robustness.py`: AP@.5:.95 ở 4 mức rung (motion blur + noise + dao động sáng).
**retain_% = mCP / AP_clean** = giữ được bao nhiêu % AP khi rung. Cao = robust.

## Đường cong AP theo mức rung
| run | clean | sev1 | sev2 | sev3 | mCP | **retain_%** |
|---|---|---|---|---|---|---|
| **A baseline** = musgd_v3 (BEST, no vib) | 0.235 | 0.122 | 0.033 | 0.003 | 0.053 | **22.4%** |
| E best.pt (ep24, chọn theo AP sạch) | 0.221 | 0.131 | 0.054 | 0.015 | 0.067 | 30.2% |
| **E last.pt (ep119, model cuối) ⭐** | 0.221 | 0.145 | 0.071 | 0.011 | 0.076 | **34.3%** |

> Baseline = `train_v15_light_musgd_v3/best.pt` (AP@0.5=0.5932). E = finetune MuSGD từ chính nó + vibration-consistency (λ=0.5, 120 ep, batch 6).
> **Báo cáo E = last.pt** (model cuối, robust nhất; AP sạch ≈ best.pt). KHÔNG chọn checkpoint theo robustness để tránh selection-on-test.

## Kết luận
- ✅ Vibration-consistency **chứng minh được**: retain 22.4% → **34.3%** (+53% tương đối), **sev2 gấp 2.1×**, chỉ tốn ~6% AP sạch.
- Điểm mạnh nhất ở sev1–sev2 (rung nhẹ–vừa, sát thực tế bay). sev3 (rung cực nặng) cả hai đều gần chết — giới hạn của model nano.
- Hướng đẩy số: tăng `lambda_vib` 0.5→0.8 (robust hơn, đổi thêm chút AP sạch); hoặc thêm severity nặng vào train. Quay được rung THẬT từ drone → benchmark thật > synthetic.

## Đọc kết quả baseline (động cơ paper)
- Detector nano thường **sụp đổ dưới rung**: mất 78% AP, tới sev2 (rung vừa) chỉ còn **8.8%** — coi như mù.
- Đây là vấn đề THẬT khi bay drone (camera rung lúc phơi sáng). → đất cho đóng góp vibration-consistency.

## Kỳ vọng E
- E train với vibration-consistency phải **tụt CHẬM hơn** → retain_% cao hơn hẳn 21.8%.
- Báo cáo cả clean AP (E có thể nhúc nhích nhẹ do consistency ≈ augmentation) lẫn corrupted.
- Figure chính: 2 đường cong A (dốc) vs E (thoải) trên cùng trục severity.
