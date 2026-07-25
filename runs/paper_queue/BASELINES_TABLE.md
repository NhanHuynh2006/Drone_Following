# BASELINES — params / GFLOPs / mAP (tính lại chuẩn)

## PHASE-1 (no-COCO): train VisDrone-only, val VisDrone-val 548

| model | params (M) | GFLOPs@640 | mAP50 | mAP50-95 | epoch đã train |
|---|---|---|---|---|---|
| rtdetr-l | 32.81 | 108.0 | 0.6856 | 0.3340 | 150 |
| yolo11n | 2.59 | 6.4 | 0.6212 | 0.2857 | 134 |
| yolo26n | 2.50 | 5.8 | 0.6087 | 0.2762 | 126 |
| yolov10n | 2.71 | 8.4 | 0.6228 | 0.2820 | 150 |
| yolov8n | 3.01 | 8.2 | 0.6158 | 0.2787 | 150 |

## PHASE-2 (deploy): train VisDrone+COCO 9125, val VisDrone-val 548

| model | params (M) | GFLOPs@640 | mAP50 | mAP50-95 | epoch đã train |
|---|---|---|---|---|---|
| rtdetr-l | 32.81 | 108.0 | 0.6930 | 0.3377 | 150 |
| yolo11n | 2.59 | 6.4 | 0.6098 | 0.2725 | 139 |
| yolo26n | 2.50 | 5.8 | 0.5979 | 0.2677 | 150 |
| yolov10n | 2.71 | 8.4 | 0.6002 | 0.2660 | 150 |
| yolov8n | 3.01 | 8.2 | 0.6114 | 0.2711 | 150 |

