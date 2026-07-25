#!/bin/bash
cd /home/nolan/Documents/pfdet_drone_follow
echo "============ $(date '+%F %T') ============"
echo "--- ĐANG TRAIN ---"
busy=0
pgrep -f "[t]rain_yolo_baselines_deploy" >/dev/null && { echo "  YOLO phase-2 (deploy, +COCO)"; busy=1; }
pgrep -f "[t]rain_yolo_baselines.py"      >/dev/null && { echo "  YOLO phase-1 (no-COCO)"; busy=1; }
pgrep -f "[t]rain_v3.py"                  >/dev/null && { echo "  pfdet $(pgrep -af '[t]rain_v3.py' | grep -oE 'v17[a-z_]*' | head -1)"; busy=1; }
[ "$busy" = 0 ] && echo "  (khong co training - dang eval / summarize / cho)"
echo
echo "--- EPOCH tung baseline (best/tong 150) ---"
for br in "PHASE-1:runs/yolo" "PHASE-2:runs/yolo_deploy"; do
  lbl=${br%%:*}; dir=${br##*:}
  printf "  [%s]\n" "$lbl"
  for m in yolov8n yolo11n yolo26n yolov10n rtdetr-l; do
    d="$dir/$m"
    if [ -f "$d/DEPLOY_DONE" ] || [ -f "$d/PAPER_DONE" ]; then st="XONG"
    elif [ -f "$d/results.csv" ]; then st="$(($(wc -l < "$d/results.csv")-1))/150 ep"
    else st="chua bat dau"; fi
    printf "    %-10s %s\n" "$m" "$st"
  done
done
echo
echo "--- MARKER (cot moc) ---"
for m in FINAL_ALL_DONE DEPLOY_ALL_DONE FIX_ALL_DONE; do
  [ -f "runs/paper_queue/$m" ] && echo "  [x] $m" || echo "  [ ] $m (chua)"
done
echo
echo "--- STAGE hien tai ---"
tail -1 runs/paper_queue/deploy_queue.log 2>/dev/null | sed 's/^/  deploy: /'
tail -1 runs/paper_queue/fix_queue.log 2>/dev/null | sed 's/^/  fix:    /'
