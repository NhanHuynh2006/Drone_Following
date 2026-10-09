#!/usr/bin/env bash
# run.sh — chay moi thu bang python CUA VENV, khoi phai nho activate.
#   ./run.sh check                              checklist truoc khi ra bai
#   ./run.sh record --alt 5 --scene grass ...   thu data chuyen bay
#   ./run.sh demo                               xem model detect (can man hinh)
#   ./run.sh mux                                CHIA cong MAVLink cho nhieu chuong trinh
#   ./run.sh gcs                                TRAM MAT DAT: video + telemetry ArduPilot (trinh duyet)
#   ./run.sh stream                             phat hinh + box len TRINH DUYET (khong can man hinh)
#   ./run.sh bench                              do latency 10 phut cho paper
#   ./run.sh py <file.py> ...                   chay file python bat ky trong venv
#   ./run.sh pip install <goi>                  cai them goi VAO VENV
set -euo pipefail
cd "$(dirname "$0")"
PY="./venv/bin/python3"
[ -x "$PY" ] || { echo "Chua co venv. Chay truoc:  bash setup_pi.sh"; exit 1; }
cmd="${1:-}"; shift || true
case "$cmd" in
  check)  exec bash check_pi.sh "$@" ;;
  record) exec "$PY" flight_record.py "$@" ;;
  mux)    exec "$PY" mavmux.py "$@" ;;
  gcs)    exec "$PY" hailo_gcs.py "$@" ;;
  stream) exec "$PY" hailo_stream_web.py "$@" ;;
  demo)   exec "$PY" hailo_demo_cam.py --hef models/pfdet_v17_640_hailo8l.hef --img-size 640 "$@" ;;
  bench)  exec "$PY" hailo_bench_latency.py --hef models/pfdet_v17_640_hailo8l.hef --img-size 640 "$@" ;;
  py)     exec "$PY" "$@" ;;
  pip)    exec "$PY" -m pip "$@" ;;
  *) sed -n '2,10p' "$0"; exit 1 ;;
esac
