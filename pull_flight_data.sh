#!/usr/bin/env bash
# pull_flight_data.sh — CHAY TREN LAPTOP. Keo data chuyen bay tu Pi ve.
#   bash pull_flight_data.sh                 # keo ve data/flight_raw/
#   bash pull_flight_data.sh --delete-remote # keo ve xong thi XOA tren Pi (giai phong the)
set -euo pipefail
# pi5.local qua mDNS thay vi IP cung: router cap DHCP nen IP doi (114 -> 115
# ngay 2026-08-28). Ghi de bang:  PI_HOST=raspi@<ip> bash <script>
PI="${PI_HOST:-raspi@pi5.local}"
DST="${DST:-data/flight_raw}"
mkdir -p "$DST"
echo "[pull] $PI:~/NhanHuynh/flight_data/  ->  $DST/"
rsync -avh --progress --partial "$PI:~/NhanHuynh/flight_data/" "$DST/"
echo
echo "[pull] xong. Cac session keo ve:"
ls -1 "$DST" | sed 's/^/  /'
echo
echo "[pull] tong quan tung session:"
for m in "$DST"/*/manifest.json; do
  [ -f "$m" ] || continue
  python3 - "$m" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
f = d.get("flight", {})
e = d.get("e2e_ms") or {}
print(f"  {d['session']}")
print(f"     alt={f.get('alt_m')}m scene={f.get('scene')} persons={f.get('persons')} "
      f"mode={f.get('mode')} vib={f.get('vibration')} shutter={f.get('shutter')}")
print(f"     {d.get('duration_s','?')}s | {d.get('saved_images','?')} anh | "
      f"{d.get('detected_frames','?')} frame detect | e2e p50={e.get('p50','-')} p99={e.get('p99','-')}ms "
      f"| throttled={d.get('throttled_end','?')}")
PY
done
if [ "${1:-}" = "--delete-remote" ]; then
  echo; read -rp "XOA flight_data tren Pi? (go 'xoa' de xac nhan): " a
  [ "$a" = "xoa" ] && ssh "$PI" 'rm -rf ~/NhanHuynh/flight_data/*' && echo "[pull] da xoa tren Pi." || echo "[pull] bo qua."
fi
