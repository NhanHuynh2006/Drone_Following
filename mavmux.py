"""
mavmux.py — CHIA MỘT CỔNG MAVLINK CHO NHIỀU CHƯƠNG TRÌNH.
==========================================================

VẤN ĐỀ: `/dev/ttyACM0` (Pixhawk qua USB) chỉ **một** tiến trình mở được. Mở tiến
trình thứ hai là pyserial ném:

    device reports readiness to read but returned no data
    (device disconnected or multiple access on port?)

Nhưng lúc bay cần ÍT NHẤT hai thứ cùng nói chuyện với mạch bay:
  - `follow_drone/main.py`  — gửi lệnh vận tốc (GUIDED)
  - `hailo_gcs.py`          — đọc telemetry cho dashboard
và thỉnh thoảng thêm `mav_health.py` hoặc Mission Planner qua WiFi.

GIẢI: một tiến trình duy nhất giữ cổng serial, rồi phát lại nguyên byte ra nhiều
đầu UDP. Không phân tích gói, không sửa gì — chỉ chuyển tiếp hai chiều, nên không
thêm độ trễ đáng kể và không thể làm hỏng nội dung gói.

CHẠY:
    ./run.sh mux                                     # mặc định: 14550 + 14551
    ./run.sh mux --serial /dev/ttyACM0 --baud 115200 \\
                 --out 127.0.0.1:14550 --out 127.0.0.1:14551 \\
                 --out 192.168.196.50:14552          # thêm máy tính chạy Mission Planner

RỒI:
    ./run.sh gcs --mavlink udpin:127.0.0.1:14551     # dashboard
    python3 follow_drone/main.py  (trỏ tới udpin:127.0.0.1:14550)

⚠️ ĐỘ ƯU TIÊN: mux chuyển tiếp MỌI thứ nhận từ UDP xuống mạch bay, kể cả lệnh.
Chỉ trỏ vào đây những chương trình bạn tin. Dashboard `hailo_gcs.py` chỉ đọc nên
an toàn; nhưng nếu mở cổng ra ngoài mạng thì bất kỳ ai trong mạng cũng gửi lệnh
được — khi bay chỉ nên để 127.0.0.1 và địa chỉ máy trạm của bạn.
"""
import argparse
import select
import socket
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--out", action="append", default=[],
                    help="host:port nhận bản sao. Lặp lại cờ này cho nhiều đầu.")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    outs = a.out or ["127.0.0.1:14550", "127.0.0.1:14551"]

    import serial
    while True:
        try:
            ser = serial.Serial(a.serial, a.baud, timeout=0)
            break
        except Exception as e:
            print(f"[mux] chưa mở được {a.serial}: {e} — thử lại sau 2s")
            time.sleep(2.0)
    print(f"[mux] giữ {a.serial} @ {a.baud}")

    socks = []
    for spec in outs:
        host, _, port = spec.rpartition(":")
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.connect((host, int(port)))        # connect -> gửi/nhận được cả hai chiều
        s.setblocking(False)
        socks.append((s, spec))
        print(f"[mux]   -> udp {spec}   (phía kia dùng  udpin:{spec})")

    n_dn = n_up = 0
    t_log = time.time()
    print("[mux] đang chạy. Ctrl-C để dừng.")
    try:
        while True:
            r, _, _ = select.select([ser.fileno()] + [s.fileno() for s, _ in socks],
                                    [], [], 0.5)
            if ser.fileno() in r:
                data = ser.read(4096)
                if data:
                    n_dn += len(data)
                    for s, _ in socks:
                        try:
                            s.send(data)
                        except OSError:
                            pass            # chưa ai bind cổng đó — bỏ qua, không chết
            for s, spec in socks:
                if s.fileno() in r:
                    try:
                        data = s.recv(4096)
                    except OSError:
                        continue
                    if data:
                        n_up += len(data)
                        ser.write(data)     # chuyển tiếp lệnh xuống mạch bay

            if not a.quiet and time.time() - t_log > 10.0:
                dt = time.time() - t_log
                print(f"[mux] xuống {n_dn/dt/1024:6.1f} KB/s   lên {n_up/dt/1024:5.2f} KB/s")
                n_dn = n_up = 0
                t_log = time.time()
    except KeyboardInterrupt:
        print("\n[mux] dừng.")
    finally:
        ser.close()
        for s, _ in socks:
            s.close()


if __name__ == "__main__":
    main()
