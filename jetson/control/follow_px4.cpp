// follow_px4.cpp — Bộ điều khiển bay-theo-người MƯỢT cho PX4 (chạy trên Jetson, C++ MAVSDK).
// =============================================================================================
// Kiến trúc (grounded theo SOTA — xem DEPLOY_GUIDE.md):
//   [Detector Python/TensorRT] --UDP target--> [file này]
//        ├─ Kalman vận-tốc-không-đổi  (lọc nhiễu detection + DỰ ĐOÁN bù trễ + ước lượng tốc độ target)
//        ├─ PD + velocity feedforward (bám tốc độ target -> không trễ)
//        ├─ Giới hạn JERK/ACCEL đầu ra (mượt, không khựng)
//        └─ Gửi TrajectorySetpoint NED (pos+vel feedforward) @ 50 Hz -> PX4 jerk-limited smoothing
//
// Vì sao mượt (khác code cũ gửi velocity-only 10Hz):
//   (1) Kalman -> nguồn lệnh sạch + dự đoán (không giật khi detection nhấp nháy / mất frame)
//   (2) 50 Hz + slew/jerk limit -> không bước nhảy
//   (3) Position+velocity feedforward -> PX4 chạy bộ làm mượt nội bộ (MPC_JERK_*) thay vì đi thẳng attitude
//
// Build: xem CMakeLists.txt (cần MAVSDK v2). Test SITL trước khi bay thật (DEPLOY_GUIDE.md).

#include <mavsdk/mavsdk.h>
#include <mavsdk/plugins/action/action.h>
#include <mavsdk/plugins/offboard/offboard.h>
#include <mavsdk/plugins/telemetry/telemetry.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <mutex>
#include <thread>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

using namespace mavsdk;
using std::chrono::steady_clock;

// ───────────────────────── THAM SỐ (chỉnh ở đây hoặc DEPLOY_GUIDE.md) ─────────────────────────
struct Cfg {
    // kết nối
    std::string conn      = "serial:///dev/ttyTHS1:921600"; // SITL: "udpin://0.0.0.0:14540"
    int   udp_target_port = 5600;                            // cổng nhận target từ detector

    // camera
    double img_w = 640, img_h = 480;
    double fx = 700.0, fy = 700.0;                           // tiêu cự pixel (calibrate!)

    // mục tiêu bám
    double follow_distance_m = 5.0;                          // khoảng cách giữ
    double follow_altitude_m = 5.0;                          // độ cao AGL giữ

    // vòng điều khiển
    double ctrl_hz   = 50.0;                                 // tần số gửi setpoint (MƯỢT)
    double target_timeout_s = 1.5;                           // mất target quá lâu -> hover/RTL

    // PD gains (chuẩn hoá; tinh chỉnh trong DEPLOY_GUIDE.md)
    double kp_fwd = 0.6,  kd_fwd = 0.15;                     // khoảng cách -> vận tốc tiến
    double kp_yaw = 1.6,  kd_yaw = 0.25;                     // lệch ngang ảnh -> đổi hướng
    double kp_alt = 0.8,  kd_alt = 0.20;                     // độ cao -> vận tốc đứng

    // giới hạn (an toàn + mượt)
    double v_max_xy = 3.0, v_max_z = 1.5;                    // m/s
    double a_max    = 1.5;                                   // m/s^2  (slew vận tốc -> jerk-limit)
    double yaw_rate_max_deg = 60.0;                          // độ/s

    double vel_ff_gain = 0.8;                                // hệ số feedforward tốc độ target (0..1)
    double lookahead_s = 0.4;                                // carrot treo trước bao xa (0.3-0.6)
};
// Setpoint: LUÔN dùng kiểu MƯỢT NHẤT (position+velocity feedforward) khi position estimate TỐT;
// TỰ ĐỘNG fallback velocity-only khi estimate chưa sẵn (an toàn). User KHÔNG cần chỉnh gì.

// ───────────────────────── Kalman vận tốc không đổi (scalar) ─────────────────────────
// Lọc 1 tín hiệu (vd: range, image_cx) -> trả vị trí mượt + tốc độ + DỰ ĐOÁN tương lai (bù trễ).
struct CVKalman {
    double x = 0, v = 0;          // state: position, velocity
    double P00 = 1, P01 = 0, P10 = 0, P11 = 1;
    double q = 1.0;               // process noise (jerk) — lớn = bám nhanh, nhỏ = mượt
    double r = 4.0;               // measurement noise — lớn = tin model hơn detection
    bool inited = false;

    void reset() { inited = false; P00 = P11 = 1; P01 = P10 = 0; v = 0; }

    void predict(double dt) {
        // x = x + v dt ; P = F P F^T + Q
        x += v * dt;
        double dt2 = dt * dt, dt3 = dt2 * dt, dt4 = dt2 * dt2;
        double nP00 = P00 + dt * (P10 + P01) + dt2 * P11 + q * dt4 / 4.0;
        double nP01 = P01 + dt * P11 + q * dt3 / 2.0;
        double nP10 = P10 + dt * P11 + q * dt3 / 2.0;
        double nP11 = P11 + q * dt2;
        P00 = nP00; P01 = nP01; P10 = nP10; P11 = nP11;
    }
    void update(double z) {
        if (!inited) { x = z; v = 0; inited = true; return; }
        double y = z - x;                  // innovation
        double S = P00 + r;
        double K0 = P00 / S, K1 = P10 / S;
        x += K0 * y; v += K1 * y;
        double nP00 = (1 - K0) * P00, nP01 = (1 - K0) * P01;
        double nP10 = P10 - K1 * P00, nP11 = P11 - K1 * P01;
        P00 = nP00; P01 = nP01; P10 = nP10; P11 = nP11;
    }
    double predict_ahead(double dt) const { return x + v * dt; }  // bù trễ: vị trí sau dt giây
};

// ───────────────────────── Nhận target qua UDP (từ detector Python/C++) ─────────────────────────
// Gói tin (little-endian): 8 x float32 = [cx, cy, w, h, distance_m, conf, t_capture_unix, valid]
#pragma pack(push, 1)
struct TargetMsg { float cx, cy, w, h, distance_m, conf, t_capture, valid; };
#pragma pack(pop)

class TargetReceiver {
public:
    explicit TargetReceiver(int port) {
        sock_ = socket(AF_INET, SOCK_DGRAM, 0);
        int yes = 1; setsockopt(sock_, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
        sockaddr_in a{}; a.sin_family = AF_INET; a.sin_addr.s_addr = INADDR_ANY; a.sin_port = htons(port);
        bind(sock_, (sockaddr*)&a, sizeof(a));
        // non-blocking
        timeval tv{0, 1000}; setsockopt(sock_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        th_ = std::thread([this] { loop(); });
    }
    ~TargetReceiver() { run_ = false; if (th_.joinable()) th_.join(); if (sock_ >= 0) close(sock_); }

    bool get(TargetMsg& out, double& age_s) {
        std::lock_guard<std::mutex> lk(mtx_);
        if (!have_) return false;
        out = last_; age_s = std::chrono::duration<double>(steady_clock::now() - t_recv_).count();
        return true;
    }
private:
    void loop() {
        TargetMsg m;
        while (run_) {
            ssize_t n = recv(sock_, &m, sizeof(m), 0);
            if (n == (ssize_t)sizeof(m)) {
                std::lock_guard<std::mutex> lk(mtx_);
                last_ = m; t_recv_ = steady_clock::now(); have_ = true;
            }
        }
    }
    int sock_ = -1; std::thread th_; std::atomic<bool> run_{true};
    std::mutex mtx_; TargetMsg last_{}; bool have_ = false; steady_clock::time_point t_recv_;
};

static double clamp(double x, double lo, double hi) { return x < lo ? lo : (x > hi ? hi : x); }
static double slew(double cur, double tgt, double max_step) {
    double d = tgt - cur; d = clamp(d, -max_step, max_step); return cur + d;
}

int main(int argc, char** argv) {
    Cfg cfg;
    if (argc > 1) cfg.conn = argv[1];   // vd: ./follow_px4 "udpin://0.0.0.0:14540"  (SITL)

    // ── Kết nối PX4 ── (API MAVSDK v2; nếu bản khác báo lỗi ComponentType, đổi theo header bản bạn cài)
    Mavsdk mavsdk{Mavsdk::Configuration{ComponentType::CompanionComputer}};
    if (mavsdk.add_any_connection(cfg.conn) != ConnectionResult::Success) {
        std::cerr << "[ERR] không kết nối được: " << cfg.conn << "\n"; return 1;
    }
    std::cout << "[INFO] đang chờ autopilot...\n";
    auto sys_opt = mavsdk.first_autopilot(10.0);
    if (!sys_opt) { std::cerr << "[ERR] không thấy autopilot\n"; return 1; }
    auto system = sys_opt.value();

    Action    action{system};
    Offboard  offboard{system};
    Telemetry telemetry{system};
    telemetry.set_rate_position_velocity_ned(50.0);
    telemetry.set_rate_attitude_euler(50.0);

    TargetReceiver rx(cfg.udp_target_port);
    std::cout << "[INFO] nghe target UDP cổng " << cfg.udp_target_port << "\n";

    // ── Kalman cho 3 tín hiệu điều khiển ──
    CVKalman kf_range, kf_ex, kf_ey;     // range (m), lệch ngang ảnh (px), lệch dọc ảnh (px)
    kf_range.q = 0.5; kf_range.r = 1.0;  // range: mượt vừa
    kf_ex.q = 200.0;  kf_ex.r = 9.0;     // ảnh: nhiễu hơn -> r cao
    kf_ey.q = 200.0;  kf_ey.r = 9.0;

    // ── Vào OFFBOARD (phải gửi setpoint TRƯỚC khi start) ──
    offboard.set_velocity_ned({0.0f, 0.0f, 0.0f, 0.0f});
    if (offboard.start() != Offboard::Result::Success) {
        std::cerr << "[ERR] không vào được offboard (kiểm tra COM_RCL_EXCEPT, đã arm chưa)\n";
    }
    std::cout << "[INFO] OFFBOARD. Bắt đầu bám @ " << cfg.ctrl_hz << " Hz.\n";

    const double dt = 1.0 / cfg.ctrl_hz;
    double v_fwd = 0, v_lat = 0, v_up = 0;     // vận tốc body hiện tại (để slew/jerk-limit)
    auto next = steady_clock::now();

    while (true) {
        next += std::chrono::microseconds((long)(dt * 1e6));

        // trạng thái drone
        auto att = telemetry.attitude_euler();      // yaw_deg
        double yaw = att.yaw_deg * M_PI / 180.0;
        // độ cao AGL ~ -down (NED). Lý tưởng dùng distance sensor; tạm dùng z.
        auto pv = telemetry.position_velocity_ned();
        double alt = -pv.position.down_m;
        // TỰ ĐỘNG chọn kiểu setpoint mượt nhất: position+velocity nếu local position OK, else velocity.
        bool pos_ok = telemetry.health().is_local_position_ok;
        static int last_mode = -1;
        if ((int)pos_ok != last_mode) {
            std::cout << "[MODE] " << (pos_ok ? "position+velocity feedforward (MƯỢT NHẤT)"
                                              : "velocity-only (đang chờ position estimate)") << "\n";
            last_mode = (int)pos_ok;
        }

        TargetMsg t; double age = 0;
        bool have = rx.get(t, age) && t.valid > 0.5f && age < cfg.target_timeout_s;

        double vx_ned = 0, vy_ned = 0, vd_ned = 0, yaw_cmd_deg = att.yaw_deg;

        if (have) {
            // độ trễ từ lúc chụp -> giờ (bù bằng Kalman dự đoán)
            double lat = clamp(age, 0.0, 0.4);

            kf_range.predict(dt); kf_range.update(t.distance_m);
            double ex = t.cx - cfg.img_w * 0.5;        // >0: target bên phải
            double ey = t.cy - cfg.img_h * 0.5;        // >0: target phía dưới
            kf_ex.predict(dt); kf_ex.update(ex);
            kf_ey.predict(dt); kf_ey.update(ey);

            // giá trị DỰ ĐOÁN (bù trễ) -> bám đúng vị trí "bây giờ" của target
            double range_p = kf_range.predict_ahead(lat);
            double ex_p    = kf_ex.predict_ahead(lat);
            double ey_p    = kf_ey.predict_ahead(lat);

            // ── Luật điều khiển ──
            // 1) Forward: giữ khoảng cách. err = range - setpoint. + feedforward tốc độ target (kf_range.v)
            double err_fwd = range_p - cfg.follow_distance_m;
            double vf = cfg.kp_fwd * err_fwd + cfg.kd_fwd * kf_range.v
                        + cfg.vel_ff_gain * kf_range.v;     // FF: target lại gần/ra xa
            vf = clamp(vf, -cfg.v_max_xy, cfg.v_max_xy);

            // 2) Yaw: đưa target về giữa ảnh theo phương ngang (IBVS). góc = ex/fx
            double ang = ex_p / cfg.fx;                     // rad
            double dyaw = cfg.kp_yaw * ang + cfg.kd_yaw * (kf_ex.v / cfg.fx);
            double yaw_rate = clamp(dyaw, -cfg.yaw_rate_max_deg * M_PI / 180.0,
                                          cfg.yaw_rate_max_deg * M_PI / 180.0);
            yaw_cmd_deg = att.yaw_deg + yaw_rate * dt * 180.0 / M_PI;  // heading setpoint mượt

            // 3) Vertical: giữ độ cao. (hoặc dùng ey để giữ target giữa khung dọc)
            double err_alt = cfg.follow_altitude_m - alt;
            double vu = cfg.kp_alt * err_alt - cfg.kd_alt * pv.velocity.down_m * (-1.0);
            vu = clamp(vu, -cfg.v_max_z, cfg.v_max_z);

            // 4) Lateral body: 0 (yaw lo bám ngang) — giữ mượt. (có thể thêm nếu muốn cua nhanh)
            double vlat = 0.0;

            // ── Jerk/accel limit (slew) -> chống khựng ──
            double a_step = cfg.a_max * dt;
            v_fwd = slew(v_fwd, vf,   a_step);
            v_lat = slew(v_lat, vlat, a_step);
            v_up  = slew(v_up,  vu,   a_step);

            // body -> NED (chỉ xoay theo yaw; bỏ qua pitch/roll nhỏ)
            vx_ned = v_fwd * std::cos(yaw) - v_lat * std::sin(yaw);
            vy_ned = v_fwd * std::sin(yaw) + v_lat * std::cos(yaw);
            vd_ned = -v_up;   // up -> NED down âm
        } else {
            // mất target: slew về 0 (hover mượt), giữ heading
            double a_step = cfg.a_max * dt;
            v_fwd = slew(v_fwd, 0, a_step); v_lat = slew(v_lat, 0, a_step); v_up = slew(v_up, 0, a_step);
            vx_ned = v_fwd * std::cos(yaw); vy_ned = v_fwd * std::sin(yaw); vd_ned = -v_up;
            kf_range.reset(); kf_ex.reset(); kf_ey.reset();
        }

        // ── Gửi setpoint xuống PX4 ──
        vx_ned = clamp(vx_ned, -cfg.v_max_xy, cfg.v_max_xy);
        vy_ned = clamp(vy_ned, -cfg.v_max_xy, cfg.v_max_xy);
        vd_ned = clamp(vd_ned, -cfg.v_max_z,  cfg.v_max_z);
        Offboard::VelocityNedYaw vsp{(float)vx_ned, (float)vy_ned, (float)vd_ned, (float)yaw_cmd_deg};

        if (pos_ok) {
            // MƯỢT NHẤT (mặc định khi position estimate tốt): vị trí "carrot" + vận tốc feedforward.
            // -> PX4 chạy vòng position + jerk-smoothing nội bộ, feedforward giữ đáp ứng nhanh.
            Offboard::PositionNedYaw psp{};
            psp.north_m = pv.position.north_m + (float)(vx_ned * cfg.lookahead_s);
            psp.east_m  = pv.position.east_m  + (float)(vy_ned * cfg.lookahead_s);
            psp.down_m  = pv.position.down_m  + (float)(vd_ned * cfg.lookahead_s);
            psp.yaw_deg = (float)yaw_cmd_deg;
            offboard.set_position_velocity_ned(psp, vsp);
        } else {
            // TỰ ĐỘNG fallback velocity-only khi position estimate chưa sẵn (an toàn, vẫn mượt nhờ Kalman+jerk-limit).
            offboard.set_velocity_ned(vsp);
        }

        std::this_thread::sleep_until(next);
    }
    return 0;
}
