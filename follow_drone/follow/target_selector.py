"""Target selector — pick which tracked person to follow.

Strategies:
  - "largest": pick the largest bbox (closest person)
  - "center":  pick the bbox closest to image center
  - "manual":  caller specifies target_id

Once locked, prefers the same ID even if a larger person appears.
"""

import time


class TargetSelector:
    def __init__(self, strategy='largest', lock_on_first=True,
                 image_width=640, image_height=480, lost_grace_s=3.0,
                 reacquire_s=5.0, auto_relock=True):
        """
        lost_grace_s: mất mục tiêu bao lâu thì THÔI giữ khoá.

            ⚠️ Đừng buộc số này vào thang mất-mục-tiêu của safety — hai thứ canh hai
            chuyện khác nhau. Thang safety quyết định MÁY BAY LÀM GÌ khi mất người
            (bám dự đoán / treo / giữ vị trí). `lost_grace_s` quyết định BAO GIỜ ĐƯỢC
            ĐỔI SANG NGƯỜI KHÁC. Đặt nó bằng 7.0 cho "khớp thang" là sai: OC-SORT đã
            giữ ID sống thêm max_age (1.5 s) rồi, nên tổng thời gian chết thành
            1.5 + 7 = 8.5 giây — người đứng ngay trước ống kính mà vòng bám làm ngơ.

            Chọn 3.0 = đúng bằng pha `coast`, tức GIỮ KHOÁ CHỪNG NÀO MÁY BAY CÒN ĐANG
            ĐUỔI THEO dự đoán. Hết coast là máy bay bỏ cuộc, lúc đó cho phép chọn lại.

        reacquire_s: sau khi nhả khoá, dành thêm bấy nhiêu giây CHỈ để tìm lại đúng
            người cũ (theo vị trí + cỡ hộp cuối cùng), chưa chấp nhận người khác.

            Không có cửa sổ này thì hỏng: hết grace mà trong khung đang có người lạ
            là khoá luôn vào người lạ, và người cũ quay lại cũng không lấy lại được
            (vì lúc đó đã có khoá mới). Đúng lỗi mà test bắt được.

        auto_relock: sau khi nhả khoá, có tự chọn mục tiêu mới không.
            True  = tự khoá vào người có hộp to nhất (hành vi cũ). Tự chủ hơn, nhưng
                    nghĩa là mất người đang bám đủ lâu thì drone sẽ bám sang người
                    LẠ đứng gần nhất mà không hỏi ai.
            False = đứng yên chờ người vận hành bấm "đổi mục tiêu" trên bảng giám sát.
        """
        self.strategy = strategy
        self.lock_on_first = lock_on_first
        self.image_width = image_width
        self.image_height = image_height
        self.lost_grace_s = lost_grace_s
        self.reacquire_s = reacquire_s
        self.auto_relock = auto_relock

        self.locked_id = None
        self.last_seen_time = None
        self.awaiting_operator = False      # đã nhả khoá và đang chờ người chọn
        self.last_box = None                # hộp cuối cùng của mục tiêu đang khoá
        self.unlocked_at = None             # thời điểm nhả khoá, để tính cửa sổ tìm lại

    def select(self, tracks, manual_id=None, current_time=None):
        """Select target track from list of confirmed tracks.

        Returns track dict or None.
        """
        if current_time is None:
            current_time = time.time()

        # Manual override
        if manual_id is not None:
            for t in tracks:
                if t['id'] == manual_id:
                    self.locked_id = manual_id
                    self.last_seen_time = current_time
                    return t
            return None

        # If we have a locked ID, prefer it
        if self.locked_id is not None:
            for t in tracks:
                if t['id'] == self.locked_id:
                    self.last_seen_time = current_time
                    self.last_box = list(t['box'])
                    return t

            # Locked ID not in current tracks — check grace period
            if self.last_seen_time and (current_time - self.last_seen_time) > self.lost_grace_s:
                # Grace exceeded, unlock and re-select
                print(f"[TargetSelector] Lost target ID={self.locked_id} > {self.lost_grace_s}s, unlocking")
                self.locked_id = None
                self.last_seen_time = None
                self.unlocked_at = current_time
                if not self.auto_relock:
                    # KHÔNG tự vồ người khác. Đứng yên tới khi người vận hành quyết định.
                    self.awaiting_operator = True
                    print("[TargetSelector] auto_relock=False -> chờ người vận hành chọn mục tiêu")
            else:
                # Still in grace period, return None (target lost briefly)
                return None

        # No lock yet, or just unlocked: select using strategy
        if self.awaiting_operator:
            return None
        if not tracks:
            return None

        # Người rời khung rồi quay lại thì OC-SORT cấp ID MỚI (nó đếm tăng dần, không
        # tái dùng ID, và không có re-identification theo ngoại hình). Không xử lý gì
        # thì "chọn hộp to nhất" có thể vồ nhầm người lạ đứng gần hơn.
        #
        # Cách bù rẻ tiền: nhớ hộp cuối của mục tiêu cũ, và khi chọn lại thì ƯU TIÊN
        # track nào xuất hiện GẦN CHỖ CŨ VÀ CỠ NGANG CỠ CŨ. Đây là liên tục không gian,
        # KHÔNG phải nhận dạng lại người — nó đúng khi người quay lại chỗ cũ, và sai
        # khi hai người đổi chỗ cho nhau. Nút "ĐỔI MỤC TIÊU" trên bảng giám sát là
        # đường thoát khi nó chọn nhầm.
        best = None
        if self.last_box is not None and tracks:
            cand = self._match_last_box(tracks)
            if cand is not None:
                best = cand
                print(f"[TargetSelector] bám lại ID={best['id']} (gần vị trí/cỡ cũ)")

        # Chưa tìm thấy người cũ mà cửa sổ tìm lại chưa hết -> ĐỪNG vồ người khác.
        if best is None and self.last_box is not None and self.unlocked_at is not None:
            if (current_time - self.unlocked_at) < self.reacquire_s:
                return None
            # Hết cửa sổ: thôi tìm người cũ nữa, chấp nhận chọn theo chiến lược.
            print(f"[TargetSelector] hết {self.reacquire_s}s tìm lại người cũ -> chọn mới")
            self.last_box = None
            self.unlocked_at = None

        if best is None:
            if self.strategy == 'largest':
                best = max(tracks, key=lambda t: self._bbox_area(t['box']))
            elif self.strategy == 'center':
                best = min(tracks, key=lambda t: self._distance_to_center(t['box']))
            else:
                best = tracks[0]

        if self.lock_on_first:
            self.locked_id = best['id']
            self.last_seen_time = current_time
            self.last_box = list(best['box'])
            self.unlocked_at = None
            print(f"[TargetSelector] Locked target ID={best['id']}")

        return best

    def _match_last_box(self, tracks, max_center_frac=0.25, max_size_ratio=2.0):
        """Track nào khớp nhất với hộp cuối của mục tiêu cũ, hoặc None.

        max_center_frac: tâm lệch tối đa, tính theo tỉ lệ đường chéo khung hình.
        max_size_ratio : diện tích chênh tối đa bao nhiêu lần.
        Hai ngưỡng này cố ý CHẶT: thà không nhận ra còn hơn bám nhầm người.
        """
        lx1, ly1, lx2, ly2 = self.last_box
        lcx, lcy = (lx1 + lx2) / 2, (ly1 + ly2) / 2
        larea = max(1.0, (lx2 - lx1) * (ly2 - ly1))
        diag = (self.image_width ** 2 + self.image_height ** 2) ** 0.5
        best, best_d = None, None
        for t in tracks:
            x1, y1, x2, y2 = t['box']
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            d = ((cx - lcx) ** 2 + (cy - lcy) ** 2) ** 0.5
            if d > max_center_frac * diag:
                continue
            area = max(1.0, (x2 - x1) * (y2 - y1))
            r = max(area / larea, larea / area)
            if r > max_size_ratio:
                continue
            if best_d is None or d < best_d:
                best, best_d = t, d
        return best

    def unlock(self, wait_for_operator=False):
        """Bỏ khoá mục tiêu. wait_for_operator=True thì KHÔNG tự chọn lại."""
        self.locked_id = None
        self.last_seen_time = None
        self.last_box = None
        self.unlocked_at = None
        self.awaiting_operator = bool(wait_for_operator)

    def retarget(self):
        """Người vận hành yêu cầu chọn lại: bỏ khoá VÀ cho phép chọn ngay khung sau.

        Xoá luôn last_box: người vận hành bấm nút này chính vì muốn ĐỔI người, nên
        đừng để cơ chế bám-lại-theo-vị-trí kéo về đúng người vừa bị bỏ.
        """
        self.locked_id = None
        self.last_seen_time = None
        self.last_box = None
        self.unlocked_at = None
        self.awaiting_operator = False
        print("[TargetSelector] người vận hành yêu cầu đổi mục tiêu")

    def status(self):
        return {"locked_id": self.locked_id,
                "awaiting_operator": self.awaiting_operator,
                "strategy": self.strategy}

    @staticmethod
    def _bbox_area(box):
        x1, y1, x2, y2 = box
        return max(0, x2 - x1) * max(0, y2 - y1)

    def _distance_to_center(self, box):
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return ((cx - self.image_width/2)**2 + (cy - self.image_height/2)**2) ** 0.5
