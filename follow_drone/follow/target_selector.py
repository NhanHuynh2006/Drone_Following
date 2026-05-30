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
                 image_width=640, image_height=480, lost_grace_s=5.0):
        self.strategy = strategy
        self.lock_on_first = lock_on_first
        self.image_width = image_width
        self.image_height = image_height
        self.lost_grace_s = lost_grace_s

        self.locked_id = None
        self.last_seen_time = None

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
                    return t

            # Locked ID not in current tracks — check grace period
            if self.last_seen_time and (current_time - self.last_seen_time) > self.lost_grace_s:
                # Grace exceeded, unlock and re-select
                print(f"[TargetSelector] Lost target ID={self.locked_id} > {self.lost_grace_s}s, unlocking")
                self.locked_id = None
                self.last_seen_time = None
            else:
                # Still in grace period, return None (target lost briefly)
                return None

        # No lock yet, or just unlocked: select using strategy
        if not tracks:
            return None

        if self.strategy == 'largest':
            best = max(tracks, key=lambda t: self._bbox_area(t['box']))
        elif self.strategy == 'center':
            best = min(tracks, key=lambda t: self._distance_to_center(t['box']))
        else:
            best = tracks[0]

        if self.lock_on_first:
            self.locked_id = best['id']
            self.last_seen_time = current_time
            print(f"[TargetSelector] Locked target ID={best['id']}")

        return best

    def unlock(self):
        self.locked_id = None
        self.last_seen_time = None

    @staticmethod
    def _bbox_area(box):
        x1, y1, x2, y2 = box
        return max(0, x2 - x1) * max(0, y2 - y1)

    def _distance_to_center(self, box):
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return ((cx - self.image_width/2)**2 + (cy - self.image_height/2)**2) ** 0.5
