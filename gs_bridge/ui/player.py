"""pygame player window: frame view + pause/continue/restore buttons + WASD controls.

Layout:
    ┌──────────────────────────────────────┬────────────┐
    │                                      │  [⏸ 暂停]  │
    │        main view 480x832             │  [▶ 继续]  │
    │   PLAYING: HY stream frames          │  [⟲ 复原]  │
    │   PAUSED : asset render @ user cam   │            │
    │                                      │  status    │
    │  WASD move · arrows look (same keys  │  frame ctr │
    │  in both states -- HY's operation    │            │
    │  language)                           │            │
    └──────────────────────────────────────┴────────────┘

State machine:
    PLAYING -> (pause)  -> PAUSED    (main view switches to asset render)
    PAUSED  -> (continue, cam moved) -> PLAYING (hard-cut regen via viewpoint_reset)
    PAUSED  -> (continue, cam still) -> PLAYING (resume original rollout, no reset)
    PAUSED  -> (restore) -> PAUSED    (camera slerps back to the pause anchor)

The window process talks to hy_driver over ZMQ:
    SUB frames topic "frame"  (b"frame" + uint32 count + HxWx3 uint8)
    PUB control topic: dict {cmd: pause|continue|restore|quit, ...}
    SUB status topic "status" (driver state for display)

Asset observation rendering during PAUSED goes through the driver too (it owns
the AssetClient connection; the UI process has no HY/MoVieS imports): the driver
republishes ATLAS renders on the "frame" topic as `obs` frames. The UI only
sends the current observation-camera pose periodically (cmd "obs_pose") while
paused; the driver throttles renders.
"""

import sys
import time

import numpy as np
import pygame
import zmq

from gs_bridge.ui.camera import FirstPersonCamera
from gs_bridge.ui.keys import key_to_logical

# window layout (matches HY 480x832 + 220px control column)
VIEW_W, VIEW_H = 832, 480
SIDEBAR_W = 220
PANEL_H = 56
WIN_W, WIN_H = VIEW_W + SIDEBAR_W, VIEW_H
BG = (18, 18, 22)
FG = (220, 220, 225)
ACCENT = (86, 156, 214)

# key repeat: hold-to-move cadence (steps per second while key held)
STEP_INTERVAL = 0.25  # 4 steps/s -- matches watching a 4-frame latent at 24fps x16
# atlas observation render request cadence
OBS_INTERVAL = 0.1


class PlayerWindow:
    def __init__(self, zmq_ctx: zmq.Context, frame_sub_endpoint: str,
                 ctrl_pub_endpoint: str, status_sub_endpoint: str):
        pygame.init()
        pygame.display.set_caption("gs_bridge · HY-WorldPlay player")
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("dejavusans", 16)
        self.font_small = pygame.font.SysFont("dejavusans", 13)
        self.font_cn = pygame.font.SysFont("dejavusans, notosanscjk, wqyzenhei", 15)

        # ZMQ wiring
        self.frame_sub = zmq_ctx.socket(zmq.SUB)
        self.frame_sub.setsockopt(zmq.SUBSCRIBE, b"frame")
        self.frame_sub.setsockopt(zmq.CONFLATE, 1)  # only latest frame matters
        self.frame_sub.connect(frame_sub_endpoint)
        self.ctrl_pub = zmq_ctx.socket(zmq.PUB)
        self.ctrl_pub.connect(ctrl_pub_endpoint)
        self.status_sub = zmq_ctx.socket(zmq.SUB)
        self.status_sub.setsockopt(zmq.SUBSCRIBE, b"status")
        self.status_sub.setsockopt(zmq.CONFLATE, 1)
        self.status_sub.connect(status_sub_endpoint)

        # state
        self.state = "PLAYING"            # PLAYING | PAUSED
        self.frame_count = 0
        self.current_frame: np.ndarray | None = None  # (H, W, 3) uint8 RGB
        self.last_frame_is_obs = False
        self.hold_keys: set[str] = set()  # currently held logical actions
        self._last_step_t = 0.0
        self._last_obs_req_t = 0.0
        self.camera = FirstPersonCamera(np.eye(4))
        self.status_text = "starting"
        self.pending_motions = 0  # queued motion steps (sent to driver)
        self._quit = False

        # restore animation state
        self._restore_anim = None  # generator or None

    # ------------------------------------------------------------------
    # UI element geometry
    # ------------------------------------------------------------------

    def _buttons(self):
        labels = [("⏸ 暂停", "pause"), ("▶ 继续", "continue"), ("⟲ 复原", "restore")]
        y = 20
        out = []
        for text, cmd in labels:
            rect = pygame.Rect(WIN_W - SIDEBAR_W + 16, y, SIDEBAR_W - 32, PANEL_H - 12)
            out.append((rect, text, cmd))
            y += PANEL_H
        return out

    def _button_enabled(self, cmd: str) -> bool:
        if cmd == "pause":
            return self.state == "PLAYING"
        if cmd in ("continue", "restore"):
            return self.state == "PAUSED"
        return False

    # ------------------------------------------------------------------
    # event handling
    # ------------------------------------------------------------------

    def handle_events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self._quit = True
            elif ev.type == pygame.KEYDOWN:
                name = pygame.key.name(ev.key)
                if name == "escape":
                    self._quit = True
                if name == "space":
                    # space = pause/continue hotkey
                    self._send_cmd("pause" if self.state == "PLAYING" else "continue")
                logical = key_to_logical(name)
                if logical:
                    self.hold_keys.add(logical)
                    self._do_step(logical)  # immediate first step
            elif ev.type == pygame.KEYUP:
                name = pygame.key.name(ev.key)
                logical = key_to_logical(name)
                if logical:
                    self.hold_keys.discard(logical)
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                pos = ev.pos
                for rect, text, cmd in self._buttons():
                    if rect.collidepoint(pos) and self._button_enabled(cmd):
                        self._on_button(cmd)

    def _on_button(self, cmd: str):
        if cmd == "pause":
            self._send_cmd("pause")
        elif cmd == "continue":
            self._send_cmd("continue", camera=self.camera.snapshot().tolist())
        elif cmd == "restore":
            self._start_restore()

    def _do_step(self, logical: str):
        """One motion step of the currently active camera semantics."""
        if self.state == "PLAYING":
            # record into the driver's pending trajectory queue
            self._send_cmd("motion", logical=logical)
            self.pending_motions += 1
        else:
            self.camera.step(logical)

    # ------------------------------------------------------------------
    # restore animation (slerp back to the pause anchor, via obs renders)
    # ------------------------------------------------------------------

    def _start_restore(self):
        if self.camera.moved_from_anchor():
            self._restore_anim = self.camera.slerp_to_anchor(n_steps=20)

    # ------------------------------------------------------------------
    # ZMQ plumbing
    # ------------------------------------------------------------------

    def _send_cmd(self, cmd: str, **fields):
        import json
        payload = {"cmd": cmd, **fields}
        self.ctrl_pub.send_string("ctrl " + json.dumps(payload))

    def poll_frames(self):
        """Non-blocking drain of the frame topic (CONFLATE keeps only the latest)."""
        try:
            while True:
                topic = self.frame_sub.recv(zmq.NOBLOCK)
                if topic[:5] == b"frame":
                    body = self.frame_sub.recv(zmq.NOBLOCK)
                    self._decode_frame(body)
        except zmq.Again:
            pass
        try:
            while True:
                topic = self.status_sub.recv(zmq.NOBLOCK)
                body = self.status_sub.recv(zmq.NOBLOCK)
                self.status_text = body.decode("utf-8", "replace")
        except zmq.Again:
            pass

    def _decode_frame(self, body: bytes):
        # b"frame" topic already stripped by SUB prefix; body = flag(1) + count(4) + pixels
        is_obs = body[0] == 1
        count = int.from_bytes(body[1:5], "little")
        pixels = np.frombuffer(body[5:], dtype=np.uint8)
        expected = VIEW_H * VIEW_W * 3
        if pixels.size < expected:
            return  # torn frame -- skip
        img = pixels[:expected].reshape(VIEW_H, VIEW_W, 3)
        # wire is RGB
        self.current_frame = img
        self.last_frame_is_obs = is_obs
        if not is_obs:
            self.frame_count = count

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------

    def draw(self):
        self.screen.fill(BG)
        # main view
        if self.current_frame is not None:
            surf = pygame.surfarray.make_surface(
                np.transpose(self.current_frame, (1, 0, 2))  # (W,H,3) for pygame
            )
            self.screen.blit(surf, (0, 0))
        else:
            txt = self.font.render("waiting for frames...", True, FG)
            self.screen.blit(txt, (VIEW_W // 2 - 100, VIEW_H // 2))

        # sidebar
        sb_x = WIN_W - SIDEBAR_W
        pygame.draw.rect(self.screen, (28, 28, 34), (sb_x, 0, SIDEBAR_W, WIN_H))
        pygame.draw.line(self.screen, (60, 60, 70), (sb_x, 0), (sb_x, WIN_H), 2)

        for rect, text, cmd in self._buttons():
            enabled = self._button_enabled(cmd)
            color = ACCENT if enabled else (70, 70, 80)
            pygame.draw.rect(self.screen, color, rect, border_radius=8)
            label = self.font.render(text, True, (255, 255, 255) if enabled else (150, 150, 155))
            self.screen.blit(label, label.get_rect(center=rect.center))

        # status panel
        y = 3 * PANEL_H + 40
        state_txt = self.font.render(
            f"state: {'生成中' if self.state == 'PLAYING' else '已暂停'}", True, FG)
        self.screen.blit(state_txt, (sb_x + 16, y)); y += 26
        v = self.font_small.render(f"view: {'asset 观察' if self.last_frame_is_obs else 'HY 生成流'}", True, FG)
        self.screen.blit(v, (sb_x + 16, y)); y += 22
        f = self.font_small.render(f"frames: {self.frame_count}", True, FG)
        self.screen.blit(f, (sb_x + 16, y)); y += 22
        q = self.font_small.render(f"queued steps: {self.pending_motions}", True, FG)
        self.screen.blit(q, (sb_x + 16, y)); y += 22
        mv = "moved" if self.camera.moved_from_anchor() else "at anchor"
        m = self.font_small.render(f"camera: {mv} ({self.camera.moved_steps})", True, FG)
        self.screen.blit(m, (sb_x + 16, y)); y += 30
        s = self.font_small.render(f"driver: {self.status_text[:26]}", True, (160, 160, 170))
        self.screen.blit(s, (sb_x + 16, y))

        # help footer
        help_txt = self.font_small.render(
            "WASD move · ←↑↓→ look · space pause/continue · esc quit", True, (150, 150, 160))
        self.screen.blit(help_txt, (16, WIN_H - 26))

        pygame.display.flip()

    def run(self):
        while not self._quit:
            self.handle_events()
            self.poll_frames()

            now = time.perf_counter()
            # held-key repeat
            if self.hold_keys and now - self._last_step_t >= STEP_INTERVAL:
                for logical in list(self.hold_keys)[:1]:  # one motion per tick (HY is 1-axis per step)
                    self._do_step(logical)
                    self._last_step_t = now
                    break

            # paused: stream observation pose to the driver for ATLAS rendering
            if self.state == "PAUSED" and self._restore_anim is None:
                if now - self._last_obs_req_t >= OBS_INTERVAL:
                    self._send_cmd("obs_pose", camera=self.camera.snapshot().tolist())
                    self._last_obs_req_t = now
            # restore animation: drive camera along the slerp, still requesting renders
            elif self._restore_anim is not None:
                try:
                    next(self._restore_anim)
                    self._send_cmd("obs_pose", camera=self.camera.snapshot().tolist())
                except StopIteration:
                    self._restore_anim = None
                    self.pending_motions = 0

            self.draw()
            self.clock.tick(30)

        self._send_cmd("quit")
        pygame.quit()


def main():
    # endpoints (same constants as driver.py / run_ui.py)
    FRAME_EP = "ipc:///tmp/gs_bridge_frames"
    CTRL_EP = "ipc:///tmp/gs_bridge_ctrl"
    STATUS_EP = "ipc:///tmp/gs_bridge_status"
    ctx = zmq.Context()
    win = PlayerWindow(ctx, FRAME_EP, CTRL_EP, STATUS_EP)
    win.run()


if __name__ == "__main__":
    main()
