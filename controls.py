from typing import Literal, assert_never
import dearpygui.dearpygui as dpg


class Controls:
    def __init__(self):
        self.lmb_down = False
        self.rmb_down = False
        self.mmb_down = False
        # Keyboard state
        self.key_w = False
        self.key_a = False
        self.key_s = False
        self.key_d = False
        self.key_e = False
        self.key_q = False
        # Mouse tracking
        self.mouse_x = 0.0
        self.mouse_y = 0.0
        self.mouse_dx = 0.0
        self.mouse_dy = 0.0
        self._last_mouse_x = None
        self._last_mouse_y = None

    def set_mb(self, button: Literal[0, 1, 2], state=True):
        if button == 0:
            self.lmb_down = state
        elif button == 1:
            self.rmb_down = state
        elif button == 2:
            self.mmb_down = state
        else:
            assert_never(button)

    def set_key(self, key, state: bool):
        if key == dpg.mvKey_W:
            self.key_w = state
        elif key == dpg.mvKey_A:
            self.key_a = state
        elif key == dpg.mvKey_S:
            self.key_s = state
        elif key == dpg.mvKey_D:
            self.key_d = state
        elif key == dpg.mvKey_E:
            self.key_e = state
        elif key == dpg.mvKey_Q:
            self.key_q = state

    def update_mouse(self, x: float, y: float):
        if self._last_mouse_x is not None and self._last_mouse_y is not None:
            self.mouse_dx = x - self._last_mouse_x
            self.mouse_dy = y - self._last_mouse_y
        else:
            self.mouse_dx = 0.0
            self.mouse_dy = 0.0
        self.mouse_x = x
        self.mouse_y = y
        self._last_mouse_x = x
        self._last_mouse_y = y

    def reset_mouse_delta(self):
        self.mouse_dx = 0.0
        self.mouse_dy = 0.0
