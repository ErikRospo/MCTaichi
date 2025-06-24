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
        # Camera rotation keys
        self.key_i = False
        self.key_j = False
        self.key_k = False
        self.key_l = False

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
        elif key == dpg.mvKey_I:
            self.key_i = state
        elif key == dpg.mvKey_J:
            self.key_j = state
        elif key == dpg.mvKey_K:
            self.key_k = state
        elif key == dpg.mvKey_L:
            self.key_l = state
