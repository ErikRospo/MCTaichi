
class Controls:
    def __init__(self):
        self.lmb_down = False
        self.rmb_down = False
        self.mmb_down = False

    def set_mb(self, button:int,state=True):
        if button == 0:
            self.lmb_down = state
        elif button == 1:
            self.rmb_down = state
        elif button == 2:
            self.mmb_down = state
