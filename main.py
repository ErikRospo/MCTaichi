import dearpygui.dearpygui as dpg
import taichi as ti
import numpy as np
import time

ti.init(arch=ti.gpu)

# Image dimensions
WIDTH = 1080
HEIGHT = 720
# Taichi field for RGBA image
img = ti.Vector.field(4, dtype=ti.f32, shape=(WIDTH, HEIGHT))

# Time variable to be passed to the render kernel
time_val = ti.field(dtype=ti.f32, shape=())

@ti.kernel
def render(t: ti.f32):
    for i, j in img:
        # Blue channel is dynamic: (sin(t) + 1) / 2
        blue = (ti.sin(t) + 1.0) * 0.5
        img[i, j] = ti.Vector([i / WIDTH, j / HEIGHT, blue, 1.0])  # simple gradient

# Numpy array for DearPyGui texture (flattened)
np_img = np.zeros((WIDTH * HEIGHT * 4,), dtype=np.float32)

dpg.create_context()
dpg.create_viewport(title='Taichi + DPG Image', width=WIDTH+20, height=HEIGHT+80)
dpg.setup_dearpygui()

with dpg.texture_registry():
    texture_id = dpg.add_raw_texture(WIDTH, HEIGHT, np_img)

with dpg.window(tag="Primary Window"):
    dpg.add_image(texture_id)

dpg.show_viewport()
dpg.set_primary_window("Primary Window", True)
start_time = time.time()

while dpg.is_dearpygui_running():
    current_time = time.time() - start_time
    render(current_time)
    img_np = img.to_numpy()
    np_img[:] = np.rot90(img_np).reshape(-1)
    dpg.set_value(texture_id, np_img)
    dpg.render_dearpygui_frame()

dpg.destroy_context()