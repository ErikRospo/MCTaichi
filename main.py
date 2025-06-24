import time

import dearpygui.dearpygui as dpg
import numpy as np
import taichi as ti

from config import HEIGHT, WIDTH
from controls import Controls

ti.init(arch=ti.gpu)

from render import img, init_triangles, render, triangles

np_img = np.zeros((WIDTH * HEIGHT * 4,), dtype=np.float32)
controls = Controls()

# Camera state
camera_pos = np.array([0.0, 0.0, 3.0], dtype=np.float32)
camera_pitch = 0.0
camera_yaw = 0.0


def handle_mouse_down(sender, app_data):
    controls.set_mb(app_data[0], True)


def handle_mouse_up(sender, app_data):
    controls.set_mb(app_data, False)


def handle_key_down(sender, app_data):
    controls.set_key(app_data[0], True)


def handle_key_up(sender, app_data):
    controls.set_key(app_data, False)

def handle_mouse_move(sender, app_data):
    x, y = dpg.get_mouse_pos()
    controls.update_mouse(x, y)


dpg.create_context()
dpg.create_viewport(title="MCTaichi", width=WIDTH + 20, height=HEIGHT + 80)
dpg.setup_dearpygui()

with dpg.handler_registry():
    dpg.add_mouse_down_handler(callback=handle_mouse_down)
    dpg.add_mouse_release_handler(callback=handle_mouse_up)
    dpg.add_key_down_handler(callback=handle_key_down)
    dpg.add_key_release_handler(callback=handle_key_up)
    dpg.add_mouse_move_handler(callback=handle_mouse_move)
with dpg.texture_registry():
    texture_id = dpg.add_raw_texture(WIDTH, HEIGHT, np_img)

with dpg.window(tag="mainwindow"):
    dpg.add_image(texture_id)

dpg.show_viewport()
dpg.set_primary_window("mainwindow", True)
start_time = time.time()

init_triangles()

MOVE_SPEED = 2.0
LOOK_SENSITIVITY = 0.002

while dpg.is_dearpygui_running():
    current_time = time.time() - start_time

    # Camera movement
    forward = np.array([
        np.sin(camera_yaw) * np.cos(camera_pitch),
        np.sin(camera_pitch),
        -np.cos(camera_yaw) * np.cos(camera_pitch)
    ], dtype=np.float32)
    right = np.array([
        np.cos(camera_yaw),
        0.0,
        np.sin(camera_yaw)
    ], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    dt = 1.0 / 60.0  # Assume fixed timestep for simplicity

    move = np.zeros(3, dtype=np.float32)
    if controls.key_w:
        move += forward
    if controls.key_s:
        move -= forward
    if controls.key_a:
        move -= right
    if controls.key_d:
        move += right
    if controls.key_e:
        move += up
    if controls.key_q:
        move -= up
    if np.linalg.norm(move) > 0:
        move = move / np.linalg.norm(move)
    camera_pos += move * MOVE_SPEED * dt

    # Camera rotation
    camera_yaw += controls.mouse_dx * LOOK_SENSITIVITY
    camera_pitch += -controls.mouse_dy * LOOK_SENSITIVITY
    camera_pitch = np.clip(camera_pitch, -np.pi/2 + 0.01, np.pi/2 - 0.01)
    controls.reset_mouse_delta()

    render(current_time, camera_pos, camera_pitch, camera_yaw)
    img_np = img.to_numpy()
    np_img[:] = np.rot90(img_np).reshape(-1)
    dpg.set_value(texture_id, np_img)
    dpg.render_dearpygui_frame()

dpg.destroy_context()
