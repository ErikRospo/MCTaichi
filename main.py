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


dpg.create_context()
dpg.create_viewport(title="MCTaichi", width=WIDTH + 20, height=HEIGHT + 80)

dpg.setup_dearpygui()

with dpg.handler_registry():
    dpg.add_mouse_down_handler(callback=handle_mouse_down)
    dpg.add_mouse_release_handler(callback=handle_mouse_up)
    dpg.add_key_down_handler(callback=handle_key_down)
    dpg.add_key_release_handler(callback=handle_key_up)
with dpg.texture_registry():
    texture_id = dpg.add_raw_texture(WIDTH, HEIGHT, np_img)

with dpg.window(tag="mainwindow"):
    dpg.add_image(texture_id)

dpg.show_viewport()
dpg.set_primary_window("mainwindow", True)
start_time = time.time()

init_triangles()

MOVE_SPEED = 2.0
LOOK_SENSITIVITY = 0.03
current_time=0.
while dpg.is_dearpygui_running():
    last_current_time = current_time
    current_time = time.time() - start_time

    # Calculate actual dt
    dt = current_time - last_current_time

    # Camera rotation using IJKL
    if controls.key_j:
        camera_yaw += LOOK_SENSITIVITY
    if controls.key_l:
        camera_yaw -= LOOK_SENSITIVITY
    if controls.key_i:
        camera_pitch += LOOK_SENSITIVITY
    if controls.key_k:
        camera_pitch -= LOOK_SENSITIVITY
    camera_pitch = np.clip(camera_pitch, -np.pi/2 + 0.01, np.pi/2 - 0.01)

    # Compute forward, right, and up vectors from camera orientation
    forward = np.array([
        np.cos(camera_yaw) * np.cos(camera_pitch),
        np.sin(camera_pitch),
        np.sin(camera_yaw) * np.cos(camera_pitch)
    ], dtype=np.float32)
    right = np.array([
        np.sin(camera_yaw - np.pi/2),
        0.0,
        np.cos(camera_yaw - np.pi/2)
    ], dtype=np.float32)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    move = np.zeros(3, dtype=np.float32)
    if controls.key_w:
        move += forward
    if controls.key_s:
        move -= forward
    if controls.key_a:
        move -= right
    if controls.key_d:
        move += right
    if controls.key_r:
        move += up
    if controls.key_f:
        move -= up
    if np.linalg.norm(move) > 0:
        move = move / np.linalg.norm(move)
    camera_pos += move * MOVE_SPEED * dt

    render(current_time, camera_pos, camera_pitch, camera_yaw)
    img_np = img.to_numpy()
    np_img[:] = np.rot90(img_np).reshape(-1)
    dpg.set_value(texture_id, np_img)
    dpg.render_dearpygui_frame()

dpg.destroy_context()
