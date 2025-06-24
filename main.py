import time

import dearpygui.dearpygui as dpg
import numpy as np
import taichi as ti

from config import HEIGHT, WIDTH
from controls import Controls

ti.init(arch=ti.gpu)

from render import img, render, set_triangles

np_img = np.zeros((WIDTH * HEIGHT * 4,), dtype=np.float32)
controls = Controls()

# Camera state
camera_pos = np.array([0.0, 0.0, 3.0], dtype=np.float32)
camera_pitch = 0.0
camera_yaw = 0.0

# Generate triangles (example: random triangles in front of camera)
N_TRIANGLES = 2000


def random_triangles(n):
    verts = np.random.uniform(-1, 1, size=(n, 3, 3)).astype(np.float32)
    verts[..., 2] += 2.5  # Move triangles in front of camera
    colors = np.random.uniform(0.2, 1.0, size=(n, 3)).astype(np.float32)
    return verts, colors


verts_np, colors_np = random_triangles(N_TRIANGLES)
set_triangles(verts_np, colors_np)

dpg.create_context()
dpg.create_viewport(title="MCTaichi", width=WIDTH + 20, height=HEIGHT + 80)

dpg.setup_dearpygui()

with dpg.texture_registry():
    texture_id = dpg.add_raw_texture(WIDTH, HEIGHT, np_img)

with dpg.window(tag="mainwindow"):
    dpg.add_image(texture_id)

dpg.show_viewport()
dpg.set_primary_window("mainwindow", True)
start_time = time.time()

MOVE_SPEED = 2.0
LOOK_SENSITIVITY = 0.03
current_time = 0.0
while dpg.is_dearpygui_running():
    last_current_time = current_time
    current_time = time.time() - start_time

    # Calculate actual dt
    dt = current_time - last_current_time

    # Poll keyboard state directly
    # Camera rotation using IJKL
    if dpg.is_key_down(dpg.mvKey_J):
        camera_yaw += LOOK_SENSITIVITY
    if dpg.is_key_down(dpg.mvKey_L):
        camera_yaw -= LOOK_SENSITIVITY
    if dpg.is_key_down(dpg.mvKey_I):
        camera_pitch += LOOK_SENSITIVITY
    if dpg.is_key_down(dpg.mvKey_K):
        camera_pitch -= LOOK_SENSITIVITY
    camera_pitch = np.clip(camera_pitch, -np.pi / 2 + 0.01, np.pi / 2 - 0.01)

    # Compute forward, right, and up vectors from camera orientation
    forward = np.array(
        [
            np.cos(camera_yaw) * np.cos(camera_pitch),
            np.sin(camera_pitch),
            np.sin(camera_yaw) * np.cos(camera_pitch),
        ],
        dtype=np.float32,
    )
    right = np.array(
        [np.sin(camera_yaw - np.pi / 2), 0.0, np.cos(camera_yaw - np.pi / 2)],
        dtype=np.float32,
    )
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    move = np.zeros(3, dtype=np.float32)
    if dpg.is_key_down(dpg.mvKey_W):
        move += forward
    if dpg.is_key_down(dpg.mvKey_S):
        move -= forward
    if dpg.is_key_down(dpg.mvKey_A):
        move -= right
    if dpg.is_key_down(dpg.mvKey_D):
        move += right
    if dpg.is_key_down(dpg.mvKey_R):
        move += up
    if dpg.is_key_down(dpg.mvKey_F):
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
