import time

import dearpygui.dearpygui as dpg
import numpy as np
import taichi as ti

from config import HEIGHT, WIDTH
from controls import Controls

ti.init(arch=ti.gpu)

# Taichi field for RGBA image
img = ti.Vector.field(4, dtype=ti.f32, shape=(WIDTH, HEIGHT))

# Time variable to be passed to the render kernel
time_val = ti.field(dtype=ti.f32, shape=())
vec2 = ti.math.vec2


@ti.dataclass
class Triangle:
    a: vec2
    b: vec2
    c: vec2


@ti.func
def pointInTriangle(triangle: Triangle, p: vec2):
    # Barycentric coordinate method
    v0 = triangle.c - triangle.a
    v1 = triangle.b - triangle.a
    v2 = p - triangle.a

    dot00 = v0.dot(v0)
    dot01 = v0.dot(v1)
    dot02 = v0.dot(v2)
    dot11 = v1.dot(v1)
    dot12 = v1.dot(v2)

    denom = dot00 * dot11 - dot01 * dot01
    inside = False
    if denom != 0:
        u = (dot11 * dot02 - dot01 * dot12) / denom
        v = (dot00 * dot12 - dot01 * dot02) / denom
        inside = (u >= 0) and (v >= 0) and (u + v <= 1)
    return inside


# Number of triangles
NUM_TRIANGLES = 3

# Taichi field for triangles
triangles = Triangle.field(shape=NUM_TRIANGLES)


# Initialize triangles in normalized coordinates
@ti.kernel
def init_triangles():
    triangles[0] = Triangle(a=vec2([0.2, 0.2]), b=vec2([0.8, 0.2]), c=vec2([0.5, 0.8]))
    triangles[1] = Triangle(a=vec2([0.3, 0.3]), b=vec2([0.4, 0.7]), c=vec2([0.7, 0.5]))
    triangles[2] = Triangle(a=vec2([0.6, 0.1]), b=vec2([0.9, 0.2]), c=vec2([0.8, 0.6]))


init_triangles()


@ti.func
def get_triangle_bbox(triangle: Triangle):
    min_x = ti.min(triangle.a.x, triangle.b.x, triangle.c.x)
    max_x = ti.max(triangle.a.x, triangle.b.x, triangle.c.x)
    min_y = ti.min(triangle.a.y, triangle.b.y, triangle.c.y)
    max_y = ti.max(triangle.a.y, triangle.b.y, triangle.c.y)
    return min_x, max_x, min_y, max_y


@ti.kernel
def render(t: float):
    # Clear image
    for i, j in img:
        img[i, j] = ti.Vector([0.0, 0.0, 0.0, 1.0])
    # Render each triangle in its bounding box
    for n in range(NUM_TRIANGLES):
        tri = triangles[n]
        min_x, max_x, min_y, max_y = get_triangle_bbox(tri)
        # Convert normalized bbox to pixel range
        i0 = int(min_x * WIDTH)
        i1 = int(max_x * WIDTH) + 1
        j0 = int(min_y * HEIGHT)
        j1 = int(max_y * HEIGHT) + 1
        for i in range(i0, i1):
            for j in range(j0, j1):
                if 0 <= i < WIDTH and 0 <= j < HEIGHT:
                    uv = vec2([i / WIDTH, j / HEIGHT])
                    if pointInTriangle(tri, uv):
                        blue = (ti.sin(t + n) + 1.0) * 0.5
                        img[i, j] = ti.Vector([uv.x, uv.y, blue, 1.0])


# Numpy array for DearPyGui texture (flattened)
np_img = np.zeros((WIDTH * HEIGHT * 4,), dtype=np.float32)


controls = Controls()


def handle_mouse_down(sender, app_data):
    controls.set_mb(app_data[0], True)


def handle_mouse_up(sender, app_data):
    controls.set_mb(app_data, False)


dpg.create_context()
dpg.create_viewport(title="MCTaichi", width=WIDTH + 20, height=HEIGHT + 80)
dpg.setup_dearpygui()

with dpg.handler_registry():
    dpg.add_mouse_down_handler(callback=handle_mouse_down)
    dpg.add_mouse_release_handler(callback=handle_mouse_up)
with dpg.texture_registry():
    texture_id = dpg.add_raw_texture(WIDTH, HEIGHT, np_img)

with dpg.window(tag="mainwindow"):
    dpg.add_image(texture_id)

dpg.show_viewport()
dpg.set_primary_window("mainwindow", True)
start_time = time.time()

while dpg.is_dearpygui_running():
    current_time = time.time() - start_time
    render(current_time)
    img_np = img.to_numpy()
    np_img[:] = np.rot90(img_np).reshape(-1)
    dpg.set_value(texture_id, np_img)
    dpg.render_dearpygui_frame()

dpg.destroy_context()
