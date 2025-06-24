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
vec3 = ti.math.vec3



@ti.dataclass
class Triangle:
    a: vec3
    b: vec3
    c: vec3
    color: vec3

@ti.func
def pointInTriangle(triangle: Triangle, p: vec2):
    # Project triangle vertices to 2D
    a2 = world_to_screen(triangle.a)
    b2 = world_to_screen(triangle.b)
    c2 = world_to_screen(triangle.c)
    # Barycentric coordinate method in 2D
    v0 = c2 - a2
    v1 = b2 - a2
    v2 = p - a2

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

# Initialize triangles in normalized 3D coordinates with color
@ti.kernel
def init_triangles():
    for i in range(NUM_TRIANGLES):
        a = vec3([ti.random(), ti.random(), ti.random()])
        b = vec3([ti.random(), ti.random(), ti.random()])
        c = vec3([ti.random(), ti.random(), ti.random()])
        color = vec3([ti.random(), ti.random(), ti.random()])
        triangles[i] = Triangle(a=a, b=b, c=c, color=color)

init_triangles()


@ti.func
def get_triangle_bbox(triangle: Triangle):
    # Project triangle vertices to 2D
    a2 = world_to_screen(triangle.a)
    b2 = world_to_screen(triangle.b)
    c2 = world_to_screen(triangle.c)
    min_x = ti.min(a2.x, b2.x, c2.x)
    max_x = ti.max(a2.x, b2.x, c2.x)
    min_y = ti.min(a2.y, b2.y, c2.y)
    max_y = ti.max(a2.y, b2.y, c2.y)
    return min_x, max_x, min_y, max_y



@ti.func
def world_to_screen(point: vec3) -> vec2:

    return vec2([point.x, point.y])

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
        color = tri.color
        for i in range(i0, i1):
            for j in range(j0, j1):
                if 0 <= i < WIDTH and 0 <= j < HEIGHT:
                    uv = vec2([i / WIDTH, j / HEIGHT])
                    if pointInTriangle(tri, uv):
                        img[i, j] = ti.Vector([color.x, color.y, color.z, 1.0])


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
