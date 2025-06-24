import taichi as ti
import numpy as np

from config import HEIGHT, WIDTH
from taichi_types import vec2, vec3
from triangle import Triangle
from util import get_rotation_matrix, world_to_screen

# Maximum number of triangles supported
MAX_TRIANGLES = 4096

img = ti.Vector.field(4, dtype=ti.f32, shape=(WIDTH, HEIGHT))
z_buffer = ti.field(dtype=ti.f32, shape=(WIDTH, HEIGHT))
# Store triangle vertex positions and colors in SoA fields for performance
tri_verts = ti.Vector.field(3, dtype=ti.f32, shape=(MAX_TRIANGLES, 3))  # [triangle, vertex]
tri_colors = ti.Vector.field(3, dtype=ti.f32, shape=MAX_TRIANGLES)
num_triangles = ti.field(dtype=ti.i32, shape=())


@ti.kernel
def set_triangles_np(verts: ti.types.ndarray(), colors: ti.types.ndarray(), n: ti.i32):
    num_triangles[None] = n
    for i in range(n):
        for j in range(3):
            for k in range(3):
                tri_verts[i, j][k] = verts[i, j, k]
        for k in range(3):
            tri_colors[i][k] = colors[i, k]


def set_triangles(verts_np: np.ndarray, colors_np: np.ndarray):
    # verts_np: [N, 3, 3], colors_np: [N, 3]
    n = verts_np.shape[0]
    set_triangles_np(verts_np, colors_np, n)


@ti.func
def barycentric(a, b, c, p):
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = v0.dot(v0)
    d01 = v0.dot(v1)
    d11 = v1.dot(v1)
    d20 = v2.dot(v0)
    d21 = v2.dot(v1)
    denom = d00 * d11 - d01 * d01
    u = (d11 * d20 - d01 * d21) / denom
    v = (d00 * d21 - d01 * d20) / denom
    w = 1.0 - u - v
    return u, v, w


@ti.kernel
def render(t: float, camera_pos: vec3, camera_pitch: float, camera_yaw: float):
    for i, j in img:
        img[i, j] = ti.Vector([0.0, 0.0, 0.0, 1.0])
        z_buffer[i, j] = 1e9

    for tri in range(num_triangles[None]):
        # Project triangle vertices to screen
        a = vec3([tri_verts[tri, 0][0], tri_verts[tri, 0][1], tri_verts[tri, 0][2]])
        b = vec3([tri_verts[tri, 1][0], tri_verts[tri, 1][1], tri_verts[tri, 1][2]])
        c = vec3([tri_verts[tri, 2][0], tri_verts[tri, 2][1], tri_verts[tri, 2][2]])
        color = tri_colors[tri]

        a2 = world_to_screen(a, camera_pos, camera_pitch, camera_yaw)
        b2 = world_to_screen(b, camera_pos, camera_pitch, camera_yaw)
        c2 = world_to_screen(c, camera_pos, camera_pitch, camera_yaw)

        # Get z in camera space (negative z axis points forward)
        rot = get_rotation_matrix(camera_pitch, camera_yaw)
        rel_a = rot @ (a - camera_pos)
        rel_b = rot @ (b - camera_pos)
        rel_c = rot @ (c - camera_pos)
        za = -rel_a.z
        zb = -rel_b.z
        zc = -rel_c.z

        # Compute bounding box in screen space
        min_x = ti.max(0, int(ti.min(a2.x, b2.x, c2.x) * WIDTH))
        max_x = ti.min(WIDTH - 1, int(ti.max(a2.x, b2.x, c2.x) * WIDTH))
        min_y = ti.max(0, int(ti.min(a2.y, b2.y, c2.y) * HEIGHT))
        max_y = ti.min(HEIGHT - 1, int(ti.max(a2.y, b2.y, c2.y) * HEIGHT))

        # Backface culling
        area = (b2.x - a2.x) * (c2.y - a2.y) - (b2.y - a2.y) * (c2.x - a2.x)
        if area < 0:
            for i in range(min_x, max_x + 1):
                for j in range(min_y, max_y + 1):
                    uv = vec2([i / WIDTH, j / HEIGHT])
                    # Barycentric coordinates
                    u, v, w = barycentric(a2, b2, c2, uv)
                    if (u >= 0) and (v >= 0) and (w >= 0):
                        z = u * za + v * zb + w * zc
                        if z < z_buffer[i, j]:
                            z_buffer[i, j] = z
                            img[i, j] = ti.Vector([color.x, color.y, color.z, 1.0])
