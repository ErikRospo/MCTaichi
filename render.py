import numpy as np
import taichi as ti

from config import HEIGHT, WIDTH
from taichi_types import vec2, vec3
from triangle import Triangle
from util import get_rotation_matrix, world_to_screen

NUM_TRIANGLES = 3
img = ti.Vector.field(4, dtype=ti.f32, shape=(WIDTH, HEIGHT))
z_buffer = ti.field(dtype=ti.f32, shape=(WIDTH, HEIGHT))  # Add z-buffer
triangles = Triangle.field(shape=NUM_TRIANGLES)


def init_triangles():
    @ti.kernel
    def _init():
        for i in range(NUM_TRIANGLES):
            a = vec3([ti.random(), ti.random(), ti.random()])
            b = vec3([ti.random(), ti.random(), ti.random()])
            c = vec3([ti.random(), ti.random(), ti.random()])
            camera_pos = vec3([0.0, 0.0, 0.0])
            camera_pitch = 0.0
            camera_yaw = 0.0
            a2 = world_to_screen(a, camera_pos, camera_pitch, camera_yaw)
            b2 = world_to_screen(b, camera_pos, camera_pitch, camera_yaw)
            c2 = world_to_screen(c, camera_pos, camera_pitch, camera_yaw)
            area = (b2.x - a2.x) * (c2.y - a2.y) - (b2.y - a2.y) * (c2.x - a2.x)
            if area > 0:
                tmp = b
                b = c
                c = tmp
            color = vec3([ti.random(), ti.random(), ti.random()])
            triangles[i] = Triangle(a=a, b=b, c=c, color=color)

    _init()


@ti.func
def pointInTriangle(triangle, p, camera_pos, camera_pitch, camera_yaw):
    a2 = world_to_screen(triangle.a, camera_pos, camera_pitch, camera_yaw)
    b2 = world_to_screen(triangle.b, camera_pos, camera_pitch, camera_yaw)
    c2 = world_to_screen(triangle.c, camera_pos, camera_pitch, camera_yaw)
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


@ti.func
def get_triangle_bbox(triangle, camera_pos, camera_pitch, camera_yaw):
    a2 = world_to_screen(triangle.a, camera_pos, camera_pitch, camera_yaw)
    b2 = world_to_screen(triangle.b, camera_pos, camera_pitch, camera_yaw)
    c2 = world_to_screen(triangle.c, camera_pos, camera_pitch, camera_yaw)
    min_x = ti.min(a2.x, b2.x, c2.x)
    max_x = ti.max(a2.x, b2.x, c2.x)
    min_y = ti.min(a2.y, b2.y, c2.y)
    max_y = ti.max(a2.y, b2.y, c2.y)
    return min_x, max_x, min_y, max_y


@ti.func
def is_triangle_front_facing(triangle, camera_pos, camera_pitch, camera_yaw):
    a2 = world_to_screen(triangle.a, camera_pos, camera_pitch, camera_yaw)
    b2 = world_to_screen(triangle.b, camera_pos, camera_pitch, camera_yaw)
    c2 = world_to_screen(triangle.c, camera_pos, camera_pitch, camera_yaw)
    area = (b2.x - a2.x) * (c2.y - a2.y) - (b2.y - a2.y) * (c2.x - a2.x)
    return area < 0


@ti.func
def barycentric_coords(a, b, c, p):
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
        z_buffer[i, j] = 1e9  # Clear z-buffer to a large value

    for n in range(NUM_TRIANGLES):
        tri = triangles[n]
        min_x, max_x, min_y, max_y = get_triangle_bbox(tri, camera_pos, camera_pitch, camera_yaw)
        i0 = int(min_x * WIDTH)
        i1 = int(max_x * WIDTH) + 1
        j0 = int(min_y * HEIGHT)
        j1 = int(max_y * HEIGHT) + 1
        color = tri.color

        # Project triangle vertices to screen and get their z in camera space
        a2 = world_to_screen(tri.a, camera_pos, camera_pitch, camera_yaw)
        b2 = world_to_screen(tri.b, camera_pos, camera_pitch, camera_yaw)
        c2 = world_to_screen(tri.c, camera_pos, camera_pitch, camera_yaw)
        # Get z in camera space (negative z axis points forward)
        rel_a = get_rotation_matrix(camera_pitch, camera_yaw) @ (tri.a - camera_pos)
        rel_b = get_rotation_matrix(camera_pitch, camera_yaw) @ (tri.b - camera_pos)
        rel_c = get_rotation_matrix(camera_pitch, camera_yaw) @ (tri.c - camera_pos)
        za = -rel_a.z
        zb = -rel_b.z
        zc = -rel_c.z

        for i in range(i0, i1):
            for j in range(j0, j1):
                if 0 <= i < WIDTH and 0 <= j < HEIGHT:
                    uv = vec2([i / WIDTH, j / HEIGHT])
                    if pointInTriangle(tri, uv, camera_pos, camera_pitch, camera_yaw):
                        # Interpolate z using barycentric coordinates
                        u, v, w = barycentric_coords(a2, b2, c2, uv)
                        z = u * za + v * zb + w * zc
                        if z < z_buffer[i, j]:
                            z_buffer[i, j] = z
                            img[i, j] = ti.Vector([color.x, color.y, color.z, 1.0])
