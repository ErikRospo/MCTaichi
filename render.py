import numpy as np
import taichi as ti

from config import HEIGHT, WIDTH
from taichi_types import vec2, vec3
from triangle import Triangle
from util import world_to_screen

NUM_TRIANGLES = 3
img = ti.Vector.field(4, dtype=ti.f32, shape=(WIDTH, HEIGHT))
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


@ti.kernel
def render(t: float, camera_pos: ti.types.vector(3, ti.f32), camera_pitch: float, camera_yaw: float):
    for i, j in img:
        img[i, j] = ti.Vector([0.0, 0.0, 0.0, 1.0])
    for n in range(NUM_TRIANGLES):
        tri = triangles[n]
        if not is_triangle_front_facing(tri, camera_pos, camera_pitch, camera_yaw):
            continue
        min_x, max_x, min_y, max_y = get_triangle_bbox(tri, camera_pos, camera_pitch, camera_yaw)
        i0 = int(min_x * WIDTH)
        i1 = int(max_x * WIDTH) + 1
        j0 = int(min_y * HEIGHT)
        j1 = int(max_y * HEIGHT) + 1
        color = tri.color
        for i in range(i0, i1):
            for j in range(j0, j1):
                if 0 <= i < WIDTH and 0 <= j < HEIGHT:
                    uv = vec2([i / WIDTH, j / HEIGHT])
                    if pointInTriangle(tri, uv, camera_pos, camera_pitch, camera_yaw):
                        img[i, j] = ti.Vector([color.x, color.y, color.z, 1.0])
