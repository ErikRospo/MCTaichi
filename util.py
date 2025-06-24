import taichi as ti

from taichi_types import vec2, vec3


@ti.func
def get_rotation_matrix(pitch: float, yaw: float):
    cy = ti.cos(yaw)
    sy = ti.sin(yaw)
    cp = ti.cos(pitch)
    sp = ti.sin(pitch)
    m00 = cy
    m01 = sy * sp
    m02 = sy * cp
    m10 = 0.0
    m11 = cp
    m12 = -sp
    m20 = -sy
    m21 = cy * sp
    m22 = cy * cp
    return ti.Matrix([[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]])


@ti.func
def world_to_screen(point: vec3) -> vec2:
    return vec2([point.x, point.y])
