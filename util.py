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
def world_to_screen(point: vec3, camera_pos: vec3, camera_pitch: float, camera_yaw: float) -> vec2:
    # Camera transform
    rel = point - camera_pos
    rot = get_rotation_matrix(camera_pitch, camera_yaw)
    rel = rot @ rel

    # Perspective projection parameters
    fov = ti.math.radians(60.0)
    aspect = 1.0  # Assume square viewport; adjust if needed
    near = 0.1
    far = 100.0

    z = -rel.z
    if z <= near:
        z = near

    f = 1.0 / ti.tan(fov * 0.5)
    x_proj = rel.x * f / aspect / z
    y_proj = rel.y * f / z

    # Map from NDC [-1,1] to screen [0,1]
    screen_x = (x_proj + 1.0) * 0.5
    screen_y = (y_proj + 1.0) * 0.5
    return vec2([screen_x, screen_y])
