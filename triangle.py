import taichi as ti

from taichi_types import vec3


@ti.dataclass
class Triangle:
    a: vec3
    b: vec3
    c: vec3
    color: vec3
