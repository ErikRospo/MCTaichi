import taichi as ti

@ti.func
def fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)

@ti.func
def lerp(a, b, t):
    return a + t * (b - a)

@ti.func
def grad(hash, x, y, z):
    h = hash & 15
    u = x if h < 8 else y
    v = y if h < 4 else (x if h in (12, 14) else z)
    return ((u if (h & 1) == 0 else -u) +
            (v if (h & 2) == 0 else -v))

@ti.data_oriented
class PerlinNoise3D:
    def __init__(self, seed=0):
        self.perm = ti.field(dtype=ti.i32, shape=512)
        self.seed = seed
        self.init_permutation()

    @ti.kernel
    def init_permutation(self):
        for i in range(256):
            self.perm[i] = i
        # Simple shuffle (not cryptographically secure)
        for i in range(256):
            j = (i * 31 + self.seed) % 256
            tmp = self.perm[i]
            self.perm[i] = self.perm[j]
            self.perm[j] = tmp
        for i in range(256):
            self.perm[256 + i] = self.perm[i]

    @ti.func
    def noise(self, x, y, z):
        X = int(ti.floor(x)) & 255
        Y = int(ti.floor(y)) & 255
        Z = int(ti.floor(z)) & 255
        xf = x - ti.floor(x)
        yf = y - ti.floor(y)
        zf = z - ti.floor(z)
        u = fade(xf)
        v = fade(yf)
        w = fade(zf)

        aaa = self.perm[self.perm[self.perm[X] + Y] + Z]
        aba = self.perm[self.perm[self.perm[X] + Y + 1] + Z]
        aab = self.perm[self.perm[self.perm[X] + Y] + Z + 1]
        abb = self.perm[self.perm[self.perm[X] + Y + 1] + Z + 1]
        baa = self.perm[self.perm[self.perm[X + 1] + Y] + Z]
        bba = self.perm[self.perm[self.perm[X + 1] + Y + 1] + Z]
        bab = self.perm[self.perm[self.perm[X + 1] + Y] + Z + 1]
        bbb = self.perm[self.perm[self.perm[X + 1] + Y + 1] + Z + 1]

        x1 = lerp(grad(aaa, xf, yf, zf), grad(baa, xf - 1, yf, zf), u)
        x2 = lerp(grad(aba, xf, yf - 1, zf), grad(bba, xf - 1, yf - 1, zf), u)
        y1 = lerp(x1, x2, v)

        x3 = lerp(grad(aab, xf, yf, zf - 1), grad(bab, xf - 1, yf, zf - 1), u)
        x4 = lerp(grad(abb, xf, yf - 1, zf - 1), grad(bbb, xf - 1, yf - 1, zf - 1), u)
        y2 = lerp(x3, x4, v)

        return (lerp(y1, y2, w) + 1) / 2  # Normalize to [0, 1]

