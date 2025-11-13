import numpy as np

def _rng(seed):
    return np.random.default_rng(seed)


def img_solid_color(size=(256,256), color=(128,128,128), seed=None):
    h, w = size
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:] = color
    return arr


def img_horizontal_gradient(size=(256,256), start=0, end=177, seed=None):
    h, w = size
    grad = np.linspace(start, end, w, dtype=np.uint8)
    arr = np.tile(grad, (h,1))
    return np.stack([arr,arr,arr], axis=2)


def img_vertical_gradient(size=(256,256), start=0, end=177, seed=None):
    h, w = size
    grad = np.linspace(start, end, h, dtype=np.uint8)
    arr = np.tile(grad[:,None], (1,w))
    return np.stack([arr,arr,arr], axis=2)


def img_radial_gradient(size=(256,256), center=None, seed=None):
    h, w = size
    cx, cy = (w//2, h//2) if center is None else center

    ys = np.arange(h)[:,None]
    xs = np.arange(w)[None,:]

    dist = np.sqrt((xs - cx)**2 + (ys - cy)**2)
    norm = dist / dist.max()
    arr = (norm*177).astype(np.uint8)
    return np.stack([arr,arr,arr], axis=2)


def img_checkerboard(size=(256,256), block_size=32, seed=None):
    h, w = size
    ys = np.arange(h)//block_size
    xs = np.arange(w)//block_size
    pat = (ys[:,None] + xs[None,:]) % 2
    arr = (pat*177).astype(np.uint8)
    return np.stack([arr,arr,arr], axis=2)


def img_random_noise(size=(256,256), seed=None):
    rng = _rng(seed)
    h, w = size
    return rng.integers(0,256,(h,w,3),dtype=np.uint8)


def img_stripes(size=(256,256), stripe_width=10, direction="vertical", seed=None):
    h, w = size
    arr = np.zeros((h,w), dtype=np.uint8)
    if direction == "vertical":
        arr[:, ::2*stripe_width] = 177
        arr[:, stripe_width::2*stripe_width] = 177
    else:
        arr[::2*stripe_width, :] = 177
        arr[stripe_width::2*stripe_width, :] = 177
    return np.stack([arr,arr,arr], axis=2)


def img_sine_pattern(size=(256,256), freq=5, seed=None):
    h, w = size
    xs = np.linspace(0, 2*np.pi*freq, w)
    sine = (np.sin(xs)*127 + 128).astype(np.uint8)
    arr = np.tile(sine, (h,1))
    return np.stack([arr,arr,arr], axis=2)


def img_perlin_noise(size=(256,256), scale=8, seed=None):
    rng = _rng(seed)
    h, w = size

    grid_y = h // scale
    grid_x = w // scale

    coarse = rng.random((grid_y+1, grid_x+1))
    arr = np.zeros((h,w))

    for y in range(h):
        gy = y / scale
        y0 = int(gy)
        y1 = y0 + 1
        ty = gy - y0

        for x in range(w):
            gx = x / scale
            x0 = int(gx)
            x1 = x0 + 1
            tx = gx - x0

            v00 = coarse[y0, x0]
            v10 = coarse[y0, x1]
            v01 = coarse[y1, x0]
            v11 = coarse[y1, x1]

            a = v00*(1-tx) + v10*tx
            b = v01*(1-tx) + v11*tx
            arr[y,x] = a*(1-ty) + b*ty

    arr = (arr*177).astype(np.uint8)
    return np.stack([arr,arr,arr], axis=2)

def img_concentric_rings(size=(256,256), ring_width=10, seed=None):
    h, w = size
    cx, cy = w//2, h//2
    ys = np.arange(h)[:,None]
    xs = np.arange(w)[None,:]
    dist = np.sqrt((xs-cx)**2 + (ys-cy)**2)
    arr = ((dist // ring_width) % 2)*177
    arr = arr.astype(np.uint8)
    return np.stack([arr,arr,arr], axis=2)


def img_diagonal_stripes(size=(256,256), width=15, seed=None):
    h, w = size
    ys = np.arange(h)[:,None]
    xs = np.arange(w)[None,:]
    arr = (((xs + ys) // width) % 2)*177
    arr = arr.astype(np.uint8)
    return np.stack([arr,arr,arr], axis=2)


def img_mandelbrot(size=(256,256), max_iter=50, seed=None):
    h, w = size
    xs = np.linspace(-2.0, 1.0, w)
    ys = np.linspace(-1.5, 1.5, h)
    arr = np.zeros((h,w), dtype=np.uint8)

    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            c = complex(x,y)
            z = 0
            count = 0
            while abs(z) <= 2 and count < max_iter:
                z = z*z + c
                count += 1
            arr[i,j] = int(177*count/max_iter)

    return np.stack([arr,arr,arr], axis=2)


def img_wave_interference(size=(256,256), freq1=5, freq2=7, seed=None):
    h, w = size
    xs = np.linspace(0, 2*np.pi, w)
    ys = np.linspace(0, 2*np.pi, h)
    X, Y = np.meshgrid(xs, ys)

    waves = (np.sin(freq1*X) + np.sin(freq2*Y)) / 2
    arr = ((waves+1)*127.5).astype(np.uint8)
    return np.stack([arr,arr,arr], axis=2)


def img_color_cycle(size=(256,256), seed=None):
    h, w = size
    xs = np.linspace(0, 2*np.pi, w)
    r = (np.sin(xs)*127 + 128).astype(np.uint8)
    g = (np.sin(xs + 2.09)*127 + 128).astype(np.uint8)  # 120° shift
    b = (np.sin(xs + 4.18)*127 + 128).astype(np.uint8)  # 240° shift
    R = np.tile(r, (h,1))
    G = np.tile(g, (h,1))
    B = np.tile(b, (h,1))
    return np.stack([R,G,B], axis=2)


def img_pixel_grid(size=(256,256), grid=16, seed=None):
    rng = _rng(seed)
    h, w = size
    gh = h // grid
    gw = w // grid

    # grid colors
    block_colors = rng.integers(0,256,(gh,gw,3),dtype=np.uint8)

    # upsample
    arr = np.repeat(np.repeat(block_colors, grid, axis=0), grid, axis=1)
    return arr[:h,:w]


def img_moire(size=(256,256), freq=4, seed=None):
    h, w = size
    xs = np.arange(w)
    ys = np.arange(h)
    X, Y = np.meshgrid(xs, ys)
    arr = ((np.sin(X/freq) + np.sin(Y/(freq+3))) * 0.5 + 0.5) * 177
    arr = arr.astype(np.uint8)
    return np.stack([arr,arr,arr], axis=2)


IMAGE_PATTERNS = (
    img_solid_color,
    img_horizontal_gradient,
    img_vertical_gradient,
    img_radial_gradient,
    img_random_noise,
    img_checkerboard,
    img_stripes,
    img_sine_pattern,
    # img_perlin_noise,
    img_concentric_rings,
    # img_diagonal_stripes,
    img_mandelbrot,
    img_wave_interference,
    img_color_cycle,
    img_pixel_grid,
    img_moire
)
