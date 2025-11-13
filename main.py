import os
import random
import numpy as np
from image_patterns import IMAGE_PATTERNS
from string import ascii_letters
from PIL import Image


def create_random_image(path, size=(512, 512), seed=42):

    width, height = size

    random_img_idx = random.choice(range(len(IMAGE_PATTERNS)))
    random_img = IMAGE_PATTERNS[random_img_idx](size=size)

    filename = os.path.join(path, generate_random_filename())
    Image.fromarray(random_img, mode="RGB").save(filename)

    return filename


def create_circle_on_image(image_path,
                           output_path,
                           center: tuple[float, float] = None, # (h, k)
                           radius = 5,  # r
                           seed=42
                           ):

    img = Image.open(image_path).convert("RGB")
    img_arr = np.array(img, dtype=np.uint8) # (H, W, 3)
    H, W = img_arr.shape[:-1]

    if not center:
        
        h = float(random.randint(0, H-1))
        k = float(random.randint(0, H-1))
        center = (h, k)
    cx, cy = center

    r = radius
    if radius is None:
        # compute farthest possible distance from (cx,cy) to image corners
        # corners: (0,0), (W-1,0), (0,H-1), (W-1,H-1)
        d0 = np.hypot(cx - 0.0, cy - 0.0)
        d1 = np.hypot(cx - (W - 1.0), cy - 0.0)
        d2 = np.hypot(cx - 0.0, cy - (H - 1.0))
        d3 = np.hypot(cx - (W - 1.0), cy - (H - 1.0))
        max_possible = max(d0, d1, d2, d3)

        # pick radius randomly between min_radius and max_possible
        # use at least min_radius and allow up to max_possible (float)
        if max_possible < radius:
            r = float(radius)
        else:
            # choose uniform in [min_radius, max_possible]
            r = float(random.random() * (max_possible - radius) + radius)

    # build circle area using circle equation (x-cx)^2 + (y-cy)^2 <= r^2
    # Create coordinate grids: xs shape (1, W), ys shape (H, 1) to broadcast
    ys = np.arange(H, dtype=np.float64)[:, None]   # shape (H,1)
    xs = np.arange(W, dtype=np.float64)[None, :]   # shape (1,W)

    dist2 = (xs - cx)**2 + (ys - cy)**2
    circle_area = dist2 <= (r * r)


    # apply white color (255) to circle area (pixels)
    img_arr[circle_area] = 255

    filename = f"defect_circle_{image_path.split('/')[-1]}"
    output_filename = os.path.join(output_path, filename)
    Image.fromarray(img_arr).save(output_filename)

    return output_filename


def generate_random_filename():
    values = ascii_letters + ''.join(str(i) for i in range(0, 10))
    return f"image_{''.join(random.choice(values) for _ in range(10))}.png"



if __name__ == "__main__":
    image_folder  = "images/"
    output_folder = "defect_shapes/"
    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    for i in range(5):
        print(f"Generating {i+1} image")
        image = create_random_image(image_folder)
        circle_defect_img = create_circle_on_image(image, output_folder,
                                                   radius=random.randint(15, 70))