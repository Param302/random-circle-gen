import os
import time
import random
import numpy as np
from PIL import Image
from string import ascii_letters
from image_patterns import IMAGE_PATTERNS


def create_random_image(path, size=(512, 512), seed=42):

    width, height = size

    random_img_idx = random.choice(range(len(IMAGE_PATTERNS)))
    random_img = IMAGE_PATTERNS[random_img_idx](size=size, seed=seed)

    filename = os.path.join(path, generate_random_filename(seed))
    Image.fromarray(random_img, mode="RGB").save(filename)

    return filename


def create_circle_on_image(image_path,
                           output_path,
                           center: tuple[float, float] = None,  # (h, k)
                           radius=5,  # r
                           seed=42
                           ):

    img = Image.open(image_path).convert("RGB")
    img_arr = np.array(img, dtype=np.uint8)  # (H, W, 3)
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

    filename = f"defect_circle_{os.path.basename(image_path)}"
    output_filename = os.path.join(output_path, filename)
    Image.fromarray(img_arr).save(output_filename)

    return output_filename


def create_ellipse_on_image(image_path,
                            output_path,
                            # (h, k) i.e. (cx, cy)
                            center: tuple[float, float] = None,
                            radius_x: float = None,  # rx (half-width)
                            radius_y: float = None,  # ry (half-height)
                            min_radius: int = 5,
                            seed: int = 42):
    random.seed(seed)

    img = Image.open(image_path).convert("RGB")
    img_arr = np.array(img, dtype=np.uint8)  # (H, W, 3)
    H, W = img_arr.shape[:-1]

    # choose center (cx, cy) -- note order: x is horizontal (0..W-1), y is vertical (0..H-1)
    if not center:
        cx = float(random.randint(0, W - 1))
        cy = float(random.randint(0, H - 1))
        center = (cx, cy)
    else:
        cx, cy = float(center[0]), float(center[1])

    # choose rx, ry dependent on center if not provided
    # maximum distances to image edges from center
    max_left = cx
    max_right = (W - 1) - cx
    max_top = cy
    max_bottom = (H - 1) - cy

    max_rx = max(max_left, max_right)
    max_ry = max(max_top, max_bottom)

    # ensure reasonable minima
    if radius_x is None:
        if max_rx < min_radius:
            rx = float(min_radius)
        else:
            rx = float(random.uniform(min_radius, max_rx))
    else:
        rx = float(radius_x)

    if radius_y is None:
        if max_ry < min_radius:
            ry = float(min_radius)
        else:
            ry = float(random.uniform(min_radius, max_ry))
    else:
        ry = float(radius_y)

    # create coordinate grids
    ys = np.arange(H, dtype=np.float64)[:, None]  # shape (H,1)
    xs = np.arange(W, dtype=np.float64)[None, :]  # shape (1,W)

    # ellipse equation mask (axis-aligned)
    # ((x-cx)/rx)^2 + ((y-cy)/ry)^2 <= 1
    # avoid division by zero
    rx = max(rx, 1e-6)
    ry = max(ry, 1e-6)

    norm = ((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2
    mask = norm <= 1.0

    # apply white color (255) to ellipse area (pixels)
    img_arr[mask] = 255

    # save output
    filename = f"defect_ellipse_{os.path.basename(image_path)}"
    output_filename = os.path.join(output_path, filename)
    os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)
    Image.fromarray(img_arr).save(output_filename)

    return output_filename


def create_circle_ellipse_merged_on_image(image_path,
                                          output_path,
                                          # center for ellipse (cx,cy)
                                          center: tuple[float, float] = None,
                                          ellipse_rx: float = None,
                                          ellipse_ry: float = None,
                                          circle_radius: float = None,
                                          # (dx, dy) relative to ellipse center
                                          circle_offset: tuple[float,
                                                               float] = None,
                                          min_radius: int = 5,
                                          seed: int = 42):
    """
    Draw one merged shape formed by an axis-aligned ellipse and a circle (both filled white).
    Guarantees the circle overlaps the ellipse to produce a single spot.

    - If center is not provided, choose random center inside image.
    - ellipse_rx/ellipse_ry chosen dependent on center if not provided.
    - circle_radius chosen dependent on ellipse size if not provided.
    - circle_offset can be provided to place circle relative to ellipse center; otherwise chosen
      randomly to ensure overlap.
    - Returns output filename.
    """
    random.seed(seed)

    img = Image.open(image_path).convert("RGB")
    img_arr = np.array(img, dtype=np.uint8)  # (H, W, 3)
    H, W = img_arr.shape[:-1]

    # pick ellipse center (cx, cy)
    if not center:
        cx = float(random.randint(0, W - 1))
        cy = float(random.randint(0, H - 1))
        center = (cx, cy)
    else:
        cx, cy = float(center[0]), float(center[1])

    # determine ellipse rx, ry if needed (dependent on center)
    max_left = cx
    max_right = (W - 1) - cx
    max_top = cy
    max_bottom = (H - 1) - cy

    max_rx = max(max_left, max_right)
    max_ry = max(max_top, max_bottom)

    if ellipse_rx is None:
        ellipse_rx = float(random.uniform(min_radius, max(min_radius, max_rx)))
    else:
        ellipse_rx = float(ellipse_rx)

    if ellipse_ry is None:
        ellipse_ry = float(random.uniform(min_radius, max(min_radius, max_ry)))
    else:
        ellipse_ry = float(ellipse_ry)

    # choose circle radius relative to ellipse size if not provided
    # reasonable default: between 0.3*min(rx,ry) and 1.1*max(rx,ry)
    min_cr = max(min_radius, 0.3 * min(ellipse_rx, ellipse_ry))
    max_cr = max(min_cr, 1.1 * max(ellipse_rx, ellipse_ry))
    if circle_radius is None:
        circle_radius = float(random.uniform(min_cr, max_cr))
    else:
        circle_radius = float(circle_radius)

    # choose circle offset so the circle overlaps the ellipse
    # If caller provided explicit offset, use it; otherwise choose offset that ensures overlap.
    if circle_offset is not None:
        dx, dy = float(circle_offset[0]), float(circle_offset[1])
    else:
        # pick a point inside the ellipse bounding box but biased near edge so partial shapes occur
        # ellipse bounding box ranges: cx - rx .. cx + rx  and cy - ry .. cy + ry
        # choose a random point inside bbox, then ensure overlap by selecting distance < (rx+circle_radius)
        # We'll attempt a few times to get a point that creates overlap; rarely fail.
        attempts = 0
        found = False
        while attempts < 20 and not found:
            attempts += 1
            # sample a candidate circle center inside ellipse bbox
            cand_x = random.uniform(cx - ellipse_rx, cx + ellipse_rx)
            cand_y = random.uniform(cy - ellipse_ry, cy + ellipse_ry)
            # compute distance to ellipse center
            d = np.hypot(cand_x - cx, cand_y - cy)
            # simple overlap condition: distance <= ellipse_major + circle_radius (conservative)
            if d <= max(ellipse_rx, ellipse_ry) + circle_radius:
                dx = cand_x - cx
                dy = cand_y - cy
                found = True
        if not found:
            # fallback: place circle near center
            dx = 0.0
            dy = 0.0

    # compute actual circle center
    circle_cx = cx + dx
    circle_cy = cy + dy

    # Build masks
    ys = np.arange(H, dtype=np.float64)[:, None]  # (H,1)
    xs = np.arange(W, dtype=np.float64)[None, :]  # (1,W)

    # ellipse mask: ((x-cx)/rx)^2 + ((y-cy)/ry)^2 <= 1
    rx = max(ellipse_rx, 1e-6)
    ry = max(ellipse_ry, 1e-6)
    ellipse_norm = ((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2
    ellipse_mask = ellipse_norm <= 1.0

    # circle mask: (x-circle_cx)^2 + (y-circle_cy)^2 <= r^2
    r = max(circle_radius, 1e-6)
    circle_dist2 = (xs - circle_cx) ** 2 + (ys - circle_cy) ** 2
    circle_mask = circle_dist2 <= (r * r)

    # merged mask: union (ensure single white spot)
    merged_mask = ellipse_mask | circle_mask

    # apply white to merged area
    img_arr[merged_mask] = 255

    # save
    filename = f"defect_merge_{os.path.basename(image_path)}"
    output_filename = os.path.join(output_path, filename)
    os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)
    Image.fromarray(img_arr).save(output_filename)

    return output_filename


def generate_random_filename(seed=42):
    random.seed(seed)
    values = ascii_letters + ''.join(str(i) for i in range(0, 10))
    return f"image_{''.join(random.choice(values) for _ in range(10))}.png"


if __name__ == "__main__":
    image_folder = "images/"
    output_folder = "defect_shapes/"
    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    for i in range(5):
        random_seed = random.randint(1, 10) * (i+1)
        print(f"Generating {i+1} image")
        image = create_random_image(image_folder, seed=random_seed)
        print("Base image", image)

        circle_defect_img = create_circle_on_image(
            image, output_folder, radius=random.randint(15, 70), seed=random_seed)

        ellipse_defect_img = create_ellipse_on_image(image, output_folder, radius_x=random.randint(
            15, 40), radius_y=random.randint(25, 70), seed=random_seed)

        mixed_defect_img = create_circle_ellipse_merged_on_image(image, output_folder, ellipse_rx=random.randint(
            15, 40), ellipse_ry=random.randint(40, 70), circle_radius=random.randint(30, 80), circle_offset=(15, 30), seed=random_seed)

        time.sleep(1)
