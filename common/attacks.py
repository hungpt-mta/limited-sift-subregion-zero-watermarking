import cv2
import numpy as np
from .config import IMG_SIZE


def clip255_u8(x):
    return np.clip(x, 0, 255).astype(np.uint8)


def attack_gaussian_noise_bgr(bgr: np.ndarray, sigma: float):
    noise = np.random.normal(0.0, float(sigma) * 255.0, size=bgr.shape).astype(np.float32)
    return clip255_u8(bgr.astype(np.float32) + noise)


def attack_salt_pepper_bgr(bgr: np.ndarray, density: float):
    out = bgr.copy()
    rnd = np.random.rand(out.shape[0], out.shape[1])
    salt = rnd < (float(density) / 2.0)
    pepper = (rnd >= (float(density) / 2.0)) & (rnd < float(density))
    out[salt] = (255, 255, 255)
    out[pepper] = (0, 0, 0)
    return out


def attack_speckle_bgr(bgr: np.ndarray, var: float):
    n = np.random.normal(0.0, np.sqrt(float(var)), size=bgr.shape).astype(np.float32)
    return clip255_u8(bgr.astype(np.float32) + bgr.astype(np.float32) * n)


def attack_gaussian_blur_bgr(bgr: np.ndarray, k: int):
    k = int(k)
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(bgr, (k, k), 0)


def attack_median_blur_bgr(bgr: np.ndarray, k: int):
    k = int(k)
    if k % 2 == 0:
        k += 1
    return cv2.medianBlur(bgr, k)


def attack_jpeg_bgr(bgr: np.ndarray, quality: int):
    enc = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, buf = cv2.imencode('.jpg', bgr, enc)
    if not ok:
        return bgr.copy()
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.resize(dec, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


def attack_resize_bgr(bgr: np.ndarray, scale: float):
    h, w = bgr.shape[:2]
    nh, nw = max(1, int(round(h * float(scale)))), max(1, int(round(w * float(scale))))
    tmp = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    return cv2.resize(tmp, (w, h), interpolation=cv2.INTER_AREA)



def attack_rotate_bgr(bgr: np.ndarray, angle_deg: float):
    """Rotate on a fixed canvas with crop effect and black padding.

    The output size is unchanged. Image content rotated outside the frame is
    lost, and newly exposed pixels are filled with black. This matches the
    geometric attack protocol used in the reviewer experiments.
    """
    h, w = bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(angle_deg), 1.0)
    return cv2.warpAffine(
        bgr, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def attack_center_crop_bgr(bgr: np.ndarray, crop_ratio: float):
    """Center cropping attack with fixed image size and blacked cropped area.

    crop_ratio is interpreted as the fraction of image area to remove from the
    centre. The image is not resized. The removed central region is filled with
    black.
    """
    h, w = bgr.shape[:2]
    cr = float(np.clip(crop_ratio, 0.0, 0.95))
    out = bgr.copy()
    side_scale = float(np.sqrt(cr))
    crop_w = max(1, int(round(w * side_scale)))
    crop_h = max(1, int(round(h * side_scale)))
    x0 = (w - crop_w) // 2
    y0 = (h - crop_h) // 2
    out[y0:y0 + crop_h, x0:x0 + crop_w] = (0, 0, 0)
    return out


def attack_crop_x_bgr(bgr: np.ndarray, crop_ratio: float):
    """Cropping attack along the X-axis from the left image edge.

    The image size is unchanged. A vertical strip from the left edge with width
    crop_ratio * image_width is filled with black. No resize is applied.
    """
    h, w = bgr.shape[:2]
    cr = float(np.clip(crop_ratio, 0.0, 0.95))
    out = bgr.copy()
    crop_w = max(1, int(round(w * cr)))
    out[:, :crop_w] = (0, 0, 0)
    return out


def attack_crop_y_bgr(bgr: np.ndarray, crop_ratio: float):
    """Cropping attack along the Y-axis from the top image edge.

    The image size is unchanged. A horizontal strip from the top edge with
    height crop_ratio * image_height is filled with black. No resize is applied.
    """
    h, w = bgr.shape[:2]
    cr = float(np.clip(crop_ratio, 0.0, 0.95))
    out = bgr.copy()
    crop_h = max(1, int(round(h * cr)))
    out[:crop_h, :] = (0, 0, 0)
    return out

def attack_brightness_only_bgr(bgr: np.ndarray, beta: float):
    return clip255_u8(bgr.astype(np.float32) + float(beta))


def attack_contrast_only_bgr(bgr: np.ndarray, alpha: float):
    return clip255_u8(float(alpha) * bgr.astype(np.float32))



def attack_translate_bgr(bgr: np.ndarray, dx_ratio: float = 0.0, dy_ratio: float = 0.0):
    """Translate image on a fixed canvas with black padding.

    Pixels shifted outside the frame are lost. Newly exposed regions are filled
    with black; no reflection, wrap-around, or content reuse is applied.
    """
    h, w = bgr.shape[:2]
    dx = int(round(w * float(dx_ratio)))
    dy = int(round(h * float(dy_ratio)))
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        bgr, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

def attack_translate_x_bgr(bgr: np.ndarray, shift_ratio: float):
    return attack_translate_bgr(bgr, dx_ratio=float(shift_ratio), dy_ratio=0.0)


def _item(name, group, intensity, fn, psnr=None, suite=None):
    return {
        'name': name,
        'group': group,
        'intensity': intensity,
        'psnr_reported': psnr,
        'suite': suite,
        'fn': fn,
    }


def make_attack_suite_1_bgr():
    """Reviewer/Nawaz-style attack suite 1.

    Median filter is applied once. Rotation is fixed-canvas with crop effect.
    """
    attacks = []
    suite = 'suite1'
    for s, psnr in [(0.01, 22.00), (0.02, 19.10), (0.05, 15.37)]:
        pct = int(round(s * 100))
        attacks.append(_item(f'gaussian_noise_{pct}pct', 'Gaussian Noise', f'{pct}%', lambda img, s=s: attack_gaussian_noise_bgr(img, s), psnr, suite))
    for q, psnr in [(15, 31.80), (25, 33.86), (30, 34.53), (40, 35.14), (50, 36.52)]:
        attacks.append(_item(f'jpeg_q{q}', 'JPEG Compression', f'{q}%', lambda img, q=q: attack_jpeg_bgr(img, q), psnr, suite))
    for k, psnr in [(3, 29.16), (5, 25.78), (7, 23.61)]:
        attacks.append(_item(f'median_filter_{k}x{k}_once', 'Median Filter', f'[{k}x{k}]', lambda img, k=k: attack_median_blur_bgr(img, k), psnr, suite))
    for ang, psnr in [(10, 16.68), (20, 15.33), (60, 13.92), (70, 13.67), (80, 13.50)]:
        attacks.append(_item(f'rotate_cw_{ang}deg', 'Rotation clockwise', f'{ang}°', lambda img, ang=ang: attack_rotate_bgr(img, -ang), psnr, suite))
    for ang, psnr in [(15, 15.86), (30, 14.89), (50, 14.27), (60, 13.92), (80, 13.50)]:
        attacks.append(_item(f'rotate_ccw_{ang}deg', 'Rotation anticlockwise', f'{ang}°', lambda img, ang=ang: attack_rotate_bgr(img, ang), psnr, suite))
    for sc in [0.4, 0.6, 0.9, 1.2, 1.4]:
        attacks.append(_item(f'scaling_x{sc}', 'Scaling', f'×{sc}', lambda img, sc=sc: attack_resize_bgr(img, sc), None, suite))
    for pct, psnr in [(10, 13.91), (15, 13.06), (20, 12.65), (30, 12.24), (35, 12.20)]:
        r = pct / 100.0
        attacks.append(_item(f'translate_left_{pct}pct', 'Left Translation', f'{pct}%', lambda img, r=r: attack_translate_bgr(img, dx_ratio=-r), psnr, suite))
    for pct, psnr in [(10, 13.98), (15, 13.10), (20, 12.73), (30, 12.20), (35, 12.11)]:
        r = pct / 100.0
        attacks.append(_item(f'translate_right_{pct}pct', 'Right Translation', f'{pct}%', lambda img, r=r: attack_translate_bgr(img, dx_ratio=r), psnr, suite))
    for pct, psnr in [(10, 13.79), (15, 12.94), (20, 12.27), (25, 11.78), (30, 11.42)]:
        r = pct / 100.0
        attacks.append(_item(f'translate_up_{pct}pct', 'Up Translation', f'{pct}%', lambda img, r=r: attack_translate_bgr(img, dy_ratio=-r), psnr, suite))
    for pct, psnr in [(7, 14.75), (10, 13.97), (15, 13.07), (20, 12.42), (25, 11.97)]:
        r = pct / 100.0
        attacks.append(_item(f'translate_down_{pct}pct', 'Down Translation', f'{pct}%', lambda img, r=r: attack_translate_bgr(img, dy_ratio=r), psnr, suite))
    for pct in [3, 10, 15, 20, 25]:
        r = pct / 100.0
        attacks.append(_item(f'crop_x_{pct}pct', 'Cropping X-axis', f'{pct}%', lambda img, r=r: attack_crop_x_bgr(img, r), None, suite))
    for pct in [3, 10, 15, 20, 25]:
        r = pct / 100.0
        attacks.append(_item(f'crop_y_{pct}pct', 'Cropping Y-axis', f'{pct}%', lambda img, r=r: attack_crop_y_bgr(img, r), None, suite))
    return attacks


def make_attack_suite_2_bgr():
    """Extended attack suite 2 from the user's benchmark specification."""
    attacks = []
    suite = 'suite2'
    for beta in [10, 20, 30, 40, 50]:
        attacks.append(_item(f'brightness_beta{beta}', 'Brightness', str(beta), lambda img, beta=beta: attack_brightness_only_bgr(img, beta), None, suite))
    for alpha in [1.1, 1.3, 1.5, 2.0]:
        attacks.append(_item(f'contrast_alpha{alpha}', 'Contrast', str(alpha), lambda img, alpha=alpha: attack_contrast_only_bgr(img, alpha), None, suite))
    for s in [0.005, 0.01, 0.03, 0.05, 0.1]:
        attacks.append(_item(f'gauss_noise_sigma{s}', 'Gaussian noise', str(s), lambda img, s=s: attack_gaussian_noise_bgr(img, s), None, suite))
    for v in [0.005, 0.01, 0.03, 0.05, 0.1]:
        attacks.append(_item(f'speckle_var{v}', 'Speckle noise', str(v), lambda img, v=v: attack_speckle_bgr(img, v), None, suite))
    for d in [0.005, 0.01, 0.03, 0.05, 0.1]:
        pct = f'{d*100:g}%'
        attacks.append(_item(f'saltpepper_d{d}', 'Salt & pepper noise', pct, lambda img, d=d: attack_salt_pepper_bgr(img, d), None, suite))
    for q in [10, 30, 50, 70, 90]:
        attacks.append(_item(f'jpeg_q{q}', 'JPEG Compression', str(q), lambda img, q=q: attack_jpeg_bgr(img, q), None, suite))
    for k in [3, 5, 7, 9, 11]:
        attacks.append(_item(f'gauss_blur_k{k}', 'Gaussian Filter', f'[{k}x{k}]', lambda img, k=k: attack_gaussian_blur_bgr(img, k), None, suite))
    for k in [3, 5, 7, 9, 11]:
        attacks.append(_item(f'median_blur_k{k}', 'Median Filter', f'[{k}x{k}]', lambda img, k=k: attack_median_blur_bgr(img, k), None, suite))
    for sc in [0.25, 0.5, 0.75, 1.25, 1.5, 1.75, 2.0]:
        attacks.append(_item(f'scaling_x{sc}', 'Scaling', f'×{sc}', lambda img, sc=sc: attack_resize_bgr(img, sc), None, suite))
    for ang in [10, 20, 60, 70, 80]:
        attacks.append(_item(f'rotate_{ang}deg', 'Rotation', f'{ang}°', lambda img, ang=ang: attack_rotate_bgr(img, ang), None, suite))
    for cr in [0.05, 0.1, 0.2, 0.3]:
        pct = f'{cr*100:g}%'
        attacks.append(_item(f'crop_center_{cr}', 'Crop center', pct, lambda img, cr=cr: attack_center_crop_bgr(img, cr), None, suite))
    for tr in [0.05, 0.1, 0.15, 0.2, 0.3]:
        pct = f'{tr*100:g}%'
        attacks.append(_item(f'translate_x_{tr}', 'Translation X-axis', pct, lambda img, tr=tr: attack_translate_x_bgr(img, tr), None, suite))
    return attacks


def make_attacks_bgr():
    # Backward-compatible default: extended Suite 2.
    return [(d['name'], d['fn']) for d in make_attack_suite_2_bgr()]


def make_attack_suite_bgr(suite: str):
    s = str(suite).lower().strip()
    if s in {'1', 'suite1', 'a'}:
        return make_attack_suite_1_bgr()
    if s in {'2', 'suite2', 'b'}:
        return make_attack_suite_2_bgr()
    if s in {'both', 'all'}:
        return make_attack_suite_1_bgr() + make_attack_suite_2_bgr()
    raise ValueError(f'Unknown attack suite: {suite}')
