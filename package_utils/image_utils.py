#-*- coding: utf-8 -*-
from PIL import Image
import numpy as np
from matplotlib import cm


def load_image(image_path):
    img = Image.open(image_path)
    img = np.array(img)
    return img


def crop_by_margin(image, margin=[0, 0]):
    H, W = image.shape[:2]
    margin_x, margin_y = margin
    image = image[margin_y:H-margin_y, margin_x:W-margin_x, :]
    return image


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return min(r1, r2, r3)


def cal_mask_wh(p, mask):
    cy, cx = p
    mask_h, mask_w = mask.shape
    w = 0
    h = 0

    for i in [-1, 1]:
        shift_y = 0
        while (cy+shift_y > -mask_h) and (cy+shift_y < mask_h) and (mask[cy+shift_y, cx] > 128):
            w += 1
            shift_y += i

        shift_x = 0
        while (cx+shift_x > -mask_w) and (cx+shift_x < mask_w) and (mask[cy, cx+shift_x] > 128):
            h += 1
            shift_x += i

    return w, h


def overlay_mask(img: Image.Image, mask: Image.Image, colormap: str = "jet", alpha: float = 0.7) -> Image.Image:
    """Overlay a colormapped mask on a background image

    >>> from PIL import Image
    >>> import matplotlib.pyplot as plt
    >>> from torchcam.utils import overlay_mask
    >>> img = ...
    >>> cam = ...
    >>> overlay = overlay_mask(img, cam)

    Args:
        img: background image
        mask: mask to be overlayed in grayscale
        colormap: colormap to be applied on the mask
        alpha: transparency of the background image

    Returns:
        overlayed image

    Raises:
        TypeError: when the arguments have invalid types
        ValueError: when the alpha argument has an incorrect value
    """

    if not isinstance(img, Image.Image) or not isinstance(mask, Image.Image):
        raise TypeError("img and mask arguments need to be PIL.Image")

    if not isinstance(alpha, float) or alpha < 0 or alpha >= 1:
        raise ValueError("alpha argument is expected to be of type float between 0 and 1")

    cmap = cm.get_cmap(colormap)
    # Resize mask and apply colormap
    overlay = mask.resize(img.size, resample=Image.BICUBIC)
    overlay = (255 * cmap(np.asarray(overlay) ** 1)[:, :, :3]).astype(np.uint8)
    # Overlay the image with the mask
    overlayed_img = Image.fromarray((alpha * np.asarray(img) + (1 - alpha) * overlay).astype(np.uint8))

    return overlayed_img
