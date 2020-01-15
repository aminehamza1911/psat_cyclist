import numpy as np
from colour import Color

# Define the helper function
def decode_segmap(image, nb_classes):

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for detected_class in range(nb_classes):
        idx = image == detected_class
        class_color = Color(pick_for=detected_class).rgb
        r[idx] = class_color[0] * 255
        g[idx] = class_color[1] * 255
        b[idx] = class_color[2] * 255

    rgb = np.stack([r, g, b], axis=2)
    return rgb
