import numpy as np

NC = 21

# Define the helper function
def decode_segmap(image):
    # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
    # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
    # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
    # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, NC):
        idx = image == l
        class_color = (0, 0, 0)
        # Vehicles
        if l in {1, 2, 4, 6, 7, 14, 19}:
            class_color = (128, 0, 0)
        # Living beings
        elif l in {3, 10, 12, 13, 15, 17}:
            class_color = (0, 128, 0)
        r[idx] = class_color[0]
        g[idx] = class_color[1]
        b[idx] = class_color[2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb
