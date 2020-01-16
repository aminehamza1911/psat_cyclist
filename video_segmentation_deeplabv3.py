import cv2
import numpy as np
import torch
import torchvision.transforms as T
from cv2.cv2 import Canny

from decode_segmap import decode_segmap
from deeplabv3.deeplabv3 import DeepLabV3

"""
- Applies a segmentation
- Detects road surface
- Calculates road condition
- Checks if sidewalk is present/visible/accessible or not
"""
ESC_CODE = 27
NB_CLASSES = 20
CANNY_THRESHOLD_1 = 200
CANNY_THRESHOLD_2 = 300
ROAD_CLASS = 0
SIDEWALK_CLASS = 1

network = DeepLabV3().cpu()
network.load_state_dict(torch.load("./deeplabv3/model_13_2_2_2_epoch_580.pth", map_location=torch.device('cpu')))
network.eval()

to_net_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])])

cap = cv2.VideoCapture('./data/Benjamin02.avi')

# take first frame of the video
retval, image = cap.read()

framenb = 1

# Skip beginning of video
for i in range(1000):
    cap.read()

with torch.no_grad():
    while retval and cv2.waitKey(1) != ESC_CODE:
        print(f"Frame {framenb}")
        retval, image = cap.read()

        # Semantic segmentation
        inp = to_net_transform(image).unsqueeze(0)
        out = network(inp)
        outputs = out.data.cpu().numpy()  # (shape: (batch_size, num_classes, img_h, img_w))
        pred_label_imgs = np.argmax(outputs, axis=1)  # (shape: (batch_size, img_h, img_w))
        pred_label_imgs = pred_label_imgs.astype(np.uint8)
        pred_label_imgs = pred_label_imgs[0]

        # Detect surface type
        line = pred_label_imgs[0]
        bottom_middle = pred_label_imgs[int(0.7 * len(pred_label_imgs)):int(0.9 * len(pred_label_imgs)),
                        int(0.4 * len(line)):int(0.6 * len(line))]
        bottom_middle = bottom_middle.flatten()
        c = np.bincount(bottom_middle)
        if np.argmax(c) == ROAD_CLASS:
            print("On road")
        elif np.argmax(c) == SIDEWALK_CLASS:
            print("On sidewalk")
        else:
            print("Surface unknown")

        # Detect road condition, equal to number of white pixels in canny image
        # TODO normalize first
        canny = Canny(image, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2)
        # TODO: find more efficient way, maybe with Image.composite or numpy function to filter values
        for i in range(len(canny)):
            for j in range(len(canny[0])):
                if pred_label_imgs[i][j] != ROAD_CLASS:
                    canny[i][j] = 0
        road_damage_value = np.count_nonzero(canny)
        print(f"Road damage: {road_damage_value}")


        # Detect if sidewalk is visible
        sidewalk = (pred_label_imgs == SIDEWALK_CLASS)
        sidewalk_count = np.count_nonzero(sidewalk)
        # If sidewalk is at least 3% of image
        if sidewalk_count >= 3/100 * len(pred_label_imgs) * len(pred_label_imgs[0]):
            print("Sidewalk present and visible")
        else:
            print("Sidewalk not present or not visible")

        rgb = decode_segmap(pred_label_imgs, NB_CLASSES)

        cv2.imshow('rgb', rgb)
        cv2.imshow('image', image)
        cv2.imshow('canny', canny)
        cv2.imwrite(f"out/frame{framenb}.jpg", image)
        framenb += 1

    cv2.destroyAllWindows()
    cap.release()
