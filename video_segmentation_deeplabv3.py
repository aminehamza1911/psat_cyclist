import time

import cv2
import numpy as np
import torch
import torchvision.transforms as T

from decode_segmap import decode_segmap
from deeplabv3.deeplabv3 import DeepLabV3

ESC_CODE = 27
NB_CLASSES = 20

network = DeepLabV3().cpu()
network.load_state_dict(torch.load("./deeplabv3/model_13_2_2_2_epoch_580.pth", map_location=torch.device('cpu')))
network.eval()

to_net_transform = T.Compose([T.ToPILImage(),
                              # T.Resize(224),  # , 300)),
                              T.ToTensor(),
                              T.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])])

cap = cv2.VideoCapture('./data/Benjamin02.avi')

# take first frame of the video
retval, image = cap.read()

framenb = 1

# Skip beginning of video
for i in range(2000):
    cap.read()

with torch.no_grad():
    while retval and cv2.waitKey(1) != ESC_CODE:
        retval, image = cap.read()

        inp = to_net_transform(image).unsqueeze(0)

        print("b4", time.time())
        out = network(inp)
        print("af", time.time())

        outputs = out.data.cpu().numpy()  # (shape: (batch_size, num_classes, img_h, img_w))

        pred_label_imgs = np.argmax(outputs, axis=1)  # (shape: (batch_size, img_h, img_w))
        pred_label_imgs = pred_label_imgs.astype(np.uint8)
        pred_label_imgs = pred_label_imgs[0]

        # Detect surface
        line = pred_label_imgs[0]
        bottom_middle = pred_label_imgs[int(0.7 * len(pred_label_imgs)):int(0.9 * len(pred_label_imgs)),
                        int(0.4 * len(line)):int(0.6 * len(line))]
        bottom_middle = bottom_middle.flatten()
        c = np.bincount(bottom_middle)
        if np.argmax(c) == 0:
            print("On road")
        elif np.argmax(c) == 0:
            print("On road")
        else:
            print("Surface unknown")

        rgb = decode_segmap(pred_label_imgs, NB_CLASSES)

        print(f"Frame {framenb}")
        cv2.imshow('rgb', rgb)
        cv2.imshow('image', image)
        cv2.imwrite(f"out/frame{framenb}.jpg", image)
        framenb += 1

    cv2.destroyAllWindows()
    cap.release()
