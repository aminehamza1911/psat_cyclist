import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision import models

from decode_segmap import decode_segmap

ESC_CODE = 27

deeplabv3 = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()
to_net_transform = T.Compose([T.ToPILImage(),
                              T.Resize(224),  # , 300)),
                              T.ToTensor(),
                              T.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])])

segmap_to_image_transform = T.Compose([T.ToPILImage(),
                                       T.Resize((360, 640))])

cap = cv2.VideoCapture('data/Benjamin02.avi')

# take first frame of the video
retval, image = cap.read()

framenb = 1

# Skip beginning of video
for i in range(2000):
    cap.read()

while retval and cv2.waitKey(1) != ESC_CODE:
    retval, image = cap.read()

    inp = to_net_transform(image).unsqueeze(0)
    out = deeplabv3(inp)['out']

    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    rgb = decode_segmap(om)

    image = image + np.array(segmap_to_image_transform(rgb))

    print(f"Frame {framenb}")
    cv2.imshow('frame', image)
    cv2.imwrite(f"out/frame{framenb}.jpg", image)
    framenb += 1

cv2.destroyAllWindows()
cap.release()
