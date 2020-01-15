import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision import models

from decode_segmap import decode_segmap

"""
Applies semantic segmentation to a single image for testing purposes
"""

NB_CLASSES = 21
deeplabv3 = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

img = Image.open('./data/bird.png')
plt.imshow(img)
plt.show()

trf = T.Compose([T.Resize(256),
                 T.CenterCrop(224),
                 T.ToTensor(),
                 T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
inp = trf(img).unsqueeze(0)

out = deeplabv3(inp)['out']
print(out.shape)

om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()

print(om.shape)
print(np.unique(om))

rgb = decode_segmap(om, NB_CLASSES)
plt.imshow(rgb)
plt.show()
