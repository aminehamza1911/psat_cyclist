### Installation
Install python3.
 
Install the following packages:

```shell script
pip3 install pytorch
pip3 install opencv-python
pip3 install colour
```

### Running

```shell script
python3 video_segmentation_deeplabv3.py
python3 video_segmentation_pytorch.py
python3 image_segmentation_pytorch.py
```

### Video link
Download https://drive.google.com/open?id=1kE7rJBlrLw-QAS3XMNGeUbxYjPUKJjme

Place video in *data* folder. 

### Semantic segmentation models
`pytorch` : Default deeplabv3 that comes with Pytorch (problem: no road or sidewalk)

`LEDNet`: Very fast semantic segmentation (problem: downsizes our already small videos too much)

`Deeplabv3`: Fregu856 implementation trained on cityscapes: https://github.com/fregu856/deeplabv3too (problem: slow)

### Weather 

```shell script
python Weather.py idontknow.jpg
```
Will add  the video functionality ( Can  only read jpg images )
The directory ' Model' contains the different pre-trained Ml models
U will need Keras - open cv
Lien : https://drive.google.com/file/d/1HFyAUvnkS61Xat9cUBAhG-4hvwR1T8lb/view
https://github.com/berkgulay/WeatherPredictionFromImage
