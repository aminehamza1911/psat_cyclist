import numpy as np
import cv2
import PIL.ImageOps
from keras.preprocessing import image as image_utils
from keras.models import load_model
import sys


data = []
label = [0,1,2,3,4]
image=sys.argv[1]
image_affichage = cv2.imread( image , cv2.IMREAD_COLOR) # save image so we can show it later

img = image_utils.load_img(image,target_size=(100, 100))  # open an image
img = PIL.ImageOps.invert(img)  # inverts it
img = image_utils.img_to_array(img)  # converts it to array

data.append(img)

np.save("data" + ".npy",
        np.array(data))  # model root to save image models(image)


data_load = np.load("data.npy") # the input Image is in this npy file
# normalization
data_load = data_load / 255.0


# loads trained model and architecture
model = load_model("Model/trainedModelE40.h5")

# -------predicting part-------
y = model.predict_classes(data_load, verbose=0)

if y[0]== 0 :
        y='Cloudy'
if y[0]== 1 :
        y='Sunny'
if y[0]== 2 :
        y='Rainy'
if y[0]== 3 :
        y='Snowy'
if y[0]== 4 :
        y='Foggy'

# We create a window to display our image
#cv2.namedwindow("Weather Prediction , Model : trainedModel.h5 ")

# We display our image and ask the program to wait until a key is pressed


image_affichage = cv2.putText(image_affichage, 'weather : ' + y , (10, 25) , cv2.FONT_HERSHEY_SIMPLEX  ,
                   1, (255, 255, 255), 2, cv2.LINE_AA)
cv2.imshow("Weather Prediction", image_affichage)
cv2.imwrite('result.png' , image_affichage)

cv2.waitKey(0)

