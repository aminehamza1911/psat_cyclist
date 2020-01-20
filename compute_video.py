import cv2
import torch

from speed_detection_ressources.speed_detection import predict_cyclist_speed
from video_segmentation_deeplabv3 import semantic_segmentation

ESC_CODE = 27
'''
Video input path
'''
cap = cv2.VideoCapture('./data/video_trim.mp4')

'''
Apply all features extracting methods to each video frame

- Semantic segmentation
- Cyclist speed detection

'''
# take first frame of the video
retval, image = cap.read()
framenb = 1

# Skip beginning of video
# for i in range(1000):
#     cap.read()
previous_frame = image
with torch.no_grad():
	while retval and cv2.waitKey(1) != ESC_CODE:
		print(f"Frame {framenb}")
		retval, image = cap.read()

		# apply semantic segmentation
		semantic_segmentation(image, retval)

		# detect cyclist speed
		predict_cyclist_speed(image, previous_frame)
		previous_frame = image
		framenb += 1

	cv2.destroyAllWindows()
	cap.release()
