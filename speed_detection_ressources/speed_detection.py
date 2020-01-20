import cv2
import numpy as np

from speed_detection_ressources import model
from speed_detection_ressources.frames_to_opticalFlow import convertToOptical
from speed_detection_ressources.model import CNNModel

RESIZE_INPUT = (640, 480)
MODEL_NAME = 'CNNModel_flow'
PRE_TRAINED_WEIGHTS = './speed_detection_ressources/best' + MODEL_NAME + '.h5'
model = CNNModel()
model.load_weights(PRE_TRAINED_WEIGHTS)


"""
- Applies optical flow between previous and current frame
- Predict speed with optical flow as a feature of CNN model
"""
def predict_cyclist_speed(prev_frame, current_frame):
	prev_frame = cv2.resize(prev_frame, RESIZE_INPUT)
	current_frame = cv2.resize(current_frame, RESIZE_INPUT)

	flow_image_bgr_prev1 = np.zeros_like(prev_frame)
	flow_image_bgr_prev2 = np.zeros_like(prev_frame)
	flow_image_bgr_prev3 = np.zeros_like(prev_frame)
	flow_image_bgr_prev4 = np.zeros_like(prev_frame)

	flow_image_bgr_next = convertToOptical(prev_frame, current_frame)
	flow_image_bgr = (flow_image_bgr_prev1 + flow_image_bgr_prev2 + flow_image_bgr_prev3 + flow_image_bgr_prev4 + flow_image_bgr_next) / 4

	combined_image_test = cv2.normalize(flow_image_bgr, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX,
	                                    dtype=cv2.CV_32F)
	combined_image_test = cv2.resize(combined_image_test, (0, 0), fx=0.5, fy=0.5)

	combined_image_test = combined_image_test.reshape(1, combined_image_test.shape[0], combined_image_test.shape[1],
	                                                  combined_image_test.shape[2])

	prediction = model.predict(combined_image_test)
	print("Speed detection " + str(prediction[0][0]))
