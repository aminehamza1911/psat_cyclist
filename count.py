# USAGE
# python count.py --input cyclist.avi --output output/cyclist_output.avi --yolo yolo-coco

# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2 
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
    help="path to input video")
ap.add_argument("-o", "--output", required=True,
    help="path to output video")
ap.add_argument("-y", "--yolo", required=True,
    help="Base for Yolo ")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability ")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
    help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

classname = []
list_of_vehicles = ["car","person","motorbike","bus","truck"]

def get_vehicle_count(boxes, classnames):
    total_vehicle = 0 
    dict_vehicle = {} 
    for i in range(len(boxes)):
        classnames = classnames[i]
        if(classnames in list_of_vehicles):
            total_vehicle += 1
            dict_vehicle[classnames] = dict_vehicle.get(classnames,0) + 1
    return total_vehicle, dict_vehicle

# load the COCO class labels which are our model is trained on
labelsPath = os.path.sep.join([args["yolo"], "./coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "./yolov3-spp.weights"])
configPath = os.path.sep.join([args["yolo"], "./yolov3.cfg"])


# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
#video_src = "D:/Downloads/Benjamin.avi"
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

# try to determine the total number of frames in the video file
try:
    if imutils.is_cv2():
        tot = int(vs.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)) 
    else:
        tot = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print("[INFO] {} total frames in video".format(tot))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    tot = -1

# loop over frames from the video file stream

list_of_vehicles = ["car","person","bus","motorbike","truck"]

while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()

    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]
        

    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
    swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
    count_warning=0
    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > args["confidence"]:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                classname.append(LABELS[classID])

    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],args["threshold"])
    
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
        # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            
            
            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
                #        person                   car                  motorbike            bus              truck         
            if classIDs[i] == 0 or classIDs[i] == 2 or classIDs[i] == 3 or classIDs[i] == 5 or classIDs[i]== 7:
                mid_y = (boxes[i][1]+boxes[i][3])/2
                mid_x = ((boxes[i][0]+boxes[i][2])/2)/1000
                apx_distance = (boxes[i][3] - boxes[i][1])/10
                text = "{} {:.4f} ".format(LABELS[classIDs[i]],confidences[i])
                cv2.putText(frame, text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                #print(apx_distance)
                #print(mid_x)
                if (apx_distance) <=0.2 and (apx_distance>=0.0):
                    if (mid_x) > 0.2 and (mid_x < 0.35) :
                        count_warning += 1 #count warnings in creepage ligne
                        
            

    # check if the video writer is None

    total_vehicles, each_vehicle = get_vehicle_count(boxes, classname)
    print("Serious Warnings ", count_warning)
    print("We Detect", total_vehicles)
    print("Each Object Count in video", each_vehicle)
    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30,
        (frame.shape[1], frame.shape[0]), True)


        # some information on processing single frame
        if total > 0:
            elap = (end - start)
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(elap * total))

    # write the output frame to disk
    writer.write(frame)

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()
