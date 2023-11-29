# main program file - Social Distancing Detector

# imports
from scipy.spatial import distance as dist
import numpy as np
import imutils
import cv2
import time
from imutils.video import FPS
from pygame import mixer

### SECTION 1

# initialize minimum probability to filter weak detections along with the
# threshold when applying non-maxim suppression
MIN_CONF = 0.3
NMS_THRESH = 0.3

# Set the threshold value for total violations limit.
Threshold_count = 1

# should NVIDIA CUDA GPU be used?
USE_GPU = False
# having Nvidia CUDA GPU & if OpenCV is installed with NVIDIA GPU support
# USE_GPU = True

# define the minimum safe distance (in pixels) that two people can be from each other
MAX_DISTANCE = 80
MIN_DISTANCE = 50


### SECTION 2

# function to detect people / human object only
def detect_people(frame, net, ln, personIdx=0):
    # grab dimensions of the frame and initialize the list of results
    (H, W) = frame.shape[:2]
    results = []

    # construct a blob from the input frame and then perform a forward pass
    # of the YOLO object detector, giving us the bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    # initialize lists of detected bounding boxes, centroids, and confidence
    boxes = []
    centroids = []
    confidences = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence(probability) of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter detections by (1) ensuring that the object detected was a person and
            # (2) that the minimum confidence is met
            if classID == personIdx and confidence > MIN_CONF:
                # scale the bounding box coordinates back relative to the size of
                # the image, keeping in mind that YOLO actually returns the center (x, y)-coordinates
                # of the bounding box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                # get the center coordinates plus height and width of bounding boxes
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x,y)-coordinates to derive the top and left corner of
                # the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update the list of bounding box coordinates, centroids and confidences
                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))

                print()
                print("Confidence :-", len(confidences))
                print("Result :-", confidences)
                print()

    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

    human_count = len(idxs)

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes being kept
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # update the results list to consist of the person prediction probability,
            # bounding box coordinates, and the centroid
            r = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(r)

    # return the list of results
    return results, human_count


### SECTION 3

# coco labels path
coco_label = r'yolo-coco\coco.names'

# coco labels are read one by one in each line & stored in a list
labels = open(coco_label).read().strip().split('\n')

## choose any one model: yolov3 or yolov3-tiny

# yolov3 weights and yolov3 config file path
yolo_weight = r'yolo-coco\yolov3.weights'
yolo_config = r'yolo-coco\yolov3.cfg'

# yolov3-tiny weights and yolov3-tiny config file path
yolo_tiny_weight = r'yolo-coco\yolov3_tiny.weights'
yolo_tiny_config = r'yolo-coco\yolov3_tiny.cfg'

# load the YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")

net = cv2.dnn.readNetFromDarknet(yolo_config, yolo_weight)
# net = cv2.dnn.readNetFromDarknet(yolo_tiny_config, yolo_tiny_weight)

# check if GPU is to be used or not
if USE_GPU:
    # set CUDA s the preferable backend and target
    print("[INFO] setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# determine only the "output" layer names that we need from YOLO
ln = net.getLayerNames()

# for python <= 3.7 or open_cv nvidia cuda build, choose this
# ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# for python > 3.7, choose this
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

### SECTION 4

# initialize the video stream and pointer to output video file
print("[INFO] accessing video stream...")
print()

# set the input video path according to your directory
vid_path = r'.\pedestrians.mp4'

# new path for recording output
new_path = r'.\test_output.avi'

# capture the video file / source
cap = cv2.VideoCapture(vid_path)
# choose this by setting video path to '0' for live webcam access
# cap = cv2.VideoCapture(0)

if cap.isOpened() is False:
    print("[Exiting]: Error accessing video source.")
    exit(0)

# printing some captured input video's properties

# in case of live webcam footage, frame count should not be selected
print("Input Video frame count :-", cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Input Video fps :-", cap.get(cv2.CAP_PROP_FPS))
print(f"Input Video dimensions :- ({cap.get(cv2.CAP_PROP_FRAME_WIDTH)},"
      f" {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)})")
print()

### SECTION 5

# initial video writer status
video_writer = None

# initial number of processed frames from input stream
num_frame_processed = 0

# set program start time
start_time = time.time()

# set the starting FPS detector, imutils library is required for this
fps = FPS().start()

# set the warning / alert sound, pygame module is required for this
mixer.init()
mixer.music.load(r'./alarm.mp3')
# intially
music_on = False

# resized frame height and width for output video
set_width = 800
set_height = 450

# video processing loop
# loop over the frames from the video stream
while True:
    # start reading frames
    grabbed, frame = cap.read()

    if grabbed is False:
        print('[Exiting] No more frames to read')
        print()
        break

    num_frame_processed += 1
    # loop / epoch time
    loop_start_time = time.time()

    # resize the frame and then detect people (only people) in it

    # choose this if imutils is available
    frame = imutils.resize(frame, width=set_width)

    # choose this if imutils is not available
    # frame = cv2.resize(frame, (set_width, set_height), interpolation=cv2.INTER_AREA)

    # creating an extra blank black frame for UI
    frame_temp = np.zeros((set_height + 155, set_width, 3), np.uint8)

    # make the temp frame complete white
    white_col = (255, 255, 255)
    frame_temp[:] = white_col

    # fit in original frame acc. to size
    frame_temp[:set_height, :set_width] = frame

    # only use the 'person' label configuration from yolo config
    results, human_count = detect_people(frame, net, ln, personIdx=labels.index("person"))

    # initialize the set of indexes that violate/trigger the minimum social distance
    violate = set()
    warning = set()

    # ensure there are at least two people detections (required in order to compute
    # the pairwise distance maps)
    if len(results) >= 2:
        # extract all centroids from the results and compute the Euclidean distances
        # between all pairs of the centroids
        centroids = np.array([r[2] for r in results])
        D = dist.cdist(centroids, centroids, metric="euclidean")

        # loop over the upper triangular of the distance matrix
        for i in range(0, D.shape[0]):
            for j in range(i + 1, D.shape[1]):
                # check to see if the distance between any two centroid pairs is less
                # than the configured number of pixels
                if D[i, j] < MIN_DISTANCE:
                    # update the violation set with the indexes of the centroid pairs
                    violate.add(i)
                    violate.add(j)
                # update our abnormal set if the centroid distance is below max distance limit
                if (D[i, j] < MAX_DISTANCE) and not violate:
                    warning.add(i)
                    warning.add(j)

    # loop over the results
    for (i, (prob, bbox, centroid)) in enumerate(results):
        # extract the bounding box and centroid coordinates, then initialize the color of the annotation
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        color = (0, 255, 0)

        # if the index pair exists within the violation/warning sets, then update the color
        if i in violate:
            color = (0, 0, 255)
        elif i in warning:
            color = (0, 255, 255)

        # draw (1) a bounding box around the person and (2) the centroid coordinates of the person
        cv2.rectangle(frame_temp, (startX, startY), (endX, endY), color, 2)
        cv2.circle(frame_temp, (cX, cY), 5, color, 1)

    # main UI part:
    # we will draw some text and UI components to appear along-with output video
    cv2.line(frame_temp, (0, set_height - 1), (set_width, set_height - 1), (0, 0, 0), 2)
    cv2.line(frame_temp, (0, set_height + 35), (set_width, set_height + 35), (0, 0, 0), 1)
    cv2.line(frame_temp, (0, set_height + 110), (set_width, set_height + 110), (0, 0, 0), 1)

    # draw the outcome parameters on the frame
    total_count = "TOTAL COUNT: {}".format(human_count)
    cv2.putText(frame_temp, total_count, (30, set_height + 25),
                cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)

    no_risk = "SAFE: {}".format(int(human_count) - int(len(violate)) - int(len(warning)))
    cv2.putText(frame_temp, no_risk, (280, set_height + 25),
                cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 170, 0), 2)

    low_risk = "LOW RISK: {}".format(len(warning))
    cv2.putText(frame_temp, low_risk, (430, set_height + 25),
                cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 120, 255), 2)

    high_risk = "HIGH RISK: {}".format(len(violate))
    cv2.putText(frame_temp, high_risk, (620, set_height + 25),
                cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 150), 2)

    Safe_Distance = "Safety Distance: > {} px".format(MAX_DISTANCE)
    cv2.putText(frame_temp, Safe_Distance, (30, set_height + 60),
                cv2.FONT_HERSHEY_COMPLEX, 0.60, (255, 0, 0), 2)

    Threshold = "Safety Limit: {}".format(Threshold_count)
    cv2.putText(frame_temp, Threshold, (30, set_height + 95),
                cv2.FONT_HERSHEY_COMPLEX, 0.60, (255, 0, 0), 2)

    cv2.putText(frame_temp, "Social Distancing Monitor", (180, set_height + 140),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

    # Violation alert part
    if len(violate) >= Threshold_count:
        cv2.putText(frame_temp, "!! ALERT !! ", (440, set_height + 85), cv2.FONT_HERSHEY_COMPLEX,
                    1.20, (0, 0, 255), 2)

        if music_on is False:
            mixer.music.play(loops=50)
            music_on = True
        else:
            mixer.music.unpause()

    if len(violate) < Threshold_count and music_on is True:
        mixer.music.pause()

    # display the output frames
    cv2.imshow("Social Distancing Monitor", frame_temp)

    # delay time between frames and 'q' for aborting the process
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # create the video_writer one time only
    if video_writer is None:
        # provide with base 25 fps -> for my setup, if fps < 25
        # record output in altered dimensions
        video_writer = cv2.VideoWriter(new_path,
                                       cv2.VideoWriter_fourcc(*"MJPG"),
                                       25,
                                       (frame_temp.shape[1], frame_temp.shape[0]),
                                       True)

    # if already created then write down the edited frames
    if video_writer is not None:
        print("[INFO] writing stream to output")
        video_writer.write(frame_temp)

    print("Processed Frame No. :-", num_frame_processed)

    # keep on updating the fps object
    fps.update()

    # loop end time
    loop_end_time = time.time()

    print(f"Frame {num_frame_processed - 1} time :-", loop_end_time - loop_start_time)

# stop the fps counter object
fps.stop()

# set the ending time here
end_time = time.time()

# end cleanup
video_writer.release()
cap.release()
# closing all windows
cv2.destroyAllWindows()

### SECTION 6

print()
print("Total No. of frames processed :-", num_frame_processed)
print("Elapsed time measured :-", fps.elapsed())
print("FPS measured:-", fps.fps())

# net time taken for loop of frames in seconds
net_time = end_time - start_time

print()
print("Time taken :-", net_time)
