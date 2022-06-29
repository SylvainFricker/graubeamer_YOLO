from flask import Response
from flask import Flask
from flask import render_template
import threading
import cv2 as cv
import numpy as np
import time
from motrackers.detectors import YOLO_v3_pimped
from motrackers import CentroidTracker, CentroidKF_Tracker
from motrackers.utils import draw_tracks
from motrackers.utils import corners_to_points
from motrackers.utils import get_PerspectiveMatrix
from motrackers import Transform
from motrackers import Helper


"""
Settings
"""
weights = "./../pretrained_models/yolo_weights/yolov3.weights"
configs = "./../pretrained_models/yolo_weights/yolov3.cfg"
labels = "./../pretrained_models/yolo_weights/coco_names.json"
video_out = "./../video_output/liveDavos_130722.avi"
# Tracker max_lost: after how many "fail to assign detection to track" the track is deleted
tracker = CentroidTracker(max_lost=20, tracker_output_format='mot_challenge')
#tracker = CentroidKF_Tracker(max_lost=20, tracker_output_format='mot_challenge')
isgpu = False

""" 
Video Capture Settings
"""

camera_IP1 = '192.168.0.11'
camera_IP2 = '192.168.0.12'
cap_1 = cv.VideoCapture('http://graubeamer:Autokino22@{}/cgi-bin/mjpeg?stream=[1]'.format(camera_IP1))
cap_2 = cv.VideoCapture('http://graubeamer:Autokino22@{}/cgi-bin/mjpeg?stream=[1]'.format(camera_IP2))

#buffer size does not work
#cap_1.set(cv.CAP_PROP_BUFFERSIZE, 1)
#cap_2.set(cv.CAP_PROP_BUFFERSIZE, 1)

"""
Size Settings
"""
# size of captured image, check camera pixels, should be 1280/720
frame_width, frame_height = 640, 360
frame_size = (frame_width, frame_height)

# Size of parking lot
plane_size_davos = [500,800,3]
plane_size_parpan = [500,800,3]

#fixed width and height for cars, in relation to plane size
w_car, h_car = 20, 40

#Initialize projection planes, one with all boxes, one with merged boxes only
white_plane_1 = np.zeros(plane_size_davos, dtype=np.uint8)
white_plane_1.fill(255)
white_plane_2 = np.zeros(plane_size_davos, dtype=np.uint8)
white_plane_2.fill(255)

"""
parameters for centroid transformation
"""
# max distance in pixels to reference (bottom centre)
max_dist = np.sqrt((frame_width/2)**2 + frame_height**2)
# correction factor for centroid
factor_centroid = 0.1
base_corr_centroid = 20

"""
important parameters for combining boxes
"""
# threshold when 2 detections of the two cameras is evaluated as one car (pixel)
combining_bboxes_threshold = 100

#which bboxes are trated as large (whith or hight)
large_bbox_threshold = 50

# increases the threshold(combining_bboxes_threshold) for large bboxes (independant of bbox_scale)
combining_margin_for_large_bboxes = 0.1

#scaling of bboxes
bbox_scale = 0.4

"""
streaming
"""
# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
outputFrame = None
lock = threading.Lock()
# initialize a flask object
app = Flask(__name__)

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")

def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock
	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue
			# encode the frame in JPEG format
			(flag, encodedImage) = cv.imencode(".jpg", outputFrame)
			# ensure the frame was successfully encoded
			if not flag:
				continue
		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")


"""
initialize models
params:
 confidence_threshold -> 0.35 - 0.5
"""

model = YOLO_v3_pimped(
    weights_path = weights,
    configfile_path = configs,
    labels_path = labels,
    confidence_threshold = 0.4,
    nms_threshold = 0.2,
    draw_bboxes = True,
    use_gpu = isgpu
)


"""
Define Area of Intrest for both cameras 
"""

def click_event(event, x, y, flags, params):

    if event == cv.EVENT_LBUTTONDOWN:
        click_corners_1.append([x,y])
        cv.putText(one_frame_1, str(x) + ',' +
                    str(y), (x + 10,y + 10),cv.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)
        cv.circle(one_frame_1,(x, y), 4, (0, 0, 255), -1)
        cv.imshow('one_frame_1', one_frame_1)

    if event == cv.EVENT_RBUTTONDOWN:
        click_corners_2.append([x,y])
        cv.putText(one_frame_2, str(x) + ',' +
                    str(y), (x + 10,y + 10),cv.FONT_HERSHEY_SIMPLEX,
                    1, (120, 255, 2), 2)
        cv.circle(one_frame_2,(x, y), 4, (120, 255, 2), -1)
        cv.imshow('one_frame_2', one_frame_2)

click_corners_1 = []
click_corners_2 = []

status,one_frame_1 = cap_1.read()
status,one_frame_2 = cap_2.read()

#choose ROI 1
cv.namedWindow('one_frame_1')
cv.setMouseCallback('one_frame_1', click_event)
cv.waitKey(1)

while(True):
    cv.imshow('one_frame_1', one_frame_1)
    one_frame_1 = cv.resize(one_frame_1, frame_size)
    one_frame_1 = Helper.put_Text_red(one_frame_1)
    if cv.waitKey(20) & 0xff == 27:
        #cv.destroyWindow('one_frame_1')
        cv.waitKey(1)
        break

click_corners_1 = np.int32(click_corners_1)
#upper left, upper right, lower left, lower right
ul_1,ur_1,ll_1,lr_1 = corners_to_points(click_corners_1)
print("points_1",ul_1,ur_1,ll_1,lr_1)

#chosse ROI 2
cv.namedWindow('one_frame_2')
cv.setMouseCallback('one_frame_2', click_event)
cv.waitKey(1)

while(True):
    cv.imshow('one_frame_2', one_frame_2)
    one_frame_2 = cv.resize(one_frame_2, frame_size)
    one_frame_2 = Helper.put_Text_green(one_frame_2)
    if cv.waitKey(20) & 0xff == 27:
        cv.destroyWindow('one_frame_1')
        cv.destroyWindow('one_frame_2')
        cv.waitKey(1)
        break

click_corners_2 = np.int32(click_corners_2)
ul_2,ur_2,ll_2,lr_2 = corners_to_points(click_corners_2)
print("points_2",ul_2,ur_2,ll_2,lr_2)


# release and renew videocapture for timing
cap_1.release()
cap_2.release()

cap_1 = cv.VideoCapture('http://graubeamer:Autokino22@{}/cgi-bin/mjpeg?stream=[1]'.format(camera_IP1))
cap_2 = cv.VideoCapture('http://graubeamer:Autokino22@{}/cgi-bin/mjpeg?stream=[1]'.format(camera_IP2))

"""Perspective Transformation Matrices"""
M1 = get_PerspectiveMatrix(ul_1,ur_1,ll_1,lr_1) #why not dependent on plane?
M2 = get_PerspectiveMatrix(ul_2,ur_2,ll_2,lr_2)

"""Run Yolo Algorithm"""
def run_yolo():
    global cap_1, cap_2, outputFrame, lock
    # saving video
    writer = None

    #start Detecting
    while True:
        # get frame with cleared buffer
        image_1, image_2 = Helper.get_frame(cap_1, cap_2)

        image_1 = cv.resize(image_1, frame_size)
        image_2 = cv.resize(image_2, frame_size)

        #detection on image_1 - bbox has format [xmin,ymin,w,h]
        bboxes_1, confidences_1, class_ids_1 = model.detect(image_1,click_corners_1)
        image_1 = model.draw_bboxes(image_1, bboxes_1, confidences_1, class_ids_1)
        for box in bboxes_1:
            centroid_x = int(box[0] + box[2]/2) 
            centroid_y = int(box[1] + box[3]/2)
            cv.circle(image_1, (centroid_x, centroid_y), 4, (0, 255, 0), -1)
            new_centroid = Transform.correct_centroid([centroid_x, centroid_y], image_1.shape[0], image_1.shape[1])
            cv.circle(image_1, (new_centroid[0], new_centroid[1]), 4, (0, 0, 255), -1)
        image_1 = Helper.draw_points_red(image_1,click_corners_1)
        cv.imshow("updated image 1", image_1)

        #transformed_bboxes_1 = Transform.get_all4points_bboxes_transformed_with_bbox_scale(white_plane_1.copy(), bboxes_1, M1, bbox_scale)
        transformed_bboxes_1, plane = Transform.get_all4points_bboxes_transformed_with_bbox_scale(white_plane_1.copy(), bboxes_1, M1, bbox_scale, image_1)
        transformed_bboxes_1 = Transform.perspective_transform_bbox(bboxes_1, M1, w_car, h_car, frame_width, frame_height)

        white_plane_detections = Helper.draw_bboxes_red(white_plane_1.copy(), transformed_bboxes_1)
        #white_plane_detections = Helper.draw_centroid_bboxes_red(white_plane_detections, transformed_bboxes_1)


        #detection on image_2
        bboxes_2, confidences_2, class_ids_2 = model.detect(image_2,click_corners_2)

        image_2 = model.draw_bboxes(image_2, bboxes_2, confidences_2, class_ids_2)
        #testing centroid shifting
        for box in bboxes_2:
            centroid_x = int(box[0] + box[2]/2) 
            centroid_y = int(box[1] + box[3]/2)
            cv.circle(image_2, (centroid_x, centroid_y), 4, (0, 255, 0), -1)
            new_centroid = Transform.correct_centroid([centroid_x, centroid_y], image_2.shape[0], image_2.shape[1])
            cv.circle(image_2, (new_centroid[0], new_centroid[1]), 4, (0, 0, 255), -1)
        image_2 = Helper.draw_points_green(image_2,click_corners_2)
        cv.imshow("updated image 2", image_2)

        #transformed_bboxes_2 = Transform.get_all4points_bboxes_transformed_with_bbox_scale(white_plane_2.copy() ,bboxes_2 ,M2 ,bbox_scale)
        transformed_bboxes_2, plane = Transform.get_all4points_bboxes_transformed_with_bbox_scale(plane ,bboxes_2 ,M2 ,bbox_scale, image_2)
        #transformed_bboxes_2 = Transform.perspective_transform_bbox(bboxes_2, M2, image_2.shape[0], image_2.shape[1])

        #white_plane_detections = Helper.draw_centroid_bboxes_green(white_plane_detections,transformed_bboxes_2)  
        white_plane_detections = Helper.draw_bboxes_green(white_plane_detections,transformed_bboxes_2) 
        cv.imshow("white_plane_detections",white_plane_detections)

        # Merge boxes
        #Iou for transformed bboxes
        combined_bboxes = Transform.selected_join(transformed_bboxes_1 ,transformed_bboxes_2 ,combining_bboxes_threshold ,large_bbox_threshold ,combining_margin_for_large_bboxes, bbox_scale)
        len_combined_bboxes = len(combined_bboxes)
        combined_confidences = Transform.combine_confidences2(len_combined_bboxes) #set to 0.8 by default
        combined_class_ids = Transform.combine_class_ids(len_combined_bboxes)
        

        #Traking on combined BBoxes
        combined_white_plane = model.draw_bboxes(white_plane_2.copy(), combined_bboxes, combined_confidences, combined_class_ids)
        
        tracks = tracker.update(combined_bboxes, combined_confidences, combined_class_ids)
        #tracks, combined_white_plane = tracker.update_with_predictions(combined_bboxes, combined_confidences, combined_class_ids,combined_white_plane)

        #without showing prediction
        combined_white_plane_and_track = draw_tracks(combined_white_plane, tracks)
        cv.imshow("combined_white_plane_and_track",combined_white_plane_and_track)            
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        if writer is None:
            fourcc = cv.VideoWriter_fourcc(*"MJPG")
            writer = cv.VideoWriter(video_out, fourcc, 2,
                (combined_white_plane_and_track.shape[1], combined_white_plane_and_track.shape[0]), True)

        writer.write(combined_white_plane_and_track)
        print("############################################################################################")
        
        # acquire the lock, set the output frame, and release the
		# lock
        with lock:
            outputFrame = combined_white_plane_and_track.copy()  
    
    writer.release()


t = threading.Thread(target=run_yolo)
t.daemon = True
t.start()

app.run(host="localhost", port=8000, debug=True, threaded=True, use_reloader=False)

print("[INFO] cleaning up...")
cap_1.release()
cap_2.release()

cv.destroyAllWindows()



