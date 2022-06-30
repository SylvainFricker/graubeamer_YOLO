from webbrowser import get
from imutils.video import VideoStream
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

islive =  False

"""
Size Settings
"""
# size of captured image, check camera pixels, should be 1280/720
frame_width, frame_height = 640, 360
frame_size = (frame_width, frame_height)

# Size of parking lot: length, width, rgb
padding = 50
length_davos, width_davos = 400, 500
plane_size_davos = [length_davos + 2*padding, width_davos + 2*padding,3]
length_parpan, width_parpan = 800, 500
plane_size_parpan = [length_parpan + 2*padding, width_parpan + 2*padding,3]

#fixed width and height for cars, in relation to plane size
w_car, h_car = 20, 40

#Initialize projection planes, one with all boxes, one with merged boxes only
white_plane_detections = np.ones(plane_size_davos, dtype=np.uint8) * 255
white_plane_merged = np.ones(plane_size_davos, dtype=np.uint8) * 255


"""
parameters for centroid transformation
"""
# max distance in pixels to reference (bottom centre)
max_dist_centroid = np.sqrt((frame_width/2)**2 + frame_height**2)
# correction factor for centroid
factor_centroid = 20
# base correction for centroid
base_corr_centroid = 15

"""
important parameters for combining boxes
"""
# threshold when 2 detections of the two cameras is evaluated as one car (pixel)
combining_centroids_dist_threshold = 100

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

""" Video Capture Settings"""

#for IP cameras
if islive:
    camera_IP1 = '192.168.0.11'
    camera_IP2 = '192.168.0.12'
    cap_1 = cv.VideoCapture('http://graubeamer:Autokino22@{}/cgi-bin/mjpeg?stream=[1]'.format(camera_IP1))
    cap_2 = cv.VideoCapture('http://graubeamer:Autokino22@{}/cgi-bin/mjpeg?stream=[1]'.format(camera_IP2))

#for videos
else:
    #video_path_1 = "./../video_input/alvaneu2_3.1.mp4"
    #video_path_2 = "./../video_input/alvaneu2_3.2.mp4"
    video_path_1 = "./../video_input/alvaneu_1.1_4fps.mp4"
    video_path_2 = "./../video_input/alvaneu_1.2_4fps.mp4"
    cap_1 = cv.VideoCapture(video_path_1)
    cap_2 = cv.VideoCapture(video_path_2)


"""
Define Area of Intrest for both cameras 
"""
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
"""
click_corners_1 = np.array([[245, 203],
       [525, 229],
       [ 77, 236],
       [486, 302]])
ul_1,ur_1,ll_1,lr_1 = Helper.corners_to_points(click_corners_1)
click_corners_2 = np.array([[113, 224],
       [404, 219],
       [126, 296],
       [570, 266]])
#upper left, upper right, lower left, lower right
ul_2,ur_2,ll_2,lr_2 = Helper.corners_to_points(click_corners_2)
print("points_2",ul_2,ur_2,ll_2,lr_2)

if islive:
    # release and renew videocapture for timing
    cap_1.release()
    cap_2.release()

    cap_1 = cv.VideoCapture('http://graubeamer:Autokino22@{}/cgi-bin/mjpeg?stream=[1]'.format(camera_IP1))
    cap_2 = cv.VideoCapture('http://graubeamer:Autokino22@{}/cgi-bin/mjpeg?stream=[1]'.format(camera_IP2))

"""Perspective Transformation Matrices"""
M1 = Transform.get_PerspectiveMatrix(ul_1,ur_1,ll_1,lr_1, plane_size_davos[1], plane_size_davos[0], padding) 
M2 = Transform.get_PerspectiveMatrix(ul_2,ur_2,ll_2,lr_2, plane_size_davos[1], plane_size_davos[0], padding)

"""Run Yolo Algorithm"""
def run_yolo():
    global outputFrame, lock
    
    # saving video
    fourcc = cv.VideoWriter_fourcc(*"MJPG")
    writer = cv.VideoWriter(video_out, fourcc, 2, (plane_size_davos[1], plane_size_davos[0]), True)

    # timing
    starttime = time.perf_counter()

    #start Detecting
    while True:
        if islive:
            starttime = time.perf_counter()
            image_1, image_2 = Helper.get_frame(cap_1, cap_2)
            frame_time = time.perf_counter() - starttime
            print(f'Frametime: {frame_time}')   
        else:
            ok1, image_1 = cap_1.read()
            ok2, image_2 = cap_2.read()

        image_1 = cv.resize(image_1, frame_size)
        image_2 = cv.resize(image_2, frame_size)

        #detection on image_1 - bbox has format [xmin,ymin,w,h]
        bboxes_1, confidences_1, class_ids_1 = model.detect(image_1,click_corners_1)
        image_1 = model.draw_bboxes(image_1, bboxes_1, confidences_1, class_ids_1)
        for box in bboxes_1:
            centroid_x = int(box[0] + box[2]/2) 
            centroid_y = int(box[1] + box[3]/2)
            cv.circle(image_1, (centroid_x, centroid_y), 4, (0, 255, 0), -1)
            new_centroid = Transform.correct_centroid([centroid_x, centroid_y], frame_width, frame_height, factor_centroid, base_corr_centroid, max_dist_centroid)
            cv.circle(image_1, (new_centroid[0], new_centroid[1]), 4, (0, 0, 255), -1)
        image_1 = Helper.draw_points_red(image_1,click_corners_1)
        cv.imshow("Image 1 with corrected centroids", image_1)

        # reset white plane and draw bboxes from camera 1
        white_plane_detections = np.ones(plane_size_davos, dtype=np.uint8) * 255
        white_plane_detections = Helper.draw_plane_bounds(white_plane_detections, padding)
        transformed_bboxes_1, plane = Transform.perspective_transform_bbox_draw(white_plane_detections.copy(), bboxes_1, M1, w_car, h_car, frame_width, frame_height, factor_centroid, base_corr_centroid, max_dist_centroid)
        #transformed_bboxes_1 = Transform.perspective_transform_bbox(bboxes_1, M1, w_car, h_car, frame_width, frame_height, factor_centroid, base_corr_centroid, max_dist_centroid)

        white_plane_detections = Helper.draw_bboxes_red(white_plane_detections, transformed_bboxes_1)
        
        #detection on image_2
        bboxes_2, confidences_2, class_ids_2 = model.detect(image_2,click_corners_2)
        image_2 = model.draw_bboxes(image_2, bboxes_2, confidences_2, class_ids_2)
        #testing centroid shifting
        for box in bboxes_2:
            centroid_x = int(box[0] + box[2]/2) 
            centroid_y = int(box[1] + box[3]/2)
            cv.circle(image_2, (centroid_x, centroid_y), 4, (0, 255, 0), -1)
            new_centroid = Transform.correct_centroid([centroid_x, centroid_y], frame_width, frame_height, factor_centroid, base_corr_centroid, max_dist_centroid)
            cv.circle(image_2, (new_centroid[0], new_centroid[1]), 4, (0, 0, 255), -1)
        image_2 = Helper.draw_points_green(image_2,click_corners_2)
        cv.imshow("Image 2 with corrected centroids", image_2)

        transformed_bboxes_2, _ = Transform.perspective_transform_bbox_draw(plane ,bboxes_2 , M2, w_car, h_car, frame_width, frame_height, factor_centroid, base_corr_centroid, max_dist_centroid)
        #transformed_bboxes_2 = Transform.perspective_transform_bbox(bboxes_2, M2, w_car, h_car, frame_width, frame_height, factor_centroid, base_corr_centroid, max_dist_centroid)

        #white_plane_detections = Helper.draw_centroid_bboxes_green(white_plane_detections,transformed_bboxes_2)  
        white_plane_detections = Helper.draw_bboxes_green(white_plane_detections,transformed_bboxes_2) 
        cv.imshow("Projected Detections", white_plane_detections)

       ## Merge bboxes
        # Greedy strategy
        combined_bboxes = Transform.selected_join(transformed_bboxes_1 ,transformed_bboxes_2, w_car, h_car, combining_centroids_dist_threshold)
        # Optimization strategy
        #combined_bboxes = Transform.selected_join_optimized(transformed_bboxes_1 ,transformed_bboxes_2, w_car, h_car, combining_centroids_dist_threshold)
        
        # Define confidence and class for tracker
        len_combined_bboxes = len(combined_bboxes)
        combined_confidences = Helper.combine_confidences(len_combined_bboxes) #set to 0.8 by default
        combined_class_ids = Helper.combine_class_ids(len_combined_bboxes)

        #Traking on combined BBoxes
        white_plane_merged = np.ones(plane_size_davos, dtype=np.uint8) * 255
        white_plane_merged = Helper.draw_plane_bounds(white_plane_merged, padding)
        white_plane_merged = model.draw_bboxes(white_plane_merged, combined_bboxes, combined_confidences, combined_class_ids)
    
        tracks = tracker.update(combined_bboxes, combined_confidences, combined_class_ids)
        #tracks, combined_white_plane = tracker.update_with_predictions(combined_bboxes, combined_confidences, combined_class_ids,combined_white_plane)

        #without showing prediction
        white_plane_merged_tracked = draw_tracks(white_plane_merged, tracks)
        cv.imshow("Tracked Merged Detections", white_plane_merged_tracked)            
        
        writer.write(white_plane_merged_tracked)
        print("####################################### \n")
                
        # acquire the lock, set the output frame, and release the lock
        with lock:
            outputFrame = white_plane_merged_tracked 

        endtime = time.perf_counter()
        print(f"Image processing time: {endtime - starttime} s \n")
        starttime = endtime

        if cv.waitKey(1) & 0xFF == ord('q'):
            break  
    
    writer.release()


t = threading.Thread(target=run_yolo)
t.daemon = True
t.start()

app.run(host="localhost", port=8000, debug=True, threaded=True, use_reloader=False)

print("[INFO] cleaning up...")
cap_1.release()
cap_2.release()

cv.destroyAllWindows()



