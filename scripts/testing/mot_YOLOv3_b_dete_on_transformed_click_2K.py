import cv2 as cv
import numpy as np
import time
from motrackers.detectors import YOLO_v2
#from motrackers.detectors import YOLOv3
from motrackers import CentroidTracker, CentroidKF_Tracker, SORT, IOUTracker, KFTrackerSORT, KFTracker2D
from motrackers.utils import draw_tracks
from motrackers.utils import corners_to_points
from motrackers import Transform

import argparse

parser = argparse.ArgumentParser(
    description='Object detections in input video using YOLOv3 trained on COCO dataset.'
)

parser.add_argument(
    '--video1', '-v', type=str, default="./../video_data/cars.mp4", help='Input video path.')

parser.add_argument(
    '--video2', '-vv', type=str, default="./../video_data/cars2.mp4", help='Input video path.')

parser.add_argument(
    '--output', '-o', type=str, default="./../output/prediction.avi", help='Output video path.')

parser.add_argument(
    '--weights', '-w', type=str,
    default="./../pretrained_models/yolo_weights/yolov3.weights",
    help='path to weights file of YOLOv3 (`.weights` file.)'
)

parser.add_argument(
    '--config', '-c', type=str,
    default="./../pretrained_models/yolo_weights/yolov3.cfg",
    help='path to config file of YOLOv3 (`.cfg` file.)'
)

parser.add_argument(
    '--labels', '-l', type=str,
    default="./../pretrained_models/yolo_weights/coco_names.json",
    help='path to labels file of coco dataset (`.names` file.)'
)

parser.add_argument(
    '--gpu', type=bool,
    default=False, help='Flag to use gpu to run the deep learning model. Default is `False`'
)

parser.add_argument(
    '--tracker', type=str, default='CentroidTracker',
    help="Tracker used to track objects. Options include ['CentroidTracker', 'CentroidKF_Tracker', 'SORT']")

args = parser.parse_args()

if args.tracker == 'CentroidTracker':
    tracker = CentroidTracker(max_lost=20, tracker_output_format='mot_challenge')
elif args.tracker == 'CentroidKF_Tracker':
    tracker = CentroidKF_Tracker(max_lost=20, tracker_output_format='mot_challenge')
elif args.tracker == 'KalmanFilter':
    tracker = KFTracker2D(time_step = 1)
elif args.tracker == 'SORT':
    tracker = SORT(max_lost=5, tracker_output_format='mot_challenge', iou_threshold=0)
elif args.tracker == 'IOUTracker':
    tracker = IOUTracker(max_lost=4, iou_threshold=0.5, min_detection_confidence=0.3, max_detection_confidence=0.7,
                             tracker_output_format='mot_challenge')
else:
    raise NotImplementedError



click_corners_1 = []
click_corners_2 = []

#important parameters

# threshold when 2 detections of the two cameras is evaluated as one car
combining_bboxes_threshold = 100

#which bboxes are trated as large (whith or hight)
large_bbox_threshold = 50

# increases the threshold(combining_bboxes_threshold) for large bboxes (independant of bbox_scale)
combining_margin_for_large_bboxes = 0.1

#scaling of bboxes
bbox_scale = 0.4




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

model = YOLO_v2(
    weights_path=args.weights,
    configfile_path=args.config,
    labels_path=args.labels,
    confidence_threshold=0.4,
    nms_threshold=0.2,
    draw_bboxes=True,
    use_gpu=args.gpu
)

video_path_1 = args.video1
video_path_2 = args.video2


#for IP cameras
'''
camera_IP1 = '192.168.0.11'
camera_IP2 = '192.168.0.12'
cap1 = cv.VideoCapture('http://graubeamer:Autokino22@{}/cgi-bin/mjpeg?stream=[1]'.format(camera_IP1))
cap2 = cv.VideoCapture('http://graubeamer:Autokino22@{}/cgi-bin/mjpeg?stream=[1]'.format(camera_IP2))
'''




#Initalize Plane
white_plane_1 = np.zeros([500,800,3],dtype=np.uint8)
white_plane_1.fill(255)
white_plane_2 = np.zeros([500,800,3],dtype=np.uint8)
white_plane_2.fill(255)

#Initalize Videos
cap_1 = cv.VideoCapture(video_path_1)
cap_2 = cv.VideoCapture(video_path_2)
status,one_frame_1 = cap_1.read()
status,one_frame_2 = cap_2.read()

writer = None





#choose ROI 1
cv.namedWindow('one_frame_1')
cv.setMouseCallback('one_frame_1', click_event)
cv.waitKey(1)

while(True):
    cv.imshow('one_frame_1', one_frame_1)
    one_frame_1 = cv.resize(one_frame_1, (700,500))
    one_frame_1 = Transform.put_Text(one_frame_1)
    if cv.waitKey(20) & 0xff == 27:
        #cv.destroyWindow('one_frame_1')
        cv.waitKey(1)
        break

click_corners_1 = np.int32(click_corners_1)
rl_1,ll_1,rt_1,lt_1 = corners_to_points(click_corners_1)
print("points_1",rl_1,ll_1,rt_1,lt_1)





#chosse ROI 2
cv.namedWindow('one_frame_2')
cv.setMouseCallback('one_frame_2', click_event)
cv.waitKey(1)

while(True):
    cv.imshow('one_frame_2', one_frame_2)
    one_frame_2 = cv.resize(one_frame_2, (700,500))
    one_frame_2 = Transform.put_Text_green(one_frame_2)
    if cv.waitKey(20) & 0xff == 27:
        cv.destroyWindow('one_frame_1')
        cv.destroyWindow('one_frame_2')
        cv.waitKey(1)
        break

click_corners_2 = np.int32(click_corners_2)
rl_2,ll_2,rt_2,lt_2 = corners_to_points(click_corners_2)
print("points_2",rl_2,ll_2,rt_2,lt_2)




#start Detecting
while True:

    ok_1, image_1 = cap_1.read()
    ok_2, image_2 = cap_2.read()

    if not ok_1 and ok_2:
        print("[INFO] Cannot read the video feed.")
        break
    image_1 = cv.resize(image_1, (700,500))
    image_2 = cv.resize(image_2, (700,500))
    




    #detection on image_1
    bboxes_1, confidences_1, class_ids_1 = model.detect(image_1,click_corners_1)

    updated_image_1 = model.draw_bboxes(image_1.copy(), bboxes_1, confidences_1, class_ids_1)
    updated_image_1 = Transform.draw_points_red(updated_image_1,click_corners_1)
    cv.imshow("updated image 1", updated_image_1)

    transformed_bboxes_1 = Transform.get_all4points_bboxes_transformed_with_bbox_scale(white_plane_1.copy(),bboxes_1,rl_1,ll_1,rt_1,lt_1,bbox_scale)

    white_plane_detections = Transform.draw_bboxes_red(white_plane_1.copy(),transformed_bboxes_1)
    white_plane_detections = Transform.draw_centroid_bboxes_red(white_plane_detections,transformed_bboxes_1)






    #detection on image_2
    bboxes_2, confidences_2, class_ids_2 = model.detect(image_2,click_corners_2)

    updated_image_2 = model.draw_bboxes(image_2.copy(), bboxes_2, confidences_2, class_ids_2)
    updated_image_2 = Transform.draw_points_green(updated_image_2,click_corners_2)
    cv.imshow("updated image 2", updated_image_2)

    transformed_bboxes_2 = Transform.get_all4points_bboxes_transformed_with_bbox_scale(white_plane_2.copy(),bboxes_2,rl_2,ll_2,rt_2,lt_2,bbox_scale)

    white_plane_detections = Transform.draw_centroid_bboxes_green(white_plane_detections,transformed_bboxes_2)  
    white_plane_detections = Transform.draw_bboxes_green(white_plane_detections,transformed_bboxes_2) 
    cv.imshow("white_plane_detections",white_plane_detections)





    #Iou for transformed bboxes
    combined_bboxes = Transform.selected_join(transformed_bboxes_1 ,transformed_bboxes_2 ,combining_bboxes_threshold ,large_bbox_threshold ,combining_margin_for_large_bboxes, bbox_scale)
    #combined_confidences = Transform.combine_confidences(confidences_1,confidences_2,len_combined_bboxes)
    len_combined_bboxes = len(combined_bboxes)
    combined_confidences = Transform.combine_confidences2(len_combined_bboxes)
    combined_class_ids = Transform.combine_class_ids(len_combined_bboxes)
    #combined class:ids





    #Traking on combined BBoxes
    combined_white_plane = model.draw_bboxes(white_plane_2.copy(), combined_bboxes, combined_confidences, combined_class_ids)

    cv.imshow("iii",combined_white_plane)

    tracks,combined_white_plane = tracker.update2(combined_bboxes, combined_confidences, combined_class_ids,combined_white_plane)

    #without showing prediction

    #tracks,xxx = tracker.update2(combined_bboxes, combined_confidences, combined_class_ids,white_plane_2.copy())
    #tracks = tracker.update(transformed_bboxes, confidences, class_ids)

    combined_white_plane_and_track = draw_tracks(combined_white_plane, tracks)
    cv.imshow("combined_white_plane_and_track",combined_white_plane_and_track)
    


    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    if writer is None:
        fourcc = cv.VideoWriter_fourcc(*"MJPG")
        writer = cv.VideoWriter(args.output, fourcc, 2,
            (combined_white_plane_and_track.shape[1], combined_white_plane_and_track.shape[0]), True)

    writer.write(combined_white_plane_and_track)
    print("############################################################################################")

print("[INFO] cleaning up...")
cap_1.release()
cap_2.release()
writer.release()
cv.destroyAllWindows()



