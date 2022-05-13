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
    '--video', '-v', type=str, default="./../video_data/cars.mp4", help='Input video path.')

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

def click_event(event, x, y, flags, params):

    if event == cv.EVENT_LBUTTONDOWN:
        click_corners.append([x,y])
        cv.putText(one_frame, str(x) + ',' +
                    str(y), (x + 10,y + 10),cv.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)
        cv.circle(one_frame,(x, y), 4, (0, 0, 255), -1)
        cv.imshow('one_frame', one_frame)

model = YOLO_v2(
    weights_path=args.weights,
    configfile_path=args.config,
    labels_path=args.labels,
    confidence_threshold=0.5,
    nms_threshold=0.2,
    draw_bboxes=True,
    use_gpu=args.gpu
)

video_path = args.video

white_plane = np.zeros([500,800,3],dtype=np.uint8)
white_plane.fill(255)
click_corners = []

cap = cv.VideoCapture(video_path)
status,one_frame = cap.read()
writer = None

cv.namedWindow('one_frame')
cv.setMouseCallback('one_frame', click_event)
cv.waitKey(1)

while(True):
    cv.imshow('one_frame', one_frame)
    one_frame = cv.resize(one_frame, (1400, 1000))
    one_frame = Transform.put_Text(one_frame)
    if cv.waitKey(20) & 0xff == 27:
        break

click_corners = np.int32(click_corners)
rl2,ll2,rt2,lt2 = corners_to_points(click_corners)
print("points2",rl2,ll2,rt2,lt2)

while True:

    ok, image = cap.read()

    if not ok:
        print("[INFO] Cannot read the video feed.")
        break
    image = cv.resize(image, (1400, 1000))

    bboxes, confidences, class_ids = model.detect(image,click_corners)

    updated_image = model.draw_bboxes(image.copy(), bboxes, confidences, class_ids)
    updated_image = Transform.draw_points(updated_image,click_corners)
    cv.imshow("updated image", updated_image)

    transformed_bboxes = Transform.get_all4points_bboxes_transformed(white_plane.copy(),bboxes,rl2,ll2,rt2,lt2)

    updated_white_plane = model.draw_bboxes(white_plane.copy(), transformed_bboxes, confidences, class_ids)
    tracks,updated_white_plane = tracker.update2(transformed_bboxes, confidences, class_ids,updated_white_plane)
    #tracks = tracker.update(transformed_bboxes, confidences, class_ids)
    updated_white_plane_and_track = draw_tracks(updated_white_plane, tracks)
    cv.imshow("updated_white_plane_and_track",updated_white_plane_and_track)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    if writer is None:
        fourcc = cv.VideoWriter_fourcc(*"MJPG")
        writer = cv.VideoWriter(args.output, fourcc, 2,
            (updated_image.shape[1], updated_image.shape[0]), True)

    writer.write(updated_image)
    print("############################################################################################")

print("[INFO] cleaning up...")
cap.release()
writer.release()
cv.destroyAllWindows()



