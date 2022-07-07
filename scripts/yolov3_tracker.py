import cv2 as cv
import numpy as np
import time
from motrackers.detectors import YOLO_v3_pimped
from motrackers import CentroidTracker, CentroidKF_Tracker
from motrackers.utils import draw_tracks
from motrackers import Transform
from motrackers import Helper

class YOLOv3_Tracker():

    def __init__(self, isgpu = False, islive = True):

        """ Configs """
        self.weights = "./../pretrained_models/yolo_weights/yolov3.weights"
        self.configs = "./../pretrained_models/yolo_weights/yolov3.cfg"
        self.labels = "./../pretrained_models/yolo_weights/coco_names.json"
        # Tracker max_lost: after how many "fail to assign detection to track" the track is deleted
        self.tracker = CentroidTracker(max_lost=20, tracker_output_format='mot_challenge')
        #self.tracker = CentroidKF_Tracker(max_lost=20, tracker_output_format='mot_challenge')
        self.isgpu = isgpu

        self.islive =  islive

        """ Size Settings """
        # size of captured image, check camera pixels, should be 1280/720
        self.frame_width = 640
        self.frame_height = 360
        self.frame_size = (self.frame_width, self.frame_height)

        # Size of parking lot: length, width, rgb
        self.padding = 50
        ## DAVOS
        self.length = 400
        self.width = 500
        self.plane_size = [self.length + 2*self.padding, self.width + 2*self.padding, 3]
        ## PARPAN
        #self.length = 800
        #self.width = 500
        #self.plane_size = [self.length + 2*self.padding, self.width + 2*self.padding, 3]

        #fixed width and height for cars, in relation to plane size
        self.w_car = 20
        self.h_car = 40

        #Initialize projection planes, one with all boxes, one with merged boxes only
        self.white_plane_detections = np.ones(self.plane_size, dtype=np.uint8) * 255
        self.white_plane_merged = np.ones(self.plane_size, dtype=np.uint8) * 255

        #allocate space for transformation matrices
        self.M1 = np.zeros((3,3))
        self.M2 = np.zeros((3,3))

        """ parameters for centroid transformation """
        # max distance in pixels to reference (bottom centre)
        self.max_dist_centroid = np.sqrt((self.frame_width/2)**2 + self.frame_height**2)
        # correction factor for centroid
        self.factor_centroid = 20
        # base correction for centroid
        self.base_corr_centroid = 15

        """ parameters for combining boxes """
        # threshold when 2 detections of the two cameras is evaluated as one car (pixel)
        self.combining_centroids_dist_threshold = 100

        """ initialize model """
        self.model = YOLO_v3_pimped(
            weights_path = self.weights,
            configfile_path = self.configs,
            labels_path = self.labels,
            confidence_threshold = 0.4, #choose between 0.35 - 0.5
            nms_threshold = 0.2,
            draw_bboxes = True,
            use_gpu = self.isgpu
        )

        """ initialize corners list and numpy array """
        self.click_corners_1 = []
        self.click_corners_2 = []
        self.click_corners_1_np = np.zeros((4, 2))
        self.click_corners_2_np = np.zeros((4, 2))

    def get_plane_size(self):
        return self.plane_size

    """ Define Area of Intrest for both cameras """
    def click_event(self, event, x, y, frame_1, frame_2, flags, params):

        if event == cv.EVENT_LBUTTONDOWN:
            self.click_corners_1.append([x,y])
            cv.putText(frame_1, str(x) + ',' +
                        str(y), (x + 10,y + 10),cv.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)
            cv.circle(frame_1,(x, y), 4, (0, 0, 255), -1)
            cv.imshow('frame_1', frame_1)

        if event == cv.EVENT_RBUTTONDOWN:
            self.click_corners_2.append([x,y])
            cv.putText(frame_2, str(x) + ',' +
                        str(y), (x + 10,y + 10),cv.FONT_HERSHEY_SIMPLEX,
                        1, (120, 255, 2), 2)
            cv.circle(frame_2,(x, y), 4, (120, 255, 2), -1)
            cv.imshow('frame_2', frame_2)

    def choose_ROI(self, cap_1, cap_2):
    
        status,frame_1 = cap_1.read()
        status,frame_2 = cap_2.read()

        #choose ROI 1
        cv.namedWindow('frame_1')
        cv.setMouseCallback('frame_1', self.click_event)
        cv.waitKey(1)

        while(True):
            cv.imshow('frame_1', frame_1)
            frame_1 = cv.resize(frame_1, self.frame_size)
            frame_1 = Helper.put_Text_red(frame_1)
            if cv.waitKey(20) & 0xff == 27:
                #cv.destroyWindow('frame_1')
                cv.waitKey(1)
                break

        self.click_corners_1_np = np.int32(self.click_corners_1)
        #upper left, upper right, lower left, lower right
        ul_1,ur_1,ll_1,lr_1 = Helper.corners_to_points(self.click_corners_1_np)
        # print("points_1",ul_1,ur_1,ll_1,lr_1)

        #chosse ROI 2
        cv.namedWindow('frame_2')
        cv.setMouseCallback('frame_2', self.click_event)
        cv.waitKey(1)

        while(True):
            cv.imshow('frame_2', frame_2)
            frame_2 = cv.resize(frame_2, self.frame_size)
            frame_2 = Helper.put_Text_green(frame_2)
            if cv.waitKey(20) & 0xff == 27:
                cv.destroyWindow('frame_1')
                cv.destroyWindow('frame_2')
                cv.waitKey(1)
                break

        self.click_corners_2_np = np.int32(self.click_corners_2)
        #print("points_2",ul_2,ur_2,ll_2,lr_2)
        ul_2,ur_2,ll_2,lr_2 = Helper.corners_to_points(self.click_corners_2_np)

        return ul_1,ur_1,ll_1,lr_1,ul_2,ur_2,ll_2,lr_2

    def fix_ROI(self):
        self.click_corners_1_np = np.array([[245, 203],
                                        [525, 229],
                                        [ 77, 236],
                                        [486, 302]])
        ul_1,ur_1,ll_1,lr_1 = Helper.corners_to_points(self.click_corners_1_np)
        #print("points_1:" ul_1, ur_1, ll_1, lr_2)
        self.click_corners_2_np = np.array([[113, 224],
                                        [404, 219],
                                        [126, 296],
                                        [570, 266]])
        #upper left, upper right, lower left, lower right
        ul_2,ur_2,ll_2,lr_2 = Helper.corners_to_points(self.click_corners_2_np)
        #print("points_2",ul_2,ur_2,ll_2,lr_2)

        return ul_1,ur_1,ll_1,lr_1,ul_2,ur_2,ll_2,lr_2

    def set_transformation_Matrices(self,ul_1,ur_1,ll_1,lr_1,ul_2,ur_2,ll_2,lr_2):
        self.M1 = Transform.get_PerspectiveMatrix(ul_1,ur_1,ll_1,lr_1, self.plane_size[1], self.plane_size[0], self.padding) 
        self.M2 = Transform.get_PerspectiveMatrix(ul_2,ur_2,ll_2,lr_2, self.plane_size[1], self.plane_size[0], self.padding)
   

    """Run Yolo Algorithm"""
    def run_tracker(self, writer, cap_1, cap_2):
        # timing
        starttime = time.perf_counter()

        if self.islive:
            image_1, image_2 = Helper.get_frame(cap_1, cap_2)
        else:
            ok1, image_1 = cap_1.read()
            ok2, image_2 = cap_2.read()

        image_1 = cv.resize(image_1, self.frame_size)
        image_2 = cv.resize(image_2, self.frame_size)

        #detection on image_1 - bbox has format [xmin,ymin,w,h]
        bboxes_1, confidences_1, class_ids_1 = self.model.detect(image_1, self.click_corners_1_np)
        image_1 = self.model.draw_bboxes(image_1, bboxes_1, confidences_1, class_ids_1)
        for box in bboxes_1:
            centroid_x = int(box[0] + box[2]/2) 
            centroid_y = int(box[1] + box[3]/2)
            cv.circle(image_1, (centroid_x, centroid_y), 4, (0, 255, 0), -1)
            new_centroid = Transform.correct_centroid([centroid_x, centroid_y], 
                                                       self.frame_width, 
                                                       self.frame_height, 
                                                       self.factor_centroid, 
                                                       self.base_corr_centroid, 
                                                       self.max_dist_centroid)
            cv.circle(image_1, (new_centroid[0], new_centroid[1]), 4, (0, 0, 255), -1)
        
        image_1 = Helper.draw_points_red(image_1, self.click_corners_1_np)
        cv.imshow("Image 1 with corrected centroids", image_1)

        # reset white plane and draw bboxes from camera 1
        white_plane_detections = np.ones(self.plane_size, dtype=np.uint8) * 255
        white_plane_detections = Helper.draw_plane_bounds(white_plane_detections, self.padding)
        transformed_bboxes_1, plane = Transform.perspective_transform_bbox_draw(white_plane_detections.copy(), 
            bboxes_1, self.M1, self.w_car, self.h_car, self.frame_width, self.frame_height, 
            self.factor_centroid, self.base_corr_centroid, self.max_dist_centroid)
        #transformed_bboxes_1 = Transform.perspective_transform_bbox(bboxes_1, M1, w_car, h_car, frame_width, frame_height, factor_centroid, base_corr_centroid, max_dist_centroid)

        white_plane_detections = Helper.draw_bboxes_red(white_plane_detections, transformed_bboxes_1)

        #detection on image_2
        bboxes_2, confidences_2, class_ids_2 = self.model.detect(image_2, self.click_corners_2_np)
        image_2 = self.model.draw_bboxes(image_2, bboxes_2, confidences_2, class_ids_2)
        #testing centroid shifting
        for box in bboxes_2:
            centroid_x = int(box[0] + box[2]/2) 
            centroid_y = int(box[1] + box[3]/2)
            cv.circle(image_2, (centroid_x, centroid_y), 4, (0, 255, 0), -1)
            new_centroid = Transform.correct_centroid([centroid_x, centroid_y], 
                                                       self.frame_width, 
                                                       self.frame_height, 
                                                       self.factor_centroid, 
                                                       self.base_corr_centroid, 
                                                       self.max_dist_centroid)
            cv.circle(image_2, (new_centroid[0], new_centroid[1]), 4, (0, 0, 255), -1)
        image_2 = Helper.draw_points_green(image_2, self.click_corners_2_np)
        cv.imshow("Image 2 with corrected centroids", image_2)

        transformed_bboxes_2, _ = Transform.perspective_transform_bbox_draw(plane,
            bboxes_2, self.M2, self.w_car, self.h_car, self.frame_width, self.frame_height, 
            self.factor_centroid, self.base_corr_centroid, self.max_dist_centroid)
        #transformed_bboxes_2 = Transform.perspective_transform_bbox(bboxes_2, M2, w_car, h_car, frame_width, frame_height, factor_centroid, base_corr_centroid, max_dist_centroid)

        #white_plane_detections = Helper.draw_centroid_bboxes_green(white_plane_detections,transformed_bboxes_2)  
        white_plane_detections = Helper.draw_bboxes_green(white_plane_detections,transformed_bboxes_2) 
        cv.imshow("Projected Detections", white_plane_detections)

        ## Merge bboxes
        # Greedy strategy
        combined_bboxes = Transform.selected_join(transformed_bboxes_1 ,transformed_bboxes_2, self.w_car, self.h_car, self.combining_centroids_dist_threshold)
        # Optimization strategy
        #combined_bboxes = Transform.selected_join_optimized(transformed_bboxes_1 ,transformed_bboxes_2, w_car, h_car, combining_centroids_dist_threshold)
            
        # Define confidence and class for tracker
        len_combined_bboxes = len(combined_bboxes)
        combined_confidences = Helper.combine_confidences(len_combined_bboxes) #set to 0.8 by default
        combined_class_ids = Helper.combine_class_ids(len_combined_bboxes)

        #Traking on combined BBoxes
        white_plane_merged = np.ones(self.plane_size, dtype=np.uint8) * 255
        white_plane_merged = Helper.draw_plane_bounds(white_plane_merged, self.padding)
        white_plane_merged = self.model.draw_bboxes(white_plane_merged, combined_bboxes, combined_confidences, combined_class_ids)

        tracks = self.tracker.update(combined_bboxes, combined_confidences, combined_class_ids)
        #tracks, combined_white_plane = tracker.update_with_predictions(combined_bboxes, combined_confidences, combined_class_ids,combined_white_plane)

        #without showing prediction
        white_plane_merged_tracked = draw_tracks(white_plane_merged, tracks)
        cv.imshow("Tracked Merged Detections", white_plane_merged_tracked)            
        
        writer.write(white_plane_merged_tracked)
        print("####################################### \n")
                
        
        endtime = time.perf_counter()
        print(f"Image processing time: {endtime - starttime} s \n")
        starttime = endtime
        
        return white_plane_merged_tracked

    



