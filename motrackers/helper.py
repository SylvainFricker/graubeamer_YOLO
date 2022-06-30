import imp
import time
import cv2 as cv
import numpy as np

"""Class with helper functions"""
class Helper():

    #get newest frame from camera
    def get_frame(cap_1, cap_2):
        while True:
            starttime = time.perf_counter()
            cap_1.grab()
            cap_2.grab()
            grab_time = time.perf_counter() - starttime
            print(f'grab_time: {grab_time}')
            # flushes buffer and returns latest image
            if grab_time > 0.1:
                ok_1, image_1 = cap_1.retrieve()
                ok_2, image_2 = cap_2.retrieve()

                if not(ok_1 and ok_2):
                    print("[INFO] Cannot read the video feed.")
                    break

                return image_1, image_2
    
    # set confidence for tracker
    def combine_confidences(len):
        combined_confidence = np.empty(len)
        for i in range(len):
            combined_confidence[i] = 0.8
        return combined_confidence

    # set class IDs for tracker
    def combine_class_ids(len):
        combined_class_ids = np.empty(len)
        for i in range(len):
            combined_class_ids[i] = 2
        return combined_class_ids

    def corners_to_points(corners):
        p1 = np.array([corners[0,0],corners[0,1]])
        p2 = np.array([corners[1,0],corners[1,1]])
        p3 = np.array([corners[2,0],corners[2,1]])
        p4 = np.array([corners[3,0],corners[3,1]])
        return p1,p2,p3,p4

    # put text on image
    def put_Text_red(image):
        cv.putText(image,'select 4 Points with LEFT MouseButton',(2,50), cv.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 0, 255), 2)
        cv.putText(image,'Order: upper left, upper right, lower left, lower right',(2,90), cv.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 0, 255), 2)
        cv.putText(image,'then press esc to close window (sometime 2x needed)',(2,130), cv.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 0, 255), 2)
        cv.putText(image,'if you selected more then 4 restart the programm',(2,160), cv.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 0, 255), 2)
        return image

    def put_Text_green(image):
        cv.putText(image,'select 4 Points with RIGHT MouseButton',(2,50), cv.FONT_HERSHEY_SIMPLEX,
            0.7, (120, 255, 2), 2)
        cv.putText(image,'Order:  upper left, upper right, lower left, lower right',(2,90), cv.FONT_HERSHEY_SIMPLEX,
            0.7, (120, 255, 2), 2)
        cv.putText(image,'then press esc to close window (sometime 2x needed)',(2,130), cv.FONT_HERSHEY_SIMPLEX,
            0.7, (120, 255, 2), 2)
        cv.putText(image,'if you selected more then 4 restart the programm',(2,160), cv.FONT_HERSHEY_SIMPLEX,
            0.5, (120, 255, 2), 2)
        return image

    # draw projection plane bounds on white plane
    def draw_plane_bounds(image, padding):
        upper_left = (padding, padding)
        lower_right = (int(image.shape[1] - padding), int(image.shape[0] - padding))
        color = (0, 0, 0) #black
        cv.rectangle(image, upper_left, lower_right, color, 4 )
        return image

    # draw points on images
    def draw_points_red(image, points):
        count=1
        for point in points:
            x = point[0].astype(int)
            y = point[1].astype(int)
            text = "ROI {}".format(count)
            count += 1
            #cv.putText(image, text, (x - 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 200), 2)
            cv.circle(image, (x, y), 4, (0, 0, 255), -1)
        return image

    def draw_points_green(image, points):
        count=1
        for point in points:
            x = point[0].astype(int)
            y = point[1].astype(int)
            text = "ROI {}".format(count)
            count += 1
            #cv.putText(image, text, (x - 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 200), 2)
            cv.circle(image, (x, y), 4, (120, 255, 2), -1)
        return image

    #draw bboxes and centroids on images
    def draw_centroid_bboxes_green(image, bboxes):
        for bbox in bboxes:
            x = (bbox[0]+ 0.5*bbox[2]).astype(int)
            y = (bbox[1]+ 0.5*bbox[3]).astype(int)
            cv.circle(image, (x, y), 4, (120, 255, 2), -1)
        return image

    def draw_bboxes_green(image, bboxes):
        for bbox in bboxes:
            #draw centroid
            x = (bbox[0]+ 0.5*bbox[2]).astype(int)
            y = (bbox[1]+ 0.5*bbox[3]).astype(int)
            cv.circle(image, (x, y), 4, (120, 255, 2), -1)
            #draw box
            cv.rectangle(image,(int(bbox[0]),int(bbox[1])),(int(bbox[0]) + int(bbox[2]),int(bbox[1]) + int(bbox[3])),(120, 255, 2),4)
        return image

    def draw_centroid_bboxes_red(image, bboxes):
        for bbox in bboxes:
            x = (bbox[0]+ 0.5*bbox[2]).astype(int)
            y = (bbox[1]+ 0.5*bbox[3]).astype(int)
            cv.circle(image, (x, y), 4, (0, 0, 255), -1)
        return image

    def draw_bboxes_red(image, bboxes):
        for bbox in bboxes:
            #draw centroid
            x = (bbox[0]+ 0.5*bbox[2]).astype(int)
            y = (bbox[1]+ 0.5*bbox[3]).astype(int)
            cv.circle(image, (x, y), 4, (0, 0, 255), -1)
            #draw box
            cv.rectangle(image,(int(bbox[0]),int(bbox[1])),(int(bbox[0]) + int(bbox[2]),int(bbox[1]) + int(bbox[3])),(0, 0, 255),4)
        return image

    
            
