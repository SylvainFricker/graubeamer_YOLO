import tracemalloc
import numpy as np
import cv2 as cv
import math as math
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment


class Transform():

    def get_PerspectiveMatrix(p1, p2, p3, p4, plane_width, plane_height, padding):
        corners = np.float32([p1,p2,p3,p4])
        # Adapted to plane size: ul, ur, ll, lr
        pts2 = np.float32([[padding, padding],
                           [plane_width - padding, padding],
                           [padding, plane_height - padding],
                           [plane_width - padding, plane_height - padding]])
        M = cv.getPerspectiveTransform(corners,pts2)
        return M
    
    def transform_point(M , point):
        x = point[0]
        y = point[1]
        pts = np.array([[x,y]], dtype = "float32")    
        pts = np.array([pts])
        transformed_point = cv.perspectiveTransform(pts,M)
        return transformed_point

    def correct_centroid(centroid, im_w, im_h, factor, base_corr, max_dist):
        ref = [im_w/2, im_h]
        dist = abs(distance.euclidean(ref, centroid))
        #dist_corr = dist * (0.9 + factor * dist / max_dist)
        dist_corr = dist - (base_corr - dist / max_dist * factor)
        angle = math.atan2(ref[1] - centroid[1], centroid[0] - ref[0])
        new_centroid_x = int(ref[0] + math.cos(angle) * dist_corr)
        new_centroid_y = int(ref[1] - math.sin(angle) * dist_corr)
        return [new_centroid_x, new_centroid_y]

    #IN USE
    def selected_join(box_1, box_2, w_car, h_car, threshold):
        if len(box_1) == 0 and len(box_2) == 0:
            return np.empty((0,4))
        elif len(box_1) == 0:
            return box_2
        elif len(box_2) == 0:
            return box_1
        else:     
            join = np.empty((0, 4))
            index_p1 = 0
            distances = np.empty((0,3))
            index_not_used_points_1 = list(range(len(box_1)))
            index_not_used_points_2 = list(range(len(box_2)))  

            for b1 in box_1:
                p1 = [b1[0],b1[1]]
                index_p2 = 0           
                for b2 in box_2:
                    p2 = [b2[0],b2[1]]
                    dist = abs(distance.euclidean(p1,p2))
                    x = [dist,index_p1,index_p2]
                    distances = np.append(distances,[x],axis=0)
                    index_p2 += 1
                index_p1 += 1          
            distances = distances[distances[:, 0].argsort()]

            for i in distances:
                if i[0] < threshold:
                    if i[1] in index_not_used_points_1 and i[2] in index_not_used_points_2:
                        joined_point = np.array([(box_1[int(i[1]),0]+box_2[int(i[2]),0]) / 2 ,(box_1[int(i[1]),1]+box_2[int(i[2]),1]) / 2 ,w_car ,h_car])
                        join = np.append(join,[joined_point],axis=0)              
                        index_not_used_points_1.remove(int(i[1]))
                        index_not_used_points_2.remove(int(i[2]))     

            for j in index_not_used_points_1:
                p1 = np.array([box_1.item(j,0),box_1.item(j,1),w_car,h_car])
                join = np.append(join,[p1],axis=0)       

            for k in index_not_used_points_2:
                p2 = np.array([box_2.item(k,0),box_2.item(k,1),w_car,h_car])
                join = np.append(join,[p2],axis=0)
            join = np.int32(join)
            return join

    def selected_join_optimized(box_1, box_2, w_car, h_car, threshold): 
        if len(box_1) == 0 and len(box_2) == 0:
            return np.empty((0,4))
        elif len(box_1) == 0:
            return box_2
        elif len(box_2) == 0:
            return box_1
        else:
            box_1 = box_1[0:,0:2]
            box_2 = box_2[0:,0:2]
            join = np.empty((0,4))
            
            distances = distance.cdist(box_1, box_2)
            distances[distances<0] = 0
            assigned_boxes_1, assigned_boxes_2 = linear_sum_assignment(distances)
            cost_matrix = distances[assigned_boxes_1, assigned_boxes_2].sum()
            index_unmatched_boxes_2, index_unmatched_boxes_1 = [], []

            for d in range(box_2.shape[0]):
                if d not in assigned_boxes_2:
                    index_unmatched_boxes_2.append(d)

            for t in range(box_1.shape[0]):
                if t not in assigned_boxes_1:
                    index_unmatched_boxes_1.append(t)
            index_matches = []

            for t, d in zip(assigned_boxes_1, assigned_boxes_2):
                if distances[t, d] > threshold:
                    index_unmatched_boxes_1.append(t)
                    index_unmatched_boxes_2.append(d)
                else:
                    index_matches.append((t, d))
            
            if len(index_matches):
                index_matches = np.array(index_matches)
            else:
                index_matches = np.empty((0, 2), dtype=int)                  
            index_unmatched_boxes_1 = np.array(index_unmatched_boxes_1)
            index_unmatched_boxes_2 = np.array(index_unmatched_boxes_2)
            index_matches = np.array(index_matches)
          
            for pair in index_matches:
                joined_point = np.array([(box_1[pair[0],0]+box_2[pair[1],0]) / 2 ,(box_1[pair[0],1]+box_2[pair[1],1]) / 2 ,w_car ,h_car])
                join = np.append(join,[joined_point],axis=0)
                               
            for j in index_unmatched_boxes_1:
                p1 = np.array([box_1.item(j,0),box_1.item(j,1),w_car ,h_car])
                join = np.append(join,[p1],axis=0)
                
            for k in index_unmatched_boxes_2:
                p2 = np.array([box_2.item(k,0),box_2.item(k,1),w_car ,h_car])
                join = np.append(join,[p2],axis=0)

            join = np.int32(join)
            return join

    # new function
    def perspective_transform_bbox(bboxes, M, w_car, h_car, im_w, im_h, factor_centroid, base_corr_centroid, max_dist_centroid):

        one_bbox = False
        if len(bboxes.shape) == 1:
            one_bbox = True
            bboxes = bboxes[None, :]

        transformed_bboxes = []

        for bbox in bboxes:

            xmin, ymin = bbox[0], bbox[1]
            w, h = bbox[2], bbox[3]
            xmid = xmin + 0.5*w
            ymid = ymin + 0.5*h
            centroid = np.array([int(xmid),int(ymid)], dtype = "float32")
            #bbox in x direction 1920-->800 == 2.4
            #bbox in y direction 1080-->800 == 1.35
            
            """ correct for offsets here"""
            corr_centroid = Transform.correct_centroid(centroid, im_w, im_h, factor_centroid, base_corr_centroid, max_dist_centroid)

            transformed_centroid = Transform.transform_point(M, corr_centroid)
            
            # safe box as lower left corner and width / height: [x,y,w,h]
            transformed_bbox = [int(transformed_centroid[0][0][0] - w_car/2), int(transformed_centroid[0][0][1] - h_car/2), w_car, h_car]
            transformed_bboxes.append(transformed_bbox)
   
        transformed_bboxes = np.array(transformed_bboxes)

        if one_bbox:
            transformed_bboxes = transformed_bboxes.flatten()

        return transformed_bboxes

    def perspective_transform_bbox_draw(plane, bboxes, M, w_car, h_car, im_w, im_h, factor_centroid, base_corr_centroid, max_dist_centroid):
       
        one_bbox = False
        if len(bboxes.shape) == 1:
            one_bbox = True
            bboxes = bboxes[None, :]

        transformed_bboxes = []

        for bbox in bboxes:
            #print("singel box:\n",bbox)

            w, h = bbox[2], bbox[3]

            #bbox in x direction 1920-->800 == 2.4
            xmin = bbox[0]
            xmid = bbox[0] + 0.5*w
            xmax = bbox[0] + w
            #bbox in y direction 1080-->800 == 1.35
            ymax = bbox[1]
            ymid = bbox[1] + 0.5*h
            ymin = bbox[1] + h

            centroid_corrected = np.array(Transform.correct_centroid([int(xmid),int(ymid)], im_w, im_h, factor_centroid, base_corr_centroid, max_dist_centroid))
            centroid = np.array([int(xmid),int(ymid)])
            middle = np.array([int(xmid), int(ymin)])
            leftlow = np.array([int(xmin),int(ymin)])
            lefttop = np.array([int(xmin),int(ymax)])
            rightlow = np.array([int(xmax),int(ymin)])
            righttop = np.array([int(xmax),int(ymax)])

            transformed_centroid_corrected = Transform.transform_point(M, centroid_corrected)
            x_transformed_centroid_corrected = int(transformed_centroid_corrected[0][0][0])
            y_transformed_centroid_corrected = int(transformed_centroid_corrected[0][0][1])
            merge_transformed_centroid_corrected = [x_transformed_centroid_corrected,y_transformed_centroid_corrected]

            transformed_centroid = Transform.transform_point(M, centroid)
            x_transformed_centroid = int(transformed_centroid[0][0][0])
            y_transformed_centroid = int(transformed_centroid[0][0][1])
            merge_transformed_centroid = [x_transformed_centroid,y_transformed_centroid]

            transformed_middle = Transform.transform_point(M, middle)
            x_transformed_middle = int(transformed_middle[0][0][0])
            y_transformed_middle = int(transformed_middle[0][0][1])
            merge_transformed_middle = [x_transformed_middle,y_transformed_middle]

            transformed_leftlow = Transform.transform_point(M, leftlow)
            x_transformed_leftlow = int(transformed_leftlow[0][0][0])
            y_transformed_leftlow = int(transformed_leftlow[0][0][1])
            merge_transformed_leftlow = [x_transformed_leftlow,y_transformed_leftlow]

            transformed_lefttop = Transform.transform_point(M, lefttop)
            x_transformed_lefttop = int(transformed_lefttop[0][0][0])
            y_transformed_lefttop = int(transformed_lefttop[0][0][1])
            merge_transformed_lefttop = [x_transformed_lefttop,y_transformed_lefttop]

            transformed_rightlow = Transform.transform_point(M, rightlow)
            x_transformed_rightlow = int(transformed_rightlow[0][0][0])
            y_transformed_rightlow = int(transformed_rightlow[0][0][1])
            merge_transformed_rightlow = [x_transformed_rightlow,y_transformed_rightlow]

            transformed_righttop = Transform.transform_point(M, righttop)
            x_transformed_righttop = int(transformed_righttop[0][0][0])
            y_transformed_righttop = int(transformed_righttop[0][0][1])
            merge_transformed_righttop = [x_transformed_righttop,y_transformed_righttop]

            cv.circle(plane, (x_transformed_centroid, y_transformed_centroid), 4, (0, 255, 0), -1)
            cv.circle(plane, (x_transformed_centroid_corrected, y_transformed_centroid_corrected), 4, (0, 0, 255), -1)
                        
            cv.line(plane,merge_transformed_leftlow,merge_transformed_lefttop, color=(0,100,255), thickness=4)
            cv.line(plane,merge_transformed_lefttop,merge_transformed_righttop, color=(0,100,255), thickness=4)
            cv.line(plane,merge_transformed_righttop,merge_transformed_rightlow, color=(0,100,255), thickness=4)
            cv.line(plane,merge_transformed_rightlow,merge_transformed_leftlow, color=(0,100,255), thickness=4)
            
            # safe box as lower left corner and width / height: [x,y,w,h]
            merge_transformed_bbox = [int(x_transformed_centroid - w_car/2),int(y_transformed_centroid - h_car/2),int(w_car),int(h_car)]
            transformed_bboxes.append(merge_transformed_bbox)

        cv.imshow("Perspective Transformed boxes with", plane)   
        transformed_bboxes = np.array(transformed_bboxes)
        if one_bbox:
            transformed_bboxes = transformed_bboxes.flatten()

        return transformed_bboxes, plane

    