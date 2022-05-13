import numpy as np
import cv2 as cv
import math as math
import matplotlib.path as mplPath
import itertools
from scipy.spatial import distance
from motrackers.utils import get_PerspectiveMatrix
from motrackers.utils import Transform_Point
from motrackers.utils.misc import get_corners_of_ROI_as_int

class Transform():

    def PerspectiveTransformation(input,p1,p2,p3,p4):

        corners = np.float32([p4,p3,p2,p1])

        #pts2 = np.float32([[20,20],[820,20],[20,820],[820,820]])
        #pts2 = np.float32([[0,0],[1920,0],[0,1080],[1920,1080]])

        #M = cv.getPerspectiveTransform(corners,pts2)
        #print("Perspectiv_Transformation:\n",M)

        M = get_PerspectiveMatrix(p1,p2,p3,p4)
        transformed_picture = cv.warpPerspective(input,M,(840,840))

        #cv.imshow("affiined",transformed_picture)
        #status = cv.imwrite(f'D:\Studium\Bachelorarbeit\Graubeamer_files/test4_result.jpg', transfomred_picture)
        return(transformed_picture)

    def transform_image(image):
        
        #p1,p2,p3,p4,corners = get_corners_of_ROI_as_int()
        p1,p2,p3,p4 = np.int32([546,644]) , np.int32([953,667]) , np.int32([283,900]) , np.int32([1139,938])
        M = get_PerspectiveMatrix(p1,p2,p3,p4)
        
        transformed_img = cv.warpPerspective(image,M,(800,400))
        return(transformed_img)

    def undistort (input):
    
        mtx =  np.float32([[1.09202756e+03, 0.00000000e+00, 9.37896523e+02]
     ,[0.00000000e+00, 1.07931388e+03, 4.99915273e+02]
     ,[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    
        dist = np.float32( [[-0.12476623 , 0.02775942 ,-0.00109251 , 0.00081815 , 0.00526848]])
    
        newcameramtx =  np.float32([[1.09145886e+03, 0.00000000e+00, 9.37408099e+02]
     ,[0.00000000e+00, 1.07831458e+03, 4.99452401e+02]
     ,[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    
        dst = cv.undistort(input, mtx, dist, None, newcameramtx)
        return dst
        
    def draw_points(image, points):
        count=1
        for point in points:
            x = point[0].astype(int)
            y = point[1].astype(int)
            text = "ROI {}".format(count)
            count += 1
            cv.putText(image, text, (x - 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 200), 2)
            cv.circle(image, (x, y), 4, (200, 0, 200), -1)
        return image

    def draw_points_green(image, points):
        count=1
        for point in points:
            x = point[0].astype(int)
            y = point[1].astype(int)
            text = "ROI {}".format(count)
            count += 1
            cv.putText(image, text, (x - 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 200), 2)
            cv.circle(image, (x, y), 4, (120, 255, 2), -1)
        return image

    def draw_points_red(image, points):
        count=1
        for point in points:
            x = point[0].astype(int)
            y = point[1].astype(int)
            text = "ROI {}".format(count)
            count += 1
            cv.putText(image, text, (x - 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 200), 2)
            cv.circle(image, (x, y), 4, (0, 0, 255), -1)
        return image

    def draw_ll_of_bboxes(image, bboxes):
        count=1
        for bbox in bboxes:
            x = bbox[0].astype(int)
            y = bbox[1].astype(int)
            text = "tBBox {}".format(count)
            count += 1
            cv.putText(image, text, (x - 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 200), 2)
            cv.circle(image, (x, y), 3, (0, 150, 200), -1)
        return image

    def draw_centroid_bboxes(image, bboxes):
        for bbox in bboxes:
            x = (bbox[0]+ 0.5*bbox[2]).astype(int)
            y = (bbox[1]+ 0.5*bbox[3]).astype(int)
            cv.circle(image, (x, y), 3, (0, 150, 200), -1)
        return image

    def draw_bboxes(image, bboxes):
        for bbox in bboxes:
            cv.rectangle(image,(int(bbox[0]),int(bbox[1])),(int(bbox[0]) + int(bbox[2]),int(bbox[1]) + int(bbox[3])),(190,190,230),-1)
            #cv.line(image,np.array([bbox[0],bbox[1]]).astype(int),np.array([bbox[0] + bbox[2],bbox[1]]).astype(int), color=(220,195,220), thickness=4)
            #cv.line(image,np.array([bbox[0] + bbox[2],bbox[1]]).astype(int),np.array([bbox[0] + bbox[2],bbox[1] + bbox[3]]).astype(int), color=(220,195,230), thickness=4)
            #cv.line(image,np.array([bbox[0] + bbox[2],bbox[1] + bbox[3]]).astype(int),np.array([bbox[0],bbox[1] + bbox[3]]).astype(int), color=(220,195,222), thickness=4)
            #cv.line(image,np.array([bbox[0],bbox[1] + bbox[3]]).astype(int),np.array([bbox[0],bbox[1]]).astype(int), color=(220,190,220), thickness=4)
        return image

    def draw_centroid_bboxes_green(image, bboxes):
        for bbox in bboxes:
            x = (bbox[0]+ 0.5*bbox[2]).astype(int)
            y = (bbox[1]+ 0.5*bbox[3]).astype(int)
            cv.circle(image, (x, y), 4, (120, 255, 2), -1)
        return image

    def draw_bboxes_green(image, bboxes):
        for bbox in bboxes:
            cv.rectangle(image,(int(bbox[0]),int(bbox[1])),(int(bbox[0]) + int(bbox[2]),int(bbox[1]) + int(bbox[3])),(120, 255, 2),4)
            #cv.line(image,np.array([bbox[0],bbox[1]]).astype(int),np.array([bbox[0] + bbox[2],bbox[1]]).astype(int), color=(220,195,220), thickness=4)
            #cv.line(image,np.array([bbox[0] + bbox[2],bbox[1]]).astype(int),np.array([bbox[0] + bbox[2],bbox[1] + bbox[3]]).astype(int), color=(220,195,230), thickness=4)
            #cv.line(image,np.array([bbox[0] + bbox[2],bbox[1] + bbox[3]]).astype(int),np.array([bbox[0],bbox[1] + bbox[3]]).astype(int), color=(220,195,222), thickness=4)
            #cv.line(image,np.array([bbox[0],bbox[1] + bbox[3]]).astype(int),np.array([bbox[0],bbox[1]]).astype(int), color=(220,190,220), thickness=4)
        return image

    def draw_centroid_bboxes_red(image, bboxes):
        for bbox in bboxes:
            x = (bbox[0]+ 0.5*bbox[2]).astype(int)
            y = (bbox[1]+ 0.5*bbox[3]).astype(int)
            cv.circle(image, (x, y), 4, (0, 0, 255), -1)
        return image

    def draw_bboxes_red(image, bboxes):
        for bbox in bboxes:
            cv.rectangle(image,(int(bbox[0]),int(bbox[1])),(int(bbox[0]) + int(bbox[2]),int(bbox[1]) + int(bbox[3])),(0, 0, 255),4)
            #cv.line(image,np.array([bbox[0],bbox[1]]).astype(int),np.array([bbox[0] + bbox[2],bbox[1]]).astype(int), color=(220,195,220), thickness=4)
            #cv.line(image,np.array([bbox[0] + bbox[2],bbox[1]]).astype(int),np.array([bbox[0] + bbox[2],bbox[1] + bbox[3]]).astype(int), color=(220,195,230), thickness=4)
            #cv.line(image,np.array([bbox[0] + bbox[2],bbox[1] + bbox[3]]).astype(int),np.array([bbox[0],bbox[1] + bbox[3]]).astype(int), color=(220,195,222), thickness=4)
            #cv.line(image,np.array([bbox[0],bbox[1] + bbox[3]]).astype(int),np.array([bbox[0],bbox[1]]).astype(int), color=(220,190,220), thickness=4)
        return image

        
    def put_Text(image):
        cv.putText(image,'select 4 Points with LEFT MouseButton',(2,50), cv.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 0, 255), 2)
        cv.putText(image,'Order: lefttop, righttop, leftlow, rightlow',(2,90), cv.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 0, 255), 2)
        cv.putText(image,'then press esc to close window (sometime 2x needed)',(2,130), cv.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 0, 255), 2)
        cv.putText(image,'if you selected more then 4 restart the programm',(2,160), cv.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 0, 255), 2)
        return image

    def put_Text_green(image):
        cv.putText(image,'select 4 Points with RIGHT MouseButton',(2,50), cv.FONT_HERSHEY_SIMPLEX,
            0.7, (120, 255, 2), 2)
        cv.putText(image,'Order: righttop,lefttop,rightlow,leftlow',(2,90), cv.FONT_HERSHEY_SIMPLEX,
            0.7, (120, 255, 2), 2)
        cv.putText(image,'then press esc to close window (sometime 2x needed)',(2,130), cv.FONT_HERSHEY_SIMPLEX,
            0.7, (120, 255, 2), 2)
        cv.putText(image,'if you selected more then 4 restart the programm',(2,160), cv.FONT_HERSHEY_SIMPLEX,
            0.5, (120, 255, 2), 2)
        return image

    def Transform_and_draw_centroids(image,tracks,p1,p2,p3,p4):

        M = get_PerspectiveMatrix(p1,p2,p3,p4)
        #print("M form centroids\n",M)
        #print("tracks;\n",tracks)

        for track in tracks:

            track_id = track[1]
            xmin = track[2]
            ymin = track[3]
            width = track[4]
            height = track[5]

            xcentroid, ycentroid = int(xmin + 0.5*width), int(ymin + 0.5*height)
            centroid = np.array([int(xmin + 0.5*width),int(ymin + 0.5*height)])
            #print("centroid\n",centroid)
            transformed_centroid = Transform_Point(M, centroid)
            #print("transformedcentroid\n",transformed_centroid)
            xcentroid_transformeddd = int(transformed_centroid[0][0][0])
            ycentroid_transformeddd = int(transformed_centroid[0][0][1])

            merge = [xcentroid_transformeddd,ycentroid_transformeddd]
            #print("merge\n",merge)
            #np.ifcondition(merge>0, merge)
            #np.ifcondition(merge<800, merge)
            #print("xcentroid\n",xcentroid_transformed)
            #print("ycentroid\n",ycentroid_transformed)
            text = "ID {}".format(track_id)

            cv.circle(image, (xcentroid_transformeddd, ycentroid_transformeddd), 4, (50, 50, 255), -1)
            cv.putText(image, text, (xcentroid_transformeddd - 10, ycentroid_transformeddd - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (150, 50, 255), 2)
            #draw squars around centroids
            
        return image


    def get_centroid_transformed(bboxes):

        p1,p2,p3,p4,corners = get_corners_of_ROI_as_int()
        M = get_PerspectiveMatrix(p1,p2,p3,p4)

        one_bbox = False
        if len(bboxes.shape) == 1:
            one_bbox = True
            bboxes = bboxes[None, :]

        transformed_centroids = []

        for bbox in bboxes:
            #print("singel centroid:\n",centroid)
            w, h = bbox[2], bbox[3]
            x_centroid = bbox[0] + 0.5*w
            y_centroid = bbox[1] + 0.5*h

            centroid = np.array([int(x_centroid),int(y_centroid)])

            #centroid = np.array([int(xmin + 0.5*w),int(ymin + 0.5*h)])
            transformed_centroid = Transform_Point(M, centroid)
            #print("transformedcentroid\n",transformed_centroid)
            xcentroid_transformed = int(transformed_centroid[0][0][0])
            ycentroid_transformed = int(transformed_centroid[0][0][1])

            #only positiv values
            #xcentroid_transformed = abs(xcentroid_transformed)
            #ycentroid_transformed = abs(ycentroid_transformed)

            merge_transformed_centroid = [xcentroid_transformed,ycentroid_transformed]
            #print("merge_transformed_centroid\n",merge_transformed_centroid)
            #transformed_centroids = np.append(transformed_centroids,merge_transformed_centroid,axis=0)
            transformed_centroids.append(merge_transformed_centroid)

        #print("transformed_centroids type:\n",type(transformed_centroids))
        #print("transfomed_centroids:\n",transformed_centroids)

        reshahped_centroids = np.array(transformed_centroids)
        #print("respahed_centroids:\n",reshahped_centroids)
        if one_bbox:
            reshahped_centroids = reshahped_centroids.flatten()

        return reshahped_centroids

    def get_bboxes_transformed(bboxes,p1,p2,p3,p4):
        ''' 
        attentoi old and function, not correct box sizes 
        '''

        p1,p2,p3,p4,corners = get_corners_of_ROI_as_int()
        M = get_PerspectiveMatrix(p1,p2,p3,p4)
        #print("M form bboxes\n",M)

        one_bbox = False
        if len(bboxes.shape) == 1:
            one_bbox = True
            bboxes = bboxes[None, :]

        transformed_bboxes = []
        #print("inizialized_array\n",transformed_bboxes)

        for bbox in bboxes:
            #print("singel box:\n",bbox)

            w, h = bbox[2], bbox[3]

            #bbox in x direction 1920-->800 == 2.4
            xmid = bbox[0] + 0.5*w
            #bbox in y direction 1080-->800 == 1.35
            ymid = bbox[1] + 0.5*h

            leftlow = np.array([int(xmid),int(ymid)])

            transformed_bbox = Transform_Point(M, leftlow)
            xmid_transformed = int(transformed_bbox[0][0][0])
            ymid_transformed = int(transformed_bbox[0][0][1])

            merge_transformed_bbox = [xmid_transformed,ymid_transformed,w,h]
            #print("merge_transformed_leftlow\n",merge_transformed_leftlow)
            transformed_bboxes.append(merge_transformed_bbox)

        reshahped_bboxes = np.array(transformed_bboxes)

        if one_bbox:
            reshahped_bboxes = reshahped_bboxes.flatten()

        return reshahped_bboxes

    
    def old_get_all4points_bboxes_transformed_and_draw_them(plane,bboxes,p1,p2,p3,p4):

        #p1,p2,p3,p4,corners = get_corners_of_ROI_as_int()
        M = get_PerspectiveMatrix(p1,p2,p3,p4)

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

            centroid = np.array([int(xmid),int(ymid)])
            leftlow = np.array([int(xmin),int(ymin)])
            lefttop = np.array([int(xmin),int(ymax)])
            rightlow = np.array([int(xmax),int(ymin)])
            righttop = np.array([int(xmax),int(ymax)])

            transformed_centroid = Transform_Point(M, centroid)
            x_transformed_centroid = int(transformed_centroid[0][0][0])
            y_transformed_centroid = int(transformed_centroid[0][0][1])
            merge_transformed_centroid = [x_transformed_centroid,y_transformed_centroid]

            transformed_leftlow = Transform_Point(M, leftlow)
            x_transformed_leftlow = int(transformed_leftlow[0][0][0])
            y_transformed_leftlow = int(transformed_leftlow[0][0][1])
            merge_transformed_leftlow = [x_transformed_leftlow,y_transformed_leftlow]

            transformed_lefttop = Transform_Point(M, lefttop)
            x_transformed_lefttop = int(transformed_lefttop[0][0][0])
            y_transformed_lefttop = int(transformed_lefttop[0][0][1])
            merge_transformed_lefttop = [x_transformed_lefttop,y_transformed_lefttop]

            transformed_rightlow = Transform_Point(M, rightlow)
            x_transformed_rightlow = int(transformed_rightlow[0][0][0])
            y_transformed_rightlow = int(transformed_rightlow[0][0][1])
            merge_transformed_rightlow = [x_transformed_rightlow,y_transformed_rightlow]

            transformed_righttop = Transform_Point(M, righttop)
            x_transformed_righttop = int(transformed_righttop[0][0][0])
            y_transformed_righttop = int(transformed_righttop[0][0][1])
            merge_transformed_righttop = [x_transformed_righttop,y_transformed_righttop]

            cv.circle(plane, (x_transformed_centroid, y_transformed_centroid), 4, (200, 75, 255), -1)
            #cv.circle(plane, (x_transformed_leftlow, y_transformed_leftlow), 4, (200, 75, 255), -1)
            #cv.circle(plane, (x_transformed_lefttop, y_transformed_lefttop), 4, (200, 75, 255), -1)
            #cv.circle(plane, (x_transformed_rightlow, y_transformed_rightlow), 4, (200, 75, 255), -1)
            #cv.circle(plane, (x_transformed_righttop, y_transformed_righttop), 4, (200, 75, 255), -1)
                        
            cv.line(plane,merge_transformed_leftlow,merge_transformed_lefttop, color=(0,100,255), thickness=4)
            cv.line(plane,merge_transformed_lefttop,merge_transformed_righttop, color=(0,100,255), thickness=4)
            cv.line(plane,merge_transformed_righttop,merge_transformed_rightlow, color=(0,100,255), thickness=4)
            cv.line(plane,merge_transformed_rightlow,merge_transformed_leftlow, color=(0,100,255), thickness=4)
           
            cv.line(plane,
            [max(x_transformed_lefttop,x_transformed_leftlow),max(y_transformed_lefttop,y_transformed_righttop)],
            [min(x_transformed_rightlow,x_transformed_righttop),max(y_transformed_righttop,y_transformed_lefttop)]
            , color=(100,255,200), thickness=4)

            cv.line(plane,
             [min(x_transformed_rightlow,x_transformed_righttop),max(y_transformed_righttop,y_transformed_lefttop)],
              [min(x_transformed_rightlow,x_transformed_righttop),min(y_transformed_rightlow,y_transformed_leftlow)]
            , color=(100,255,200), thickness=4)

            cv.line(plane,
             [min(x_transformed_rightlow,x_transformed_righttop),min(y_transformed_rightlow,y_transformed_leftlow)],
              [max(x_transformed_lefttop,x_transformed_leftlow),min(y_transformed_rightlow,y_transformed_leftlow)]
            , color=(100,255,200), thickness=4)            

            cv.line(plane,
             [max(x_transformed_lefttop,x_transformed_leftlow),min(y_transformed_rightlow,y_transformed_leftlow)],
              [max(x_transformed_lefttop,x_transformed_leftlow),max(y_transformed_lefttop,y_transformed_righttop)]
            , color=(100,255,200), thickness=4)

            width_top = abs(distance.euclidean(merge_transformed_lefttop,merge_transformed_righttop))
            width_low = abs(distance.euclidean(merge_transformed_leftlow,merge_transformed_rightlow))
            height_left = abs(distance.euclidean(merge_transformed_lefttop,merge_transformed_leftlow))
            height_right = abs(distance.euclidean(merge_transformed_righttop,merge_transformed_rightlow))
                          
            w = (width_top + width_low) / 2
            h = (height_left + height_right) / 2
            merge_transformed_bbox = [int(x_transformed_centroid - w/2),int(y_transformed_centroid - h/2),int(w),int(h)]
            transformed_bboxes.append(merge_transformed_bbox)

            cv.rectangle(plane,(int(x_transformed_centroid - w/2),int(y_transformed_centroid - h/2)),(int(x_transformed_centroid + w/2),int(y_transformed_centroid + h/2)),(1,1,1),3)
            
        transformed_bboxes = np.array(transformed_bboxes)
        if one_bbox:
            transformed_bboxes = transformed_bboxes.flatten()

        return plane, transformed_bboxes
    

    def get_all4points_bboxes_transformed(plane,bboxes,p1,p2,p3,p4):

        #thresehold to change box size
        bbox_scale = 1

        #p1,p2,p3,p4,corners = get_corners_of_ROI_as_int()
        M = get_PerspectiveMatrix(p1,p2,p3,p4)
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

            centroid = np.array([int(xmid),int(ymid)])
            leftlow = np.array([int(xmin),int(ymin)])
            lefttop = np.array([int(xmin),int(ymax)])
            rightlow = np.array([int(xmax),int(ymin)])
            righttop = np.array([int(xmax),int(ymax)])

            transformed_centroid = Transform_Point(M, centroid)
            x_transformed_centroid = int(transformed_centroid[0][0][0])
            y_transformed_centroid = int(transformed_centroid[0][0][1])
            merge_transformed_centroid = [x_transformed_centroid,y_transformed_centroid]

            transformed_leftlow = Transform_Point(M, leftlow)
            x_transformed_leftlow = int(transformed_leftlow[0][0][0])
            y_transformed_leftlow = int(transformed_leftlow[0][0][1])
            merge_transformed_leftlow = [x_transformed_leftlow,y_transformed_leftlow]

            transformed_lefttop = Transform_Point(M, lefttop)
            x_transformed_lefttop = int(transformed_lefttop[0][0][0])
            y_transformed_lefttop = int(transformed_lefttop[0][0][1])
            merge_transformed_lefttop = [x_transformed_lefttop,y_transformed_lefttop]

            transformed_rightlow = Transform_Point(M, rightlow)
            x_transformed_rightlow = int(transformed_rightlow[0][0][0])
            y_transformed_rightlow = int(transformed_rightlow[0][0][1])
            merge_transformed_rightlow = [x_transformed_rightlow,y_transformed_rightlow]

            transformed_righttop = Transform_Point(M, righttop)
            x_transformed_righttop = int(transformed_righttop[0][0][0])
            y_transformed_righttop = int(transformed_righttop[0][0][1])
            merge_transformed_righttop = [x_transformed_righttop,y_transformed_righttop]

            #cv.circle(plane, (x_transformed_centroid, y_transformed_centroid), 4, (200, 75, 255), -1)
                        
            cv.line(plane,merge_transformed_leftlow,merge_transformed_lefttop, color=(0,100,255), thickness=4)
            cv.line(plane,merge_transformed_lefttop,merge_transformed_righttop, color=(0,100,255), thickness=4)
            cv.line(plane,merge_transformed_righttop,merge_transformed_rightlow, color=(0,100,255), thickness=4)
            cv.line(plane,merge_transformed_rightlow,merge_transformed_leftlow, color=(0,100,255), thickness=4)
            
            width_top = abs(distance.euclidean(merge_transformed_lefttop,merge_transformed_righttop))
            width_low = abs(distance.euclidean(merge_transformed_leftlow,merge_transformed_rightlow))
            height_left = abs(distance.euclidean(merge_transformed_lefttop,merge_transformed_leftlow))
            height_right = abs(distance.euclidean(merge_transformed_righttop,merge_transformed_rightlow))
                          
            w = ((width_top + width_low) / 2)*bbox_scale
            h = ((height_left + height_right) / 2)*bbox_scale
            merge_transformed_bbox = [int(x_transformed_centroid - w/2),int(y_transformed_centroid - h/2),int(w),int(h)]
            transformed_bboxes.append(merge_transformed_bbox)

            cv.rectangle(plane,(int(x_transformed_centroid - w/2),int(y_transformed_centroid - h/2)),(int(x_transformed_centroid + w/2),int(y_transformed_centroid + h/2)),(1,1,1),3)
            
        transformed_bboxes = np.array(transformed_bboxes)
        if one_bbox:
            transformed_bboxes = transformed_bboxes.flatten()

        return transformed_bboxes

    def combine_confidences(conf1,conf2):
        print("\n")
        print(conf1)
        print("\n")
        print(conf2)
        print("\n")
        combined_confidence = max(conf1,conf2)
        return combined_confidence

    def combine_confidences2(len):
        combined_confidence = np.empty(len)
        for i in range(len):
            combined_confidence[i] = 0.8
        return combined_confidence

    def combine_class_ids(len):
        combined_class_ids = np.empty(len)
        for i in range(len):
            combined_class_ids[i] = 2
        return combined_class_ids

    def selected_join(box_1,box_2, threshold = 100, large_bbox_threshold = 50, margin = 0.5 ,bbox_scale = 1):
        
        if len(box_1) == 0:
            join = box_2
        elif len(box_2) == 0:
            join = box_1
        else:
            join = np.append(box_1,box_2,0)
            join = join.tolist()

        for b2, b1 in itertools.combinations(join, 2):
            p1 = [b1[0],b1[1]]
            p2 = [b2[0],b2[1]]
            x = large_bbox_threshold / bbox_scale
            if b1[2] > x or b1[3] > x or b2[2] > x or b2[3] > x:
                y = (b1[2]+b1[2]+b2[2]+b2[3]) / 4
                if abs(distance.euclidean(p1,p2)) < threshold + margin*y:
                    try:
                        join.remove(b1)
                    except:
                        print("your boxjoin-threshold is to large") 
                    try:   
                        join.remove(b2)
                    except:
                        print("your boxjoin-threshold is to large")
                    join.append([(b1[0]+b2[0]) / 2 , (b1[1]+b2[1]) / 2 , (b1[2]+b2[2]) / 2 , (b1[3]+b2[3]) / 2])
                    print("joined a large box with threshold ",threshold + margin*y)
                    print("\n")
            else:
                if abs(distance.euclidean(p1,p2)) < threshold:
                    try:
                        join.remove(b1)
                    except:
                        print("your boxjoin-threshold is to large") 
                    try:   
                        join.remove(b2)
                    except:
                        print("your boxjoin-threshold is to large")
                    join.append([(b1[0]+b2[0]) / 2 , (b1[1]+b2[1]) / 2 , (b1[2]+b2[2]) / 2 , (b1[3]+b2[3]) / 2])
                    print("\n")
        output = np.int32(join)
        output = np.unique(output,axis=0)
        return output

    def get_all4points_bboxes_transformed_with_bbox_scale(plane,bboxes,p1,p2,p3,p4, bbox_scale):

        #thresehold to change box size

        #p1,p2,p3,p4,corners = get_corners_of_ROI_as_int()
        M = get_PerspectiveMatrix(p1,p2,p3,p4)
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

            centroid = np.array([int(xmid),int(ymid)])
            leftlow = np.array([int(xmin),int(ymin)])
            lefttop = np.array([int(xmin),int(ymax)])
            rightlow = np.array([int(xmax),int(ymin)])
            righttop = np.array([int(xmax),int(ymax)])

            transformed_centroid = Transform_Point(M, centroid)
            x_transformed_centroid = int(transformed_centroid[0][0][0])
            y_transformed_centroid = int(transformed_centroid[0][0][1])
            merge_transformed_centroid = [x_transformed_centroid,y_transformed_centroid]

            transformed_leftlow = Transform_Point(M, leftlow)
            x_transformed_leftlow = int(transformed_leftlow[0][0][0])
            y_transformed_leftlow = int(transformed_leftlow[0][0][1])
            merge_transformed_leftlow = [x_transformed_leftlow,y_transformed_leftlow]

            transformed_lefttop = Transform_Point(M, lefttop)
            x_transformed_lefttop = int(transformed_lefttop[0][0][0])
            y_transformed_lefttop = int(transformed_lefttop[0][0][1])
            merge_transformed_lefttop = [x_transformed_lefttop,y_transformed_lefttop]

            transformed_rightlow = Transform_Point(M, rightlow)
            x_transformed_rightlow = int(transformed_rightlow[0][0][0])
            y_transformed_rightlow = int(transformed_rightlow[0][0][1])
            merge_transformed_rightlow = [x_transformed_rightlow,y_transformed_rightlow]

            transformed_righttop = Transform_Point(M, righttop)
            x_transformed_righttop = int(transformed_righttop[0][0][0])
            y_transformed_righttop = int(transformed_righttop[0][0][1])
            merge_transformed_righttop = [x_transformed_righttop,y_transformed_righttop]

            #cv.circle(plane, (x_transformed_centroid, y_transformed_centroid), 4, (200, 75, 255), -1)
                        
            cv.line(plane,merge_transformed_leftlow,merge_transformed_lefttop, color=(0,100,255), thickness=4)
            cv.line(plane,merge_transformed_lefttop,merge_transformed_righttop, color=(0,100,255), thickness=4)
            cv.line(plane,merge_transformed_righttop,merge_transformed_rightlow, color=(0,100,255), thickness=4)
            cv.line(plane,merge_transformed_rightlow,merge_transformed_leftlow, color=(0,100,255), thickness=4)
            
            width_top = abs(distance.euclidean(merge_transformed_lefttop,merge_transformed_righttop))
            width_low = abs(distance.euclidean(merge_transformed_leftlow,merge_transformed_rightlow))
            height_left = abs(distance.euclidean(merge_transformed_lefttop,merge_transformed_leftlow))
            height_right = abs(distance.euclidean(merge_transformed_righttop,merge_transformed_rightlow))
                          
            w = ((width_top + width_low) / 2)*bbox_scale
            h = ((height_left + height_right) / 2)*bbox_scale
            merge_transformed_bbox = [int(x_transformed_centroid - w/2),int(y_transformed_centroid - h/2),int(w),int(h)]
            transformed_bboxes.append(merge_transformed_bbox)

            cv.rectangle(plane,(int(x_transformed_centroid - w/2),int(y_transformed_centroid - h/2)),(int(x_transformed_centroid + w/2),int(y_transformed_centroid + h/2)),(1,1,1),3)
            
        transformed_bboxes = np.array(transformed_bboxes)
        if one_bbox:
            transformed_bboxes = transformed_bboxes.flatten()

        return transformed_bboxes