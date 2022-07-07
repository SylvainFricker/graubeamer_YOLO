from flask import Response
from flask import Flask
from flask import render_template
from yolov3_tracker import YOLOv3_Tracker
import cv2 as cv
from motrackers import Transform

""" Settings """
isgpu = False
islive = False


""" streaming function """
# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)

# initialize a flask object
app = Flask(__name__, template_folder='templates', static_folder='styling')

@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")

def generate(tracker):
    global cap_1, cap_2, writer
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

    ul_1,ur_1,ll_1,lr_1,ul_2,ur_2,ll_2,lr_2 = tracker.fix_ROI()
    #ul_1,ur_1,ll_1,lr_1,ul_2,ur_2,ll_2,lr_2 = tracker.choose_ROI(cap_1, cap_2)

    """Perspective Transformation Matrices"""
    tracker.set_transformation_Matrices(ul_1,ur_1,ll_1,lr_1,ul_2,ur_2,ll_2,lr_2)

    if islive:
        # release and renew videocapture for timing
        cap_1.release()
        cap_2.release()

        cap_1 = cv.VideoCapture('http://graubeamer:Autokino22@{}/cgi-bin/mjpeg?stream=[1]'.format(camera_IP1))
        cap_2 = cv.VideoCapture('http://graubeamer:Autokino22@{}/cgi-bin/mjpeg?stream=[1]'.format(camera_IP2))

    # saving video
    plane_size = tracker.get_plane_size()
    video_out = "./../video_output/liveDavos_130722_test_class.avi"
    fourcc = cv.VideoWriter_fourcc(*"MJPG")
    writer = cv.VideoWriter(video_out, fourcc, 4, (plane_size[1], plane_size[0]), True)

    # loop over frames from the output stream
    while True:
        outputFrame = tracker.run_tracker(writer, cap_1, cap_2)
        (flag, encodedImage) = cv.imencode(".jpg", outputFrame)

        if not flag:
            continue
        
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage) + b'\r\n')

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(YOLOv3_Tracker(isgpu = isgpu, islive = islive)),
        mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':    
    global cap_1, cap_2, writer

    app.run(host="localhost", port=8000, debug=True, threaded=True, use_reloader=False)
    print("Shut Down")
    writer.release() 
    print("[INFO] cleaning up...")
    #cv.destroyAllWindows()
    #cv.waitKey(1)
    #cv.destroyWindow("Tracked Merged Detections")
    cap_1.release()
    cap_2.release()
    print("released cap")
    

    