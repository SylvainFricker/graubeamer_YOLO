import time

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
            
