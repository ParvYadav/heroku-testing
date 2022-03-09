import cv2, io, base64
from PIL import Image
import threading
from time import sleep
from imutils.video import WebcamVideoStream
import numpy as np
import face_recognition
import dlib
from scipy.spatial import distance as dist



class VideoCamera(object):
    def __init__(self):

        self.to_process = []
        self.output_image_rgb = []
        self.output_image_bgr = []
        thread = threading.Thread(target=self.keep_processing, args=())
        thread.daemon = True
        thread.start()
        
    
    def process_one(self):
        if not self.to_process:
            return
        input_str = self.to_process.pop(0)
        imgdata = base64.b64decode(input_str)
        input_img = np.array(Image.open(io.BytesIO(imgdata)))
        """
        After getting the image you can do any preprocessing here
        """
        # _______________________________________Performing some pre processing_______________________________________________

        bgr_image = cv2.flip(input_img, 1)  # Flip the image
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)  # Changing color from bgr to rgb
        # _______________________________________Performing some pre processing_______________________________________________
        known_face_encodings = np.load('encodeListKnown_test_v5.npy', allow_pickle=True)
        known_face_names = ['Aman', 'Aman', 'Aman', 'Aman', 'Aman', 'Aman', 'Aman', 'Ashi', 'Ashi', 'Ashi', 'Ashi', 'Ashish', 'Ashish', 'Ashish', 'Ashish', 'Ashish', 'Barkha Sharma', 'Barkha Sharma', 'Barkha Sharma', 'Barkha Sharma', 'Barkha Sharma', 'Bhaneshwari', 'Bhaneshwari', 'Bhaneshwari', 'Chhaya', 'Chhaya', 'Chhaya', 'Chhaya', 'Chhaya', 'Deepak', 'Deepak', 'Deepak', 'Madan Agrawal', 'Madan', 'Madhav', 'Madhav', 'Madhav', 'Madhav', 'Madhav', 'Manvi', 'Manvi', 'Navdeep', 'Navdeep', 'Navdeep', 'Nikita ', 'Nikita', 'Nikita', 'Nikita', 'Nikita', 'Nikita', 'Nikita', 'Pankaj Dwivedi', 'Parv Yadav', 'Parv', 'Payal', 'Payal', 'Payal', 'Pooja', 'Pooja', 'Pooja', 'Prashasht', 'Prashasht', 'Prashasht', 'Prashasht', 'Prashasht', 'Prashasht', 'Prashasht',
        'Prashasht', 'Priyank', 'Priyank', 'Priyank', 'Shivangi', 'Shivangi', 'Shivangi', 'Shivangi', 'Shivangi', 'Shraddha', 'Shraddha', 'Shraddha', 'Shraddha', 'Shraddha', 'Shreya', 'Shreya Goyal', 'Tanishka', 'Tanishka', 'Tanishka', 'Tanishka', 'Tanvi Bhave', 'Tarun', 'Tarun Sinhal', 'Tilottama Sharma', 'Vandana Chouhan', 'Vijay Patidar', 'Vijeet Agrawal']

        
        RIGHT_EYE_POINTS = list(range(36, 42))
        LEFT_EYE_POINTS = list(range(42, 48))
        
        EYE_AR_THRESH = 0.33
        EYE_AR_CONSEC_FRAMES = 2
        
        COUNTER = 0
        TOTAL = 0
        
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks (2).dat')
        
        # _______________________________________Performing some pre processing_______________________________________________
        
        cv2.putText(rgb_image, "Winked {}".format('1'), (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0),1)
                        

        # ______________________________________________________________________________________________________________________

        ret, rgb_jpeg = cv2.imencode('.jpg', rgb_image)
        _, bgr_jpeg = cv2.imencode('.jpg', bgr_image)

        self.output_image_rgb.append(rgb_jpeg.tobytes())
        self.output_image_bgr.append(bgr_jpeg.tobytes())

    def keep_processing(self):
        while True:
            self.process_one()
            sleep(0.01)

    def enqueue_input(self, input):
        self.to_process.append(input)

    def get_frame(self):
        while not self.output_image_rgb:
            sleep(0.05)
        return self.output_image_rgb.pop(0), self.output_image_bgr.pop(0)


"""
# def stringToImage(base64_string):
#     imgdata = base64.b64decode(base64_string)
#     return np.array(Image.open(io.BytesIO(imgdata)))
"""
