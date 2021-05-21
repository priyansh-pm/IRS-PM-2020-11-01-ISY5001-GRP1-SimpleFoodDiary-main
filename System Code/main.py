import pickle
import sys
import boto3
import json
import os
import threading
import cv2
import numpy as np
import atexit
import ipywidgets as widgets
import warnings
import pandas as pd

sys.path.append('models-master/research')
sys.path.append('models-master/research/object_detection')
sys.path.append('models-master/research/slim')

os.environ['PATH'] += ':./:./slim/'

from datetime import datetime
from PIL import Image
from flask import Flask, render_template, Response, jsonify, request, session
from werkzeug.utils import secure_filename
from tensorflow.keras import models
import tensorflow as tf
from object_detection.utils import label_map_util
from dateutil.parser import parse


IMG_HEIGHT = IMG_WIDTH = 224

IMAGE = None
FRUIT_LABEL = None

food_dict = {}
GRAPH = None
AWS_ACCESS_KEY = ""
AWS_SECRET_KEY = ""


def create_app():
    curr_app = Flask(__name__)

    def start_thread():
        keras_thread.start()

    def interrupt():
        keras_thread.cancel()

    # Initiate
    start_thread()
    # When you kill Flask (SIGTERM), clear the trigger for the next thread
    atexit.register(interrupt)
    return curr_app


class VideoCamera(object):
    """
    Class in charge of handling images from user's webcam
    """
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.video.set(3, 1920)
        self.video.set(4, 1080)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        """
        Get frame from the user's webcam and resize it.
        :return: The .jpg version of the frame
        """
        _, original = self.video.read()

        # Resize the high resolution feed smaller
        # to fit web page
        global IMAGE
        IMAGE = cv2.resize(original, (960, 540))

        _, jpeg = cv2.imencode('.jpg', IMAGE)

        return jpeg

    def return_frame(self):
        return self.get_frame().tobytes()


class MyThread(threading.Thread):
    """
    Class handling separate thread for the frame prediction
    of the five types of fruits
    """
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        # Load the VGG16 network
        print("[INFO] Load VGG16 network and keep predicting picture in frame.")
        while True:
            if IMAGE is not None:
                global FRUIT_LABEL
                FRUIT_LABEL = self.predict_frame(IMAGE)

    def predict_frame(self, frame):
        """
        Frame from camera is preprocessed and fed into
        fruit prediction model. If the best guess from fruit prediction model
        (max(preds[0]) is less than 0.9, return empty label.
        Else return label correlated to best guess.
        :param frame: Image
        :return: Label in string
        """
        preds = predict_fruit(frame)

        if max(preds[0]) < 0.9:
            return "Not sure..."
        else:
            pred_label_index = preds.argmax(axis=1)
            labels = ['Apple', 'Longbean', 'Onion', 'Pineapple', 'Potato']

            return labels[pred_label_index[0]]


video_stream = VideoCamera()
keras_thread = MyThread()
app = create_app()


@app.route('/')
def index():
    """
    Renders first page. Clear any food_dict that there was previously.
    :return: First page's HTML
    """
    global food_dict
    food_dict = {}
    return render_template('diary.html', label=FRUIT_LABEL)


def gen(camera):
    while True:
        frame = camera.return_frame()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def predict_and_crop_label(image_path, image_name):
    """
    Predict the location of nutrition label, crop it out and save to file
    :param image_path: Relative path to the actual image without cropping
    :param image_name: In the format of {date_str}_{label}.
                       This is just used to save the cropped image
    :return: List of full file locations of cropped images
    """
    nutrition_label_filenames = []

    image = Image.open(image_path)

    # if GRAPH is None:
    PATH_TO_FROZEN_GRAPH = "frozen_inference_graph_Final.pb"

    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        # Unwrap the frozen graph
        with tf.compat.v1.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        # Crop image according to bounding box
        with tf.compat.v1.Session() as sess:
            image_tensor = graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = graph.get_tensor_by_name('detection_scores:0')
            detection_classes = graph.get_tensor_by_name('detection_classes:0')
            num_detections = graph.get_tensor_by_name('num_detections:0')

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image, axis=0)

            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            (width, height) = image.size

            box = np.squeeze(boxes)
            max_boxes_to_draw = box.shape[0]
            scores = np.squeeze(scores)
            min_score_thresh = 0.5
            bounding_boxes = []
            for i in range(min(max_boxes_to_draw, box.shape[0])):
                if scores[i] > min_score_thresh:
                    ymin = (int(box[i, 0] * height))
                    xmin = (int(box[i, 1] * width))
                    ymax = (int(box[i, 2] * height))
                    xmax = (int(box[i, 3] * width))
                    bounding_boxes.append([xmin, ymin, xmax, ymax])

            num = 0
            for box in bounding_boxes:
                xmin, ymin, xmax, ymax = box
                image_nd_array = cv2.imread(image_path)
                img = image_nd_array[ymin:ymax, xmin:xmax]
                filename = f'{image_name}_crop_{num}.jpg'
                cv2.imwrite(os.path.join("static/images", f"{filename}.jpg"), img)
                nutrition_label_filenames.append(os.path.join("static/images", f"{filename}.jpg"))
                num += 1

    return nutrition_label_filenames


@app.route('/video_feed')
def video_feed():
    """
    Feed frames from webcam to UI
    :return: Response with JPEG
    """
    return Response(gen(video_stream), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/label")
def label():
    """
    Feed fruit prediction to UI
    :return: JSON in string form
    """
    return json.dumps({"label": FRUIT_LABEL})


@app.route("/capture", methods=['POST'])
def capture():
    """
    The function is called when user click capture on the front end.
    food_label is the food_label that is sent from the front-end,
    to ensure that that is the label user wants. No matter what image it is,
    we will save it first with cv2.imwrite then send to prediction for nutrition labels
    If there is a nutrition label, update the food_dict/user's food diary with nutrition label
    information, else just the fruit label info.
    :return: JSON string with food_dict
    """
    food_label = request.json["food_label"]
    date_str = datetime.now().strftime("%d-%m-%Y, %H:%M:%S")
    filename = secure_filename(f"{date_str}_{food_label}")

    cv2.imwrite(os.path.join("static/images", f"{filename}.jpg"), IMAGE)
    cropped_dict = get_cropped_image_text(os.path.join("static/images", f"{filename}.jpg"), filename)

    if cropped_dict:
        food_dict.update(cropped_dict)
    else:
        curr_fruit_dict = {
            "Label": food_label,
            "Date": date_str
        }
        food_dict[json.dumps(curr_fruit_dict)] = os.path.join("static/images", f"{filename}.jpg")

    return json.dumps({'food_list': food_dict}), 200, {'ContentType': 'application/json'}


@app.route("/file-upload", methods=['POST'])
def file_upload():
    """
    Function that is called when a file is uploaded.
    :return: Dictionary with json string as key, and file name/location as value. The JSON
    string either are values extracted from the nutrition label or label from fruit model
    """
    file = request.files['file']
    # Convert Filestorage file to ndarray so as to feed into predict_fruit()
    content = file.read()
    content_as_ndarray = cv2.imdecode(np.frombuffer(content, np.uint8), -1)
    filename, file_extension = os.path.splitext(file.filename)
    fruit_preds = predict_fruit(content_as_ndarray)

    # Here we first send to nutrition label models first, if no cropped_dict (i.e.no nutrition label)
    # then send to fruit model to guess fruit. If yes, update the food_dict
    cv2.imwrite(os.path.join("static/images", f"{filename}_original.jpg"), content_as_ndarray)
    cropped_dict = get_cropped_image_text(os.path.join("static/images", f"{filename}_original.jpg"),
                                          f"{filename}_original.jpg")
    if cropped_dict:
        food_dict.update(cropped_dict)
    else:
        pred_label_index = fruit_preds.argmax(axis=1)
        labels = ['Apple', 'Longbean', 'Onion', 'Pineapple', 'Potato']
        fruit_name = labels[pred_label_index[0]]
        date_str = datetime.now().strftime("%d-%m-%Y, %H:%M:%S")
        full_filename = secure_filename(f"{date_str}_{fruit_name}")

        cv2.imwrite(os.path.join("static/images", f"{full_filename}.jpg"), content_as_ndarray)
        curr_fruit_dict = {
            "Label": fruit_name,
            "Date": date_str
        }
        food_dict[json.dumps(curr_fruit_dict)] = os.path.join("static/images", f"{full_filename}.jpg")

    return json.dumps({'food_list': food_dict}), 200, {'ContentType': 'application/json'}


def recognition(file_names):
    """
    Given cropped images, extract text using Amazon Web Services then
    compute the total points and grading of the food.
    :param file_names: List of file names of cropped images
    :return: Dictionary with JSON string of text extracted from label as key,
    and file name of the cropped image as value
    """
    region = 'ap-southeast-1'
    access_key = AWS_ACCESS_KEY
    secret_key = AWS_SECRET_KEY
    client = boto3.client('rekognition',
                          region_name=region,
                          aws_access_key_id=access_key,
                          aws_secret_access_key=secret_key)

    json_dict = {}

    for file_name in file_names:
        with open(file_name, "rb") as image:
            f = image.read()
            imagedata = bytearray(f)

        # Send the image data to Amazon to recognize
        response = client.detect_text(
            Image={'Bytes': imagedata},
        )

        list_x = []

        for textDetection in response["TextDetections"]:
            if textDetection["Type"] == "LINE":
                list_x.append(textDetection["DetectedText"])

        warnings.filterwarnings("ignore")

        if not list_x:
            continue

        # Extracted lines are loaded to a dataframe
        df = pd.DataFrame(list_x, columns=['lines'])

        # Splitting the lines into each column
        split_data = df["lines"].str.split(" ")
        data = split_data.to_list()
        new_df = pd.DataFrame(data)

        if len(list(new_df.columns)) < 3:
            continue

        # Selecting only the first 3 columns with max extracted information
        new_df = new_df.iloc[:, :3]

        # Combining the first 2 columns to give meaningful representation of the nutrient names extracted
        new_df[0] = new_df[0] + new_df[1]
        del new_df[1]

        # Drop all the Non-Numeric values from column B considering to retain max nutrient measure values
        new_df[2] = new_df[2].str.extract('(\d+)', expand=False)
        new_df = new_df.dropna()

        new_df[2] = pd.to_numeric(new_df[2], downcast="float")

        list_one = new_df[0].tolist()

        Total = 0
        for i in list_one:
            if 'TotalFat' in i:
                r1 = new_df[new_df[0].str.contains('TotalFat') == True].index.values.astype(int)[0]
                tFat = new_df.loc[r1, 2] * (- 0.0538)
                Total = tFat

            if 'Saturated' in i:
                r2 = new_df[new_df[0].str.contains('Saturated') == True].index.values.astype(int)[0]
                sFat = new_df.loc[r2, 2] * (- 0.423)
                Total = Total + sFat

            if 'Cholesterol' in i:
                r3 = new_df[new_df[0].str.contains('Cholesterol') == True].index.values.astype(int)[0]
                ch = new_df.loc[r3, 2] * (- 0.00398)
                Total = Total + ch

            if 'Sodium' in i:
                r4 = new_df[new_df[0].str.contains('Sodium') == True].index.values.astype(int)[0]
                s = new_df.loc[r4, 2] * (- 0.00254)
                Total = Total + s

            if 'Fiber' in i:
                r6 = new_df[new_df[0].str.contains('Fiber') == True].index.values.astype(int)[0]
                fib = new_df.loc[r6, 2] * (+ 0.561)
                Total = Total + fib

            if 'Protein' in i:
                r8 = new_df[new_df[0].str.contains('Protein') == True].index.values.astype(int)[0]
                pro = new_df.loc[r8, 2] * (+ 0.123)
                Total = Total + pro

            if 'VitaminA' in i:
                r9 = new_df[new_df[0].str.contains('VitaminA') == True].index.values.astype(int)[0]
                vitA = new_df.loc[r9, 2] * (+ 0.00562)
                Total = Total + vitA

            if 'VitaminC' in i:
                r10 = new_df[new_df[0].str.contains('VitaminC') == True].index.values.astype(int)[0]
                vitC = new_df.loc[r10, 2] * (+ 0.0137)
                Total = Total + vitC

            if 'Calcium' in i:
                r11 = new_df[new_df[0].str.contains('Calcium') == True].index.values.astype(int)[0]
                cal = new_df.loc[r11, 2] * (+ 0.0685)
                Total = Total + cal

            if 'Iron' in i:
                r12 = new_df[new_df[0].str.contains('Iron') == True].index.values.astype(int)[0]
                ir = new_df.loc[r12, 2] * (- 0.0186)
                Total = Total + ir

            if 'Sugar' in i:
                r7 = new_df[new_df[0].str.contains('Sugar') == True].index.values.astype(int)[0]
                su = new_df.loc[r7, 2] * (- 0.0245)
                Total = Total + su

            if 'Carb' in i:
                r5 = new_df[new_df[0].str.contains('Carb') == True].index.values.astype(int)[0]
                carb = new_df.loc[r5, 2] * (- 0.0300)
                Total = Total + carb

        # Compare Food Grade
        grade = ""
        if Total >= 1.1:
            grade = 'A'
        if Total > 0.5 and Total <= 1.0:
            grade = 'A-'
        if Total > 0 and Total <= 0.5:
            grade = 'B+'
        if Total > -0.5 and Total <= 0:
            grade = 'B'
        if Total > -1 and Total <= -0.5:
            grade = 'B-'
        if Total > -1.5 and Total <= -1:
            grade = 'C+'
        if Total > -2 and Total <= -1.5:
            grade = 'C'
        if Total > -2.5 and Total <= -2:
            grade = 'C-'
        if Total > -3.0 and Total <= -2.5:
            grade = 'D+'
        if Total <= -3.0:
            grade = 'D'

        final_df = pd.DataFrame()
        for row in new_df.iterrows():
            final_df[list(row[1])[0]] = [list(row[1])[1]]

        final_df["Total points"] = [Total]
        final_df["Grade"] = [grade]
        final_df.insert(0, "Date", datetime.now().strftime("%d-%m-%Y, %H:%M:%S"))
        final_df.insert(0, "Label", "Nutrition label")

        final_list = final_df.to_dict('records')

        final_string = json.dumps(final_list[0])

        json_dict[final_string] = file_name

    return json_dict


def get_cropped_image_text(image_path, image_name):
    """
    From original image, crop, extract text and calculate grading
    :param image_path: Location of original image
    :param image_name: File name
    :return: Dictionary of json string as key, file location as value. JSON string
    will be text extracted from nutrition label.
    """
    cropped_image_path_list = predict_and_crop_label(image_path, image_name)
    return recognition(cropped_image_path_list)


def predict_fruit(image_as_nparray):
    """

    :param image_as_nparray: Image from video camera converted to ndarray
    :return: Array of probability for each of 5 fruits
    """
    resized_image = cv2.resize(image_as_nparray, (IMG_WIDTH, IMG_HEIGHT))
    recolored_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    final_image = recolored_image.reshape((1,) + recolored_image.shape)

    return models.load_model("vgg_model_ga.h5").predict(final_image)


if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True, port="5000")
