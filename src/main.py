from PIL import Image
import numpy as np
import cv2
import os
import io
import tensorflow as tf
from tf_model_object_detection import Model
from google.cloud import storage
from tensorflow.keras.models import load_model
from tensorflow import keras
import firebase_admin
from firebase_admin import credentials, messaging


cred = credentials.Certificate("service-account-file.json")
firebase_admin.initialize_app(cred)

storage_client = storage.Client.from_service_account_json('service-account-file.json')
bucket = storage_client.get_bucket('###########Fill Your bucket name#################')    

#defining a message payload to the app.
def sendPush(title, msg, registration_token,image, dataObject=None):
    message = messaging.MulticastMessage(
        notification=messaging.Notification(
            title=title,
            body=msg,
            image=image
        ),
        data=dataObject,
        tokens=registration_token,
    )

    # Send a message to the device corresponding to the provided
    # registration token.
    response = messaging.send_multicast(message)
    # Response is a message ID string.
    print('Successfully sent message:', response)

tokens2=["###########Fill the token code#################"]

def get_human_box_detection(boxes, scores, classes, height, width):
    """
    For each object detected, check if it is a human and if the confidence >> our threshold.
    Return 2 coordonates necessary to build the box.
    @ boxes : all our boxes coordinates
    @ scores : confidence score on how good the prediction is -> between 0 & 1
    @ classes : the class of the detected object ( 1 for human )
    @ height : of the image -> to get the real pixel value
    @ width : of the image -> to get the real pixel value
    """
    human_detected = False

    array_boxes = list()  # Create an empty list
    for i in range(boxes.shape[1]):
        # If the class of the detected object is 1 and the confidence of the prediction is > 0.6
        if int(classes[i]) == 1 and scores[i] > 0.75:
            # Multiply the X coordonnate by the height of the image and the Y coordonate by the width
            # To transform the box value into pixel coordonate values.
            box = [boxes[0, i, 0], boxes[0, i, 1], boxes[0, i, 2], boxes[0, i, 3]] * np.array(
                [height, width, height, width])
            # Add the results converted to int
            array_boxes.append((int(box[0]), int(box[1]), int(box[2]), int(box[3])))
            human_detected = True
    return array_boxes, human_detected


image_size = (180, 180)
COLOR_GREEN = (0, 255, 0)
write_each = 20
frame_number = 0


human_model_file = 'frozen_inference_graph.pb' 
model = Model(human_model_file)
helmet_model_file = '../saved_model_helmet'
model1 = load_model(helmet_model_file, compile=True)

video_names_list = [name for name in os.listdir("../video/") if name.endswith(".MP4") or name.endswith(".avi")]
for index,video_name in enumerate(video_names_list):
    print(" - {} [{}]".format(video_name, index))
    video_path = "../video/"+video_names_list[int(index)]
    vs = cv2.VideoCapture(video_path)

    while True:

        # Load the frame
        (frame_exists, frame) = vs.read()
        # Test if it has reached the end of the video
        if not frame_exists:
            break
        else:
            if not (frame_number % write_each):
                # Make the predictions for this frame
                (boxes, scores, classes) = model.predict(frame)

                # Get the human detected in the frame and return the 2 points to build the bounding box
                array_boxes_detected, human_detected = get_human_box_detection(boxes, scores[0].tolist(),
                                                                               classes[0].tolist(), frame.shape[0],
                                                                               frame.shape[1])
                if human_detected:

                    for index, box in enumerate(array_boxes_detected):
                        human_img = frame[box[0]:box[2], box[1]:box[3]].copy()
                    img = cv2.resize(human_img, image_size)
                    img = Image.fromarray(img)
                    roi_img = img.crop(box)
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='PNG')
                    img_byte_arr = img_byte_arr.getvalue()
                    img_array = keras.preprocessing.image.img_to_array(img)
                    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
                    predictions = model1.predict(img_array)
                    score = predictions[0]
                    print("This image is %.2f percent no and %.2f percent yes." % (100 * (1 - score), 100 * score))
                    if (100 * (1 - score) > 95):
                        cv2.imwrite(f'../img/{frame_number}.jpg', human_img)
                        blob = bucket.blob(f'{frame_number}.jpg')
                        blob.upload_from_string(img_byte_arr)
                        sendPush("opp!!", "a human without an helmet detected", tokens2, f"https://storage.googleapis.com/ ###########Fill Your bucket name#################/{frame_number}.jpg") 

            frame_number += 1

        key = cv2.waitKey(1) & 0xFF

        # Break the loop
        if key == ord("q"):
            break

