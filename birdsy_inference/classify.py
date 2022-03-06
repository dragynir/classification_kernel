"""
Script to run a classification on one image
Run using:
python classify.py
"""

import json
import cv2
from PIL import Image
from numpy import asarray
import os
import argparse
import pika
import requests
from requests import Request, Session
from io import StringIO, BytesIO
import validators
import boto3
# from botocore.exceptions import ProfileNotFound,
import botocore
import glob
import time
import numpy as np
import random


from inference_docker import *


parser = argparse.ArgumentParser()

parser.add_argument("--input_image_path", help="path to darknet input image in .png or .jpg format")
parser.add_argument("--json_output_path",
                    help="path to output folder where .json with classifciation results is stored", default='./')
parser.add_argument("--number_of_output_classes", type=int, help="first n classes to return", default=40)

parser.add_argument("--rabbitmq_host", help="host of rabbitmq", default='rabbitmq.birdsy.com')
parser.add_argument("--rabbitmq_port", help="port of rabbitmq", default=5672)
parser.add_argument("--rabbitmq_queue", help="queue of rabbitmq", default='birds')
parser.add_argument("--rabbitmq_user", help="user of rabbitmq", default='birds_classify')
parser.add_argument("--rabbitmq_pw", help="password of rabbitmq", default='ahLaing2mohj1oni')

parser.add_argument("--user_server", help="user name of upload server", default='birdsycam')
parser.add_argument("--pw_server", help="password for upload server", default='cgwUK8ttMmPqL8Lb')
args = parser.parse_args()

cfg_path = 'source/base.yml'
checkpoint_path = 'source/epoch=21_val_loss=0.0823.ckpt'
labels_path = 'source/birdsy_labels.txt'


def is_corrupted(img):
    try:
        image = Image.open(img)
        image.verify()
        image.close()  # reload is necessary after verify
        image = Image.open(img)
        image.transpose(Image.FLIP_LEFT_RIGHT)
        image.close()
        return False
    except:
        return True


def main():
    print('Classifying image.')

    ### only load once
    # load model

    predictor = ClassificationPredictor(cfg_path, labels_path, checkpoint_path)

    print('Predictor activation complete')


    ### S3
    s3 = boto3.resource('s3', endpoint_url='https://s3.us-east.gcorelabs.com',
                        config=botocore.client.Config(signature_version=botocore.UNSIGNED))  ## can take outside?

    # rabbitMQ connection for receiving
    credentials = pika.PlainCredentials(args.rabbitmq_user, args.rabbitmq_pw)
    parameters = pika.ConnectionParameters(args.rabbitmq_host,
                                           args.rabbitmq_port,
                                           '/',
                                           credentials)
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()
    channel.queue_declare(queue=args.rabbitmq_queue)

    # that stuff will be repeated in queue
    def callback(ch, method, properties, body):

        message = json.loads(body)
        print(f'Incoming Message: {message}')

        classification_results = {
            'video_id': message['video_id'],
            'objects': [],
            'bird_count': 0
        }

        for obj in message['data']:

            url = obj['image_url']
            u = url.split('/')
            if validators.url(url):
                object_id = obj['object_id']
                detected_at = obj['detected_at']
                index = obj['index']
                filename = url.split("/")[-1]
                # print(object_id,filename)
                BUCKET_NAME = 'birdsy-uploads'
                OBJECT_NAME = '/'.join(u[4:])  #
                obj = s3.Object(BUCKET_NAME, OBJECT_NAME)

                x = np.fromstring(obj.get()['Body'].read(), dtype='uint8')
                image = cv2.imdecode(x, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                class_ids = list()

                #TODO here input call effnet
                result_effnet = predictor.predict_image(image)
                classification_results['objects'].append(result_effnet)

                '''
                try:

                        if result_effnet == 'good':
                            result = result_effnet
                        else:
                            print('nothing detected')

                        classification_results['objects'].append({'object_id': object_id,
                                                                  'detected_at': detected_at,
                                                                  'index': index,
                                                                  'data': result})
                        classification_results['bird_count'] += 1


                    except botocore.exceptions.ClientError as e:

                        if e.response['Error']['Code'] == "404":
                            print(f'Image not found. Status Code 404')
                            classification_results['objects'].append({'object_id': object_id,
                                                                  'data': f'Image not found. Status Code 404'})

                        else:
                            print(f'Invalid url: {url}')
                            classification_results['objects'].append({'object_id': object_id,
                                                                  'data': f'Invalid url.'})
                '''

        ### here goes where to send the results
        url = f"https://birdsy.com/api/video/v2/videos/{message['video_id']}"
        headers = {"content-type": "application/json"}  # , "Authorization": "<auth-key>" }

        data = json.dumps(classification_results)
        r = requests.put(url, data=data, headers=headers)
        print(f'Status code {r.status_code}')

        print(json.dumps(classification_results))
        # return json.dumps(classification_results)

    channel.basic_consume(queue=args.rabbitmq_queue, on_message_callback=callback, auto_ack=True)

    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()


if __name__ == "__main__":
    main()