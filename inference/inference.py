from efficientnet_pytorch import EfficientNet
from torch import nn
import pytorch_lightning as pl
import torch
import yaml
from addict import Dict
import cv2
from torch.nn import functional as F
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
import numpy as np

import ttach as tta


import argparse
import glob
import time
import logging
from pathlib import Path

import cv2
import numpy as np
import os
import collections


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s-%(name)s-%(message)s')

file_handler = logging.FileHandler("deneme.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)




class Model(pl.LightningModule):

    def __init__(self, num_classes, architecture, *args, **kwargs):
        super().__init__()
        print('Num classes:', num_classes)
        print('Architecture:', architecture)
        print('before from name')
        self.net = EfficientNet.from_name(architecture)
        print('after from_name')
        self.net._fc = nn.Linear(in_features=self.net._fc.in_features, out_features=num_classes, bias=True)

    def forward(self, x):
        return self.net(x)

class ClassificationTTAWrapper(nn.Module):
    """Wrap PyTorch nn.Module (classification model) with test time augmentation transforms
    Args:
        model (torch.nn.Module): classification model with single input and single output
            (.forward(x) should return either torch.Tensor or Mapping[str, torch.Tensor])
        transforms (ttach.Compose): composition of test time transforms
        merge_mode (str): method to merge augmented predictions mean/gmean/max/min/sum/tsharpen
        output_label_key (str): if model output is `dict`, specify which key belong to `label`
    """

    def __init__(
        self,
        model,
        transforms,
        merge_mode = "mean",
        output_label_key = None,
    ):
        super().__init__()
        self.model = model
        self.transforms = transforms
        self.merge_mode = merge_mode
        self.output_key = output_label_key

    def forward(
        self, image: torch.Tensor, *args
    ):
        merger = Merger(type=self.merge_mode, n=len(self.transforms))

        for transformer in self.transforms:
            augmented_image = transformer.augment_image(image)
            augmented_output = self.model(augmented_image, *args)
            if self.output_key is not None:
                augmented_output = augmented_output[self.output_key]
            deaugmented_output = transformer.deaugment_label(augmented_output)
            merger.append(deaugmented_output)

        result = merger.result
        if self.output_key is not None:
            result = {self.output_key: result}

        return result


class Merger:

    def __init__(
            self,
            type: str = 'mean',
            n: int = 1,
    ):

        if type not in ['mean', 'gmean', 'tsharpen']:
            raise ValueError('Not correct merge type `{}`.'.format(type))

        self.output = None
        self.type = type
        self.n = n
        self.all_outputs = []

    def append(self, x):

        if self.type == 'tsharpen':
            x = x ** 0.5

        if self.output is None:
            self.output = x
        elif self.type in ['mean', 'tsharpen']:
            self.output = self.output + x
        elif self.type == 'gmean':
            self.output = self.output * x

        self.all_outputs.append(F.softmax(x, dim=1))

    @property
    def result(self):

        if self.type in ['mean', 'tsharpen']:
            result = self.output / self.n
        elif self.type in ['gmean']:
            result = self.output ** (1 / self.n)
        else:
            raise ValueError('Not correct merge type `{}`.'.format(self.type))

        all_results = torch.stack(self.all_outputs, dim=1)

        return F.softmax(result, dim=1), all_results


class ClassificationPredictor():

    def __init__(self, cfg_path, labels_path, checkpoint_path, use_tta=True):

        self.use_tta = use_tta
        self.opt = ClassificationPredictor.__load_cfg(cfg_path)
        print('Predictor options downloaded')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'We are on device:{self.device}')
        self.model = ClassificationPredictor.__load_model(self.opt, checkpoint_path, self.device, self.use_tta)
        print('Model ready')
        self.labels = ClassificationPredictor.__load_labels(labels_path)
        print('Labels ready')
        self.transforms = ClassificationPredictor.__create_transforms(self.opt)
        print('Transforms setup')

    @staticmethod
    def __load_cfg(cfg_path):
        with open(cfg_path, 'r') as f:
            opt = Dict(yaml.safe_load(f))
        return opt

    @staticmethod
    def __load_labels(labels_path):
        with open(labels_path, 'r') as f:
            labels = f.read().splitlines()
        print(f'Labels {labels_path} loaded')
        return labels

    @staticmethod
    def __load_model(model_cfg, checkpoint_path, device, use_tta):
        model = Model(model_cfg.num_classes, model_cfg.architecture)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device)['state_dict'], strict=False)
        model.eval()
        model.to(device)
        print(f'Model {checkpoint_path} loaded')

        if use_tta:
            transforms = tta.Compose(
                [
                    tta.HorizontalFlip(),
                    tta.Scale(scales=[1, 2]),
                ]
            )

            model = ClassificationTTAWrapper(model, transforms, merge_mode='mean')
            print(f'Model uses test time augmentations')

        return model

    @staticmethod
    def __create_transforms(opt):
        transforms =  A.Compose([
                A.Resize(height=opt.resolution, width=opt.resolution, p=1.0),
                A.Normalize(),
                ToTensorV2(),
            ], p=1.0)
        print('Transforms completed')
        return transforms

    @staticmethod
    def __indexes2labels(labels, indexes):
        return list(map(lambda x: labels[x], indexes))

    def predict_image(self, bgr_image, top_k=4):

        sample = self.transforms(image=bgr_image)
        image_tr  = sample['image']
        image_tr = torch.unsqueeze(image_tr, dim=0)

        with torch.no_grad():

            if self.use_tta:
                probs, all_probs = self.model(image_tr.to(self.device))
            else:
                logits = self.model(image_tr.to(self.device))
                probs = F.softmax(logits, dim=1)

            top_probs, top_labels = torch.topk(probs, top_k, dim=1)
            y_pred = top_labels.squeeze().cpu().numpy()

            if self.use_tta:
                top_confidence = torch.index_select(all_probs, dim=-1, index=top_labels.squeeze())
                top_probs, _ = torch.max(top_confidence, dim=1)

            probs = list(map(str, top_probs.squeeze().cpu().numpy()))

        pred_labels = ClassificationPredictor.__indexes2labels(self.labels, y_pred)
        results = {lb:p for lb, p in zip(pred_labels, probs)}

        return results

def extract_vall_loss(name):
    return float(name.split('=')[-1].replace('.ckpt', ''))

def get_best_checkpoint(checkpoint_path):

    checkpoints = os.listdir(checkpoint_path)
    checkpoints = list(filter(lambda x: 'tmp' not in x, checkpoints))

    best_ckpt = np.argmin(map(extract_vall_loss, checkpoints))

    return os.path.join(checkpoint_path, checkpoints[best_ckpt])


def detect(origin_frm, frm, net, net_size, layer, class_net):
    (H, W) = origin_frm.shape[:2]
    blob = cv2.dnn.blobFromImage(frm, 1/255.0, (net_size, net_size),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start_time = time.time()
    layerOutputs = net.forward(layer)
    end_time = time.time()

    boxes = []
    classIds = []
    confidences = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > CONFIDENCE_THRESHOLD:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width/2))
                y = int(centerY - (height/2))

                boxes.append([x, y, int(width), int(height)])
                classIds.append(classID)
                confidences.append(float(confidence))

    idxs = cv2.dnn.NMSBoxes(
        boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    top_detected_class = []

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            x = max(0, x)
            y = max(0, y)

            best_class = 0
            best_class_prob = 0

            if class_net:
                frm_crop = origin_frm[y:y+h, x:x+w]

                logger.info('{}, {}, {}, {}'.format(x, y, w, h))

                frm_crop = cv2.cvtColor(frm_crop, cv2.COLOR_BGR2RGB)
                top_classes = class_net.predict_image(frm_crop)
                top_classes = {k: round(float(v), 4) for k, v in top_classes.items()}

                best_class = max(top_classes, key=top_classes.get)

                logger.info(top_classes)
                best_class_prob = top_classes[best_class]

                top_detected_class.append(best_class)

            # color = [int(c) for c in COLORS[classIds[i]]]
            cv2.rectangle(origin_frm, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # class_names[classIds[i]], confidences[i],
            text = "{}, {:.4f}".format(best_class, best_class_prob)

            labelSize, baseLine = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
            label_ymin = max(y, labelSize[1] + 10)  # Make sure not to draw label too close to top of window
            cv2.rectangle(origin_frm, (x, label_ymin - labelSize[1] - 10), (x + labelSize[0], label_ymin + baseLine - 10),
                          (0, 255, 0), cv2.FILLED)  # Draw white box to put label text in
            cv2.putText(origin_frm, text, (x, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),
                        2)  # Draw label text

            fps_label = "FPS: %.2f" % (1 / (end_time - start_time))
            cv2.putText(origin_frm, fps_label, (0, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

        detected_classes = np.unique(top_detected_class)
        logger.info('Detected classes: ' + str(detected_classes))

        return detected_classes

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, default="../../test_random_source/video/sample.mp4",
                    help="path to input video file")
parser.add_argument("-o", "--output", type=str, default="",
                    help="path to (optional) output video file")
parser.add_argument("-d", "--display", action='store_true',
                    help="display output or not")
parser.add_argument("-od", "--od_only", action='store_true',
                    help="only object detection")
parser.add_argument("-cl", "--classification_only", action='store_true',
                    help="only classification")
parser.add_argument("-s", "--size", type=int, default=640,
                    help="model input size")
parser.add_argument("-mf", "--max_frames", type=int, default=np.inf,
                    help="max frames to process")
parser.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="confidence threshold")
parser.add_argument("-t", "--threshold", type=float, default=0.4,
                    help="non-maximum supression threshold")


if __name__ == '__main__':
    args = vars(parser.parse_args())
    logger.info("Parsed Arguments")

    if args['output'] == "":
        args['output'] = os.path.join("../../test_random_source/results/", \
            os.path.basename(args['input']))

    logger.info('Output will be here: ' + args['output'])

    # detection model
    CONFIDENCE_THRESHOLD = args["confidence"]
    NMS_THRESHOLD = args["threshold"]
    if not Path(args["input"]).exists():
        raise FileNotFoundError(
            "Path to video file is not exist.")

    logger.info('Reading source: ' + args["input"])

    vc = cv2.VideoCapture(args["input"])
    weights = glob.glob("../../detection_ckpts/half_model/*.weights")[0]
    labels = glob.glob("../../detection_ckpts/half_model/*.txt")[0]
    cfg = glob.glob("../../detection_ckpts/half_model/*.cfg")[0]

    logger.info("Using {} weights ,{} configs and {} labels.".format(
        weights, cfg, labels))

    class_names = list()
    with open(labels, "r") as f:
        class_names = [cname.strip() for cname in f.readlines()]

    COLORS = np.random.randint(0, 255, size=(len(class_names), 3), dtype="uint8")

    net = cv2.dnn.readNetFromDarknet(cfg, weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    layer = net.getLayerNames()
    layer = [layer[i - 1] for i in net.getUnconnectedOutLayers()]
    writer = None

    class_predictor = None
    logger.info(f'Od only:' + str(args['od_only']))
    # # classification model
    if not args['od_only']:
        experiment = 'exp6'
        exp_path = f'/home/mborisov/CLM/experiments/us/{experiment}'
        cfg_path = f'{exp_path}/config.yml'
        labels_path = f'{exp_path}/dataset/labelmap.txt'
        checkpoint_path = get_best_checkpoint(f'{exp_path}/checkpoint/')
        logger.info(f'Use classificatioin checkpoint: {checkpoint_path}')

        # use preloaded model
        class_predictor = ClassificationPredictor(cfg_path, labels_path, checkpoint_path, use_tta=False)

    frame_count = 0

    detected_classes = []

    while True:
        (grabbed, frame) = vc.read()
        if not grabbed:
            break

        logger.info(f'Process frame: {frame_count}')
        frame_count+=1

        resized_frame = cv2.resize(frame, (args["size"], args["size"]))
        frame_classes = detect(frame, resized_frame, net, args['size'], layer, class_predictor)

        if frame_classes is not None:
            detected_classes.extend(frame_classes)

        if args["display"]:
           cv2.imshow("detections", frame)
           cv2.waitKey(1)

        if args["output"] != "" and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 25,
                                     (frame.shape[1], frame.shape[0]), True)

        if writer:
            writer.write(frame)
        if frame_count >= args['max_frames']:
            break

    logger.info('Detected classes in source (unique on frames): ' + str(collections.Counter(detected_classes)))

    if writer:
        writer.release()

    if vc:
        vc.release()
