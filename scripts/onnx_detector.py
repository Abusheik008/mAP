import onnxruntime
import numpy as np
import os,shutil
import cv2
from tqdm import tqdm
from scripts.reusable_code import Reusable_code


class OnnxDetector:
    def __init__(self, model_weights, classes):
        # Constructor to initialize the OnnxDetector class
        # Sets up the ONNX runtime InferenceSession and related attributes
        self.net = onnxruntime.InferenceSession(model_weights)
        self.classes = classes
        self.layer_names = self.net.get_modelmeta()
        self.inputlayers = self.net.get_inputs()[0].name
        self.outputlayers = [self.net.get_outputs()[0].name, self.net.get_outputs()[1].name]

    @staticmethod
    def get_output_format(box):
        # Static method to convert bounding box coordinates to a specific format
        x, y, w, h = box
        return int(x), int(y), int(x+w), int(y+h)

    @staticmethod
    def txt_format(height, width, classes, box, key):
        # Static method to convert bounding box coordinates to a specific text format
        H = height
        W = width
        x, y, w, h = box
        w = str(round(float(w) / W, 5))
        h = str(round(float(h) / H, 5))
        x = str(round((float(x) / W) + (float(w) / 2), 5))
        y = str(round((float(y) / H) + (float(h) / 2), 5))
        annotation_list = ' '.join([str(classes.index(key)), x, y, w, h])
        return annotation_list

    def detect(self, img, conf=0.2, nms_thresh=0.2, non_max_suppression=True, class_conf=None):
        # Method for object detection using the ONNX model
        threshold = conf
        if class_conf is None:
            class_conf = []
        if len(class_conf) < len(self.classes):
            conf = {k:conf for k in self.classes}
        else:
            conf = class_conf
        class_conf_dict = {k: conf[k] for i, k in enumerate(self.classes)}
        final_result = {k: [] for k in self.classes}
        final_list = [ ]
        fin_same = {k:[] for k in self.classes}
        confidences = {k: [] for k in self.classes}
        boxes = {k: [] for k in self.classes}
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), (0, 0, 0), True, crop=False)
        layers_result = self.net.run([self.outputlayers[0], self.outputlayers[1]],
                                             {self.inputlayers: blob})
        outputs = np.concatenate([layers_result[1], layers_result[0]], axis=1)
        height, width, _ = img.shape
        matches = outputs[np.where(np.max(outputs[:, 4:], axis=1) > threshold)]
        for detect in matches:
            scores = detect[4:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            center_x = int(detect[0] * width)
            center_y = int(detect[1] * height)
            w = int(detect[2] * width)
            h = int(detect[3] * height)
            x = int(center_x - w/2)
            y = int(center_y - h / 2)
            confidences[self.classes[class_id]].append(float(confidence))
            boxes[self.classes[class_id]].append([int(i) for i in [x, y, w, h]])
        indices = {}
        if non_max_suppression:
            for class_name, box in boxes.items():
                indices[class_name] = cv2.dnn.NMSBoxes(box, confidences[class_name], class_conf_dict[class_name], nms_thresh)
        else:
            for class_name, box in boxes.items():
                indices[class_name] = [[w] for w in range(len(box))]

        for key, index in indices.items():
            for i in index:
                try:
                    select = i[0]
                except:
                    select = i
                final_result[key].append(self.get_output_format(boxes[key][select]))
                annotation_list = self.txt_format(height, width, self.classes, boxes[key][select], key)
                final_list.append(annotation_list)
                fin_same[key].append([self.get_output_format(boxes[key][select]), confidences[key][select]])
        return confidences, boxes

    def mAP_input_data(model_onnx,classes,Test_dataset, destination_path_gt, destination_path_result, img_path_dest):
      # Method to prepare input data for mean Average Precision (mAP) calculation
      os.makedirs(destination_path_gt, exist_ok=True)
      os.makedirs(destination_path_result, exist_ok=True)
      os.makedirs(img_path_dest, exist_ok=True)


      if Test_dataset is not None:
        for dataset in tqdm(os.listdir(Test_dataset), desc= "Finding and Moving Detection Result"):
          if dataset.endswith(".jpg") or dataset.endswith(".png") or dataset.endswith(".jpeg"):
            image_path = os.path.join(Test_dataset,dataset)
            img_dest = os.path.join(img_path_dest,dataset)
            txt_path = os.path.join(Test_dataset,dataset.replace(".jpg", ".txt").replace(".png", ".txt").replace(".jpeg", ".txt"))
            detector = OnnxDetector(model_onnx,classes)
            print(f"The Image Used Is : {image_path}")
            read_img = cv2.imread(image_path)
            outputs = detector.detect(read_img)
            detections_with_conf, detections = outputs
            result_txt_path = os.path.join(destination_path_result,dataset.replace(".jpg", ".txt").replace(".png", ".txt").replace(".jpeg", ".txt"))
            with open(result_txt_path, "w") as f:
                for class_name, class_detections in detections.items():
                    for detection in class_detections:
                        if detection is not None:
                          x1, y1, x2, y2 = detection
                          img_width, img_height= Reusable_code.get_image_size(image_path)
                          dt_x = (x1 + x2) / 2 / img_width
                          dt_y = (y1 + y2) / 2 / img_height
                          dt_width = (x2 - x1) / img_width
                          dt_height = (y2 - y1) / img_height
                          confidence = detections_with_conf[class_name][class_detections.index(detection)]
                          line = f"{class_name} {confidence:.6f} {dt_x:.6f} {dt_y:.6f} {dt_width:.6f} {dt_height:.6f}\n"
                          f.write(line)
                        else:
                          line = f"{class_name} {0} {0} {0} {0} {0}\n"
                          f.write(line)
            # copying txt files
            destination_path = os.path.join(destination_path_gt, dataset.replace(".jpg", ".txt").replace(".png", ".txt").replace(".jpeg", ".txt"))
            shutil.copy(txt_path, destination_path)
            updated_lines = []
            with open(txt_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        parts[0] = "vehicle"
                    updated_line = " ".join(parts) + "\n"
                    updated_lines.append(updated_line)
            shutil.copy(image_path, img_dest)

            with open(destination_path, "w") as f:
                print(f"Updating the line in path :{destination_path} the text : {updated_lines}")
                f.writelines(updated_lines)









