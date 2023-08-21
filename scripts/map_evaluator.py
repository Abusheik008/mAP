import glob,os
import json

class MAPFinder:
    # Class to calculate mean Average Precision (mAP) for object detection
    def __init__(self, annotation_path, results_path, output_json_path):
        # Constructor to initialize paths for annotation, detection results, and output JSON
        self.annotation_path = annotation_path
        self.results_path = results_path
        self.output_json_path = output_json_path

    def parse_annotations(self):
        # Parse ground truth annotation files and store the information
        all_annotations = {}
        for txt_file in glob.glob(self.annotation_path + '/*.txt'):
            image_name = os.path.basename(txt_file).replace('.txt', '')
            image_annotations = []

            with open(txt_file, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                obj_info = {
                    'class': parts[0],
                    'bbox': [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
                }
                image_annotations.append(obj_info)

            all_annotations[image_name] = image_annotations

        return all_annotations

    def parse_detection_results(self):
        # Parse detection result files and store the information
        detection_results = {}
        for txt_file in glob.glob(self.results_path + '/*.txt'):
            image_name = os.path.basename(txt_file).replace('.txt', '')
            image_detections = []

            with open(txt_file, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                obj_info = {
                    'class': parts[0],
                    'confidence': float(parts[1]),
                    'bbox': [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
                }
                image_detections.append(obj_info)

            detection_results[image_name] = image_detections

        return detection_results

    def calculate_iou(self,bbox1, bbox2):
        # Calculate Intersection over Union (IoU) between two bounding boxes
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))

        intersection = x_overlap * y_overlap
        union = w1 * h1 + w2 * h2 - intersection

        iou = intersection / union
        return iou

    def calculate_ap(self,recall, precision):
        # Calculate Average Precision (AP) using recall and precision valuesq
        recall.insert(0, 0.0)
        recall.append(1.0)
        precision.insert(0, 0.0)
        precision.append(0.0)

        for i in range(len(precision) - 2, -1, -1):
            precision[i] = max(precision[i], precision[i + 1])

        ap = 0.0
        for i in range(len(recall) - 1):
            ap += (recall[i + 1] - recall[i]) * precision[i + 1]

        return ap

    def evaluate_mAP(self):
        # Evaluate mean Average Precision (mAP) and class-wise APs
        ground_truth = self.parse_annotations()
        detection_results = self.parse_detection_results()

        all_classes = set()
        for image_annotations in ground_truth.values():
            for obj_info in image_annotations:
                all_classes.add(obj_info['class'])
        all_classes = sorted(list(all_classes))

        ap_dict = {}
        for class_name in all_classes:
            true_positives = []
            false_positives = []

            for image_name, image_annotations in ground_truth.items():
                image_detections = detection_results.get(image_name, [])
                detected_objects = set()

                for obj_info in image_annotations:
                    if obj_info['class'] != class_name:
                        continue
                    best_iou = 0.0
                    best_detection = None

                    for det_info in image_detections:
                        if det_info['class'] != class_name:
                            continue
                        iou = self.calculate_iou(obj_info['bbox'], det_info['bbox'])
                        if iou > best_iou:
                            best_iou = iou
                            best_detection = det_info

                    if best_detection is not None and best_iou >= 0.1:
                        det_tuple = (
                            best_detection['class'],
                            tuple(best_detection['bbox'])
                        )
                        if det_tuple not in detected_objects:
                            true_positives.append(1)
                            detected_objects.add(det_tuple)
                        else:
                            false_positives.append(1)
                    else:
                        false_positives.append(1)

            num_ground_truth = len(ground_truth)
            num_detections = len(detection_results)
            num_true_positives = sum(true_positives)
            num_false_positives = sum(false_positives)


            recall = num_true_positives / num_ground_truth if num_ground_truth > 0 else 0
            precision = num_true_positives / (num_true_positives + num_false_positives) if (num_true_positives + num_false_positives) > 0 else 0
            ap = self.calculate_ap([recall], [precision])
            ap_dict[class_name] = ap

        mAP = sum(ap_dict.values()) / 1
        rounded_ap_dict = {class_name: round(ap, 3) for class_name, ap in ap_dict.items()}

        results = {
            'num_ground_truth': num_ground_truth,
            'num_detections': num_detections,
            'num_true_positives': num_true_positives,
            'num_false_positives': num_false_positives,
            'mAP': round(mAP, 3),
            'class_map': rounded_ap_dict,
            'model_new':False
        }

        if os.path.exists(self.output_json_path):
            with open(self.output_json_path, 'r') as json_file:
              current_results = json.load(json_file)
            current_mAP = current_results['mAP']
            if round(mAP, 3) > current_mAP:
              results_new = {
                  'num_ground_truth': num_ground_truth,
                  'num_detections': num_detections,
                  'num_true_positives': num_true_positives,
                  'num_false_positives': num_false_positives,
                  'mAP': round(mAP, 3),
                  'class_map': rounded_ap_dict,
                  'model_new': True
              }
              with open(self.output_json_path, 'w') as json_file:
                json.dump(results_new, json_file, indent=4)
            else:
              with open(self.output_json_path, 'w') as json_file:
                  json.dump(results, json_file, indent=4)

        return mAP, ap_dict
    


    
