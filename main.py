# main.py

from scripts.onnx_detector import OnnxDetector
from scripts.map_evaluator import MAPFinder

if __name__ == "__main__":
    model_onnx = "model/name_of_your_model.onnx"
    test_dataset = "Test dataset path"
    Classes = ["vehicle"]  # class name
    gt_path = "input/ground-truth"
    dt_path = "input/detection-results"
    output_json_path = r"input/map.json"

    detector = OnnxDetector(model_onnx, Classes)
    detector.mAP_input_data(model_onnx, Classes, test_dataset, gt_path, dt_path)

    evaluator = MAPFinder(gt_path, dt_path, output_json_path)
    mAP, ap_dict = evaluator.evaluate_mAP()

    print("mAP:", mAP)
    print("Class-wise APs:", ap_dict)
