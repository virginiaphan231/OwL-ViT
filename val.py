import argparse
import torch
import os
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from transformers import OwlViTProcessor
from data import *
from box_utils import *
from model import OwlViTForObjectDetectionModel
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate your object detection model")
    parser.add_argument("--eval_ann_file", type=str, help="Path to evaluation annotation file")
    parser.add_argument("--test_data_dir", type=str, help="Path to test data directory")
    parser.add_argument("--checkpoint_path", type=str, help="Path to trained model checkpoint")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--num_worker", type=int, default=0, help="Number of workers for data loading")
    parser.add_argument("--plot_path", type=str, help="Path to save Evaluation plot")
    args = parser.parse_args()
    return args

def get_normalized_predictions(model, images, batch_text_labels, device):
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    inputs = processor(images=images, text=batch_text_labels, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}  

    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
        pred_logits = outputs["logits"]
        pred_probs = torch.sigmoid(pred_logits)
        pred_labels = pred_probs.argmax(dim=-1)
        pred_boxes = box_cxcywh_to_xywh(outputs["pred_boxes"])

    return pred_boxes, pred_probs, pred_labels

def evaluate_model(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    coco_gt = COCO(args.eval_ann_file)
    coco_dt = []

    test_dataset = CustomCocoDataset(annotation_file=args.eval_ann_file, data_dir=args.test_data_dir, is_train=False)
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker, collate_fn=custom_collate_fn)

    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetectionModel.from_pretrained("google/owlvit-base-patch32")
    model.to(device)
    
    # Load the trained model checkpoint
    if os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
    else:
        print("Model checkpoint not found. Please provide a valid checkpoint path.")
        return

    with tqdm(test_loader) as t:
        for images, targets in t:
            images = [img.to(device) for img in images]
            batch_text_labels = [target["text_labels"] for target in targets]
            batch_image_ids = [target["image_id"] for target in targets]  # Get the image_id for each image in the batch

            pred_boxes, pred_probs, pred_labels = get_normalized_predictions(model, images, batch_text_labels, device)

            for image_id, box, score, label in zip(batch_image_ids,pred_boxes, pred_probs, pred_labels):
                coco_dt.append({"image_id": image_id,  # Use the image ID corresponding to the current image
                                "category_id": label.tolist(),
                                "bbox": box.tolist(),
                                "score": score.tolist()})
            print(coco_dt)

    output_json_path = os.path.join(args.test_data_dir, "coco_detection_results.json")
    with open(output_json_path, "w") as json_file:
        json.dump(coco_dt, json_file)

    coco_res = coco_gt.loadRes(output_json_path)
    coco_eval = COCOeval(coco_gt, coco_res, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Get the results as a dictionary
    evaluation_results = {
        'AP': coco_eval.stats[0],  # Average Precision
        'AP50': coco_eval.stats[1],  # Average Precision at IoU 0.5
        'AP75': coco_eval.stats[2],  # Average Precision at IoU 0.75
        'AP_small': coco_eval.stats[3],  # AP for small objects
        'AP_medium': coco_eval.stats[4],  # AP for medium objects
        'AP_large': coco_eval.stats[5],  # AP for large objects
    }

    return evaluation_results

if __name__ == "__main__":
    args = parse_args()
    evaluation_results = evaluate_model(args)

    if evaluation_results:
        # Plot the results
        plt.bar(evaluation_results.keys(), evaluation_results.values())
        plt.xlabel('Evaluation Metric')
        plt.ylabel('Score')
        plt.title('Object Detection Evaluation Results')
        plt.savefig(fname=args.plot_path, format="png")
        plt.show()



