import argparse
import torch
import os
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from transformers import OwlViTProcessor
from data import CustomCocoDataset, custom_collate_fn
from box_utils import box_cxcywh_to_xywh
from model import OwlViTForObjectDetectionModel

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate your object detection model")
    parser.add_argument("--eval_ann_file", type=str, help="Path to evaluation annotation file")
    parser.add_argument("--test_data_dir", type=str, help="Path to test data directory")
    parser.add_argument("--checkpoint_path", type=str, help="Path to trained model checkpoint")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--num_worker", type=int, default=0, help="Number of workers for data loading")
    args = parser.parse_args()
    return args

def get_normalized_predictions(model, images, batch_text_labels, device):
    inputs = processor(images=images, text=batch_text_labels, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}  

    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
        pred_logits = outputs["logits"]
        pred_probs = torch.sigmoid(pred_logits)
        pred_labels = pred_probs.argmax(dim=-1)
        pred_boxes = box_cxcywh_to_xywh(outputs["pred_boxes"])

    return pred_boxes, pred_probs, pred_labels

def evaluate_model(test_data_dir, eval_ann_file, checkpoint_path, batch_size, num_worker):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    coco_gt = COCO(eval_ann_file)
    coco_dt = []

    test_dataset = CustomCocoDataset(annotation_file=eval_ann_file, data_dir=test_data_dir, is_train=False)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker, collate_fn=custom_collate_fn)

    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetectionModel.from_pretrained("google/owlvit-base-patch32")
    
    # Load the trained model checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    with tqdm(test_loader) as t:
        for image_idx, (images, targets) in enumerate(t):
            images = [img.to(device) for img in images]
            batch_text_labels = [target["text_labels"] for target in targets]

            pred_boxes, pred_probs, pred_labels = get_normalized_predictions(model, images, batch_text_labels, device)

            for box, score, label in zip(pred_boxes, pred_probs, pred_labels):
                coco_dt.append({"image_id": targets[image_idx]["image_id"],  # Use the image ID corresponding to the current image
                                "category_id": label.item(),
                                "bbox": box.tolist(),
                                "score": score.item()})

                        
    with open("coco_detection_results.json", "w") as json_file:
        json.dump(coco_dt, json_file)

    coco_res = coco_gt.loadRes("coco_detection_results.json")
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

if __name__ == "__main__":
    args = parse_args()
    evaluate_model(args.test_data_dir, args.eval_ann_file, args.checkpoint_path, args.batch_size, args.num_worker)

    # Plot the results
    plt.bar(results.keys(), results.values())
    plt.xlabel('Evaluation Metric')
    plt.ylabel('Score')
    plt.title('Object Detection Evaluation Results')
    plt.show()

