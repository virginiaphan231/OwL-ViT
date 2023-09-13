from model import *
from data import *
from losses import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import pprint
from transformers import OwlViTProcessor
import matplotlib.pyplot as plt
import json
import os
import shutil
import yaml
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.io import write_png
from util import PostProcess

from train_util import (
    coco_to_model_input,
    labels_to_classnames,
    model_output_to_image,
    update_metrics
)
from util import BoxUtil, GeneralLossAccumulator, ProgressFormatter


def get_training_config():
    with open("config.yaml", "r") as stream:
        data = yaml.safe_load(stream)
        return data["training"]


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True).to(device)
    scaler = torch.cuda.amp.GradScaler()
    general_loss = GeneralLossAccumulator()
    progress_summary = ProgressFormatter()

    if os.path.exists("debug"):
        shutil.rmtree("debug")
    training_cfg = get_training_config()
    
    
    train_dataloader, test_dataloader, scales, labelmap = get_dataloaders()

    model = OwlViTForObjectDetectionModel.from_pretrained("google/owlvit-base-patch32")
    model.to(device)
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")

    class_loss_coef, bbox_loss_coef, giou_loss_coef = training_cfg["class_loss_coef"], training_cfg["bbox_loss_coef"], training_cfg["giou_loss_coef"]


    criterion = Loss(n_classes= len(labelmap),scales= None,class_loss_coef= class_loss_coef, bbox_loss_coef=bbox_loss_coef, giou_loss_coef=giou_loss_coef)

    postprocess = PostProcess(
        confidence_threshold=training_cfg["confidence_threshold"],
        iou_threshold=training_cfg["iou_threshold"],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=training_cfg["weight_decay"],
    )
    model.train()
    classMAPs = {v: [] for v in list(labelmap.values())}

    for epoch in range(training_cfg["n_epochs"]):
        if training_cfg["save_eval_images"]:
            os.makedirs(f"debug/{epoch}", exist_ok=True)

        # Train loop
        losses = []
        for i, (image, labels, text_labels, boxes, text_queries, metadata) in enumerate(
            tqdm(train_dataloader, ncols=60)):
            # train_dataloader
            optimizer.zero_grad()

            # Prep inputs
            image = image.to(device)
            
            labels = labels.to(device)
            text_labels = text_labels
            convert_text_queries = [item[0] for item in text_queries]
            
            # Converting boxes from COCO format [xywh] to [cxcywh] normalize by image size
            boxes = coco_to_model_input(boxes, metadata).to(device)
            # print("target boxes", boxes)

            inputs = processor(images = image, text= convert_text_queries, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            # import pdb;pdb.set_trace()
            outputs = model(**inputs)

            # Get predictions and save output
            pred_logits, pred_boxes = outputs['logits'], outputs['pred_boxes']
            # print("pred_boxes", pred_boxes)
            #print("pred_boxes shape", pred_boxes.shape)
            losses = criterion(pred_logits, labels, pred_boxes, boxes)
            loss = (
                losses["loss_ce"]
                + losses["loss_bg"]
                + losses["loss_bbox"] 
                + losses["loss_giou"]
            )
            
            loss.backward()
            optimizer.step()

            general_loss.update(losses)

        train_metrics = general_loss.get_values()
        general_loss.reset()


        # Eval loop
        model.eval()
        with torch.no_grad():
            for i, (image, labels, text_labels, boxes, text_queries, metadata) in enumerate(
                tqdm(train_dataloader, ncols=60)
            ):
                # Prep inputs
                image = image.to(device)
                
                labels = labels.to(device)
                text_labels = text_labels
                convert_text_queries = [item[0] for item in text_queries]
                # Converting boxes from COCO format [xywh] to [cxcywh] 
                boxes = coco_to_model_input(boxes, metadata).to(device)
                #print("boxes shape", boxes.shape)


                inputs = processor(images = image, text= convert_text_queries, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)

                # Get predictions and save output
                pred_logits, pred_boxes = outputs['logits'], outputs['pred_boxes']
                
                pred_boxes, pred_classes, scores = postprocess(pred_boxes, pred_logits)
                        
               
                update_metrics(metric,
                               metadata,
                               pred_boxes,
                               pred_classes, 
                               scores,
                               boxes,
                               labels)

                # if training_cfg["save_eval_images"]:
                #     pred_classes_with_names = labels_to_classnames(
                #         pred_classes, labelmap
                #     )
                #     pred_boxes = model_output_to_image(pred_boxes.cpu(), metadata)
                #     image_with_boxes = BoxUtil.draw_box_on_image(
                #         metadata["impath"].pop(),
                #         pred_boxes,
                #         pred_classes_with_names,
                #     )

                #     write_png(image_with_boxes, f"debug/{epoch}/{i}.jpg")

        print("Computing metrics...")
        val_metrics = metric.compute()
        for i, p in enumerate(val_metrics["map_per_class"].tolist()):
            label = labelmap[str(i)]
            classMAPs[label].append(p)

        with open("class_maps.json", "w") as f:
            json.dump(classMAPs, f)

        metric.reset()
        progress_summary.update(epoch, train_metrics, val_metrics)
        progress_summary.print()







