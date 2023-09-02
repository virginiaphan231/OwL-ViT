from model import *
from box_utils import *
from data import *
from losses import *
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
import argparse
import random
import numpy as np
from tqdm import tqdm
from transformers import OwlViTProcessor

def parse_args():
    parser = argparse.ArgumentParser(description="Train your model")
    parser.add_argument("--train_data_dir", type=str, default="path/to/train/data", help="Path to training data directory")
    parser.add_argument("--ann_file", type=str, default="path/to/annotations.json", help="Path to annotation file")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--num_worker", type=int, default=0, help="Number of workers for data loading")
    parser.add_argument("--class_loss_coef", type=float, default=1.0, help="Coefficient for class loss")
    parser.add_argument("--bbox_loss_coef", type=float, default=2.0, help="Coefficient for bbox loss")
    parser.add_argument("--giou_loss_coef", type=float, default=2.0, help="Coefficient for GIoU loss")
    parser.add_argument("--focal_loss", type=bool, default=True, help="Whether to use focal loss")
    parser.add_argument("--focal_alpha", type=float, default=0.25, help="Alpha parameter for focal loss")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Gamma parameter for focal loss")
    parser.add_argument("--freeze_encoders", type=bool, default=True, help="Whether to freeze encoders")
    parser.add_argument("--checkpoints_dir", type = str, help = "path to save checkpoints file")
    args = parser.parse_args()
    return args

def train_model(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Set random seed for reproducibility
    random_seed = 42
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_dataset = CustomCocoDataset(annotation_file=args.ann_file, data_dir=args.train_data_dir, is_train=True)
    

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker, collate_fn=custom_collate_fn)
    

    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetectionModel.from_pretrained("google/owlvit-base-patch32")

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))

    epoch_losses = []

    # Freeze layers from vision_model and text_model
    if args.freeze_encoders:
        for param in model.owlvit.text_model.parameters():
            param.requires_grad = False

        for param in model.owlvit.vision_model.parameters():
            param.requires_grad = False

    for epoch in tqdm(range(args.num_epochs), desc="Training Progress", leave=False):
        model.train()
        total_loss = 0.0

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}", leave=False) as t:
            for batch_idx, (images, targets) in enumerate(t):
                images = [img.to(device) for img in images]
                batch_target_boxes = [target["boxes"] for target in targets]
                batch_target_labels = [target["labels"] for target in targets]
                
                # Batch text labels directly from the targets
                batch_text_labels = [target["text_labels"] for target in targets]

                inputs = processor(images=images, text=batch_text_labels, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}  
                
                outputs = model(**inputs, return_dict=True)
                pred_logits = outputs["logits"]
                pred_boxes = outputs["pred_boxes"]

                loss = compute_cost(
                    tgt_labels = batch_target_labels,
                    out_logits = pred_logits,
                    tgt_bbox = batch_target_boxes,
                    out_bbox = pred_boxes,
                    num_classes = 91,
                    class_loss_coef = args.class_loss_coef,
                    bbox_loss_coef = args.bbox_loss_coef,
                    giou_loss_coef = args.giou_loss_coef,
                    focal_loss= args.focal_loss,
                    focal_alpha = args.focal_alpha,
                    focal_gamma = args.focal_gamma
                ).sum()
                loss.backward()
                
                optimizer.step()
                total_loss += loss.item()

                t.set_postfix(loss=total_loss)

        epoch_losses.append(total_loss)
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Loss: {total_loss}")

    # Save checkpoint after training
    checkpoint_dir = args.checkpoints_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "final_model_checkpoint.pt")
    torch.save(model.state_dict(), checkpoint_path)

    # Plotting the training loss over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.num_epochs + 1), epoch_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Over Epochs')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    args = parse_args()
    train_model(args)

