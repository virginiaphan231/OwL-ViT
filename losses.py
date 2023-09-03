from box_utils import *
from model import *
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from typing import Optional, Union, List

def pad_instances(instance_list, padding_value=-1):
    """
    Pad instances in a list of tensors to a common length.

    Args:
    instance_list (list of tensors): List of instances where each instance is a tensor.
    padding_value (int, optional): The value used for padding. Default is -1.

    Returns:
    padded_instance_list (list of tensors): Padded instances.
    max_length (int): The maximum length among instances after padding.
    """

    # Determine the maximum length among all instances
    max_length = max(tensor.size(0) for tensor in instance_list)

    # Pad instances to the same length
    padded_instance_list = [torch.cat((tensor, torch.zeros(max_length - tensor.size(0), *tensor.shape[1:]).fill_(padding_value)), dim=0) for tensor in instance_list]

    return padded_instance_list, max_length

def sigmoid_cost(logit: torch.Tensor,
                 *,
                 focal_loss: bool = False,
                 focal_alpha: Optional[float] = None,
                 focal_gamma: Optional[float] = None
) -> torch.Tensor:
    """
    Computes the classification cost.

    Args:
    logit: Sigmoid classification logit(s).
    focal_loss: Whether to apply focal loss for classification cost.
    focal_alpha: Alpha scaling factor for focal loss.
    focal_gamma: Gamma scaling factor for focal loss.

    Returns:
    Classification cost.
    """
    neg_cost_class = -torch.nn.functional.logsigmoid(-logit)
    pos_cost_class = -torch.nn.functional.logsigmoid(logit)
    
    if focal_loss:
        neg_cost_class *= (1 - focal_alpha) * torch.sigmoid(logit) ** focal_gamma
        pos_cost_class *= focal_alpha * torch.sigmoid(-logit) ** focal_gamma
    
    return pos_cost_class - neg_cost_class


def compute_cost(*,
                tgt_labels: List[torch.Tensor],
                out_logits: torch.Tensor,
                tgt_bbox: List[torch.Tensor],
                out_bbox: torch.Tensor,
                num_classes: int,
                class_loss_coef: float,
                bbox_loss_coef: torch.Tensor,
                giou_loss_coef: torch.Tensor,
                focal_loss: bool = False,
                focal_alpha: Optional[float] = None,
                focal_gamma: Optional[float] = None):
    """
    Args:
    tgt_labels: List of class labels tensors of shape [B, M, C] (one/multi-hot) for each instance.
      Note that the labels corresponding to empty bounding boxes are not yet supposed to be filtered out.
    out_logits: Classification sigmoid logits of shape [B, N, C].
    tgt_bbox: List of target box tensors of shape [B, M, 4] for each instance.
      Note that the empty bounding boxes are not yet supposed to be filtered out.
    out_bbox: Predicted box coordinates of shape [B, N, 4].
    class_loss_coef: Relative weight of classification loss.
    bbox_loss_coef: Relative weight of bbox loss.
    giou_loss_coef: Relative weight of giou loss.
    focal_loss: Whether to apply focal loss for classification cost.
    focal_alpha: Alpha scaling factor for focal loss.
    focal_gamma: Gamma scaling factor for focal loss.

    Returns:
    A cost matrix [B, N, M].
    Number of unpadded columns per batch element [B].
    """
    if focal_loss and (focal_alpha is None or focal_gamma is None):
        raise ValueError("For focal loss, focal_alpha and focal_gamma must be set.")
    
    
    # Getting desired dimensions from out_logits
    batch_size, num_patches, num_queries = out_logits.size()
    # Pad instances and convert to tensors
    padded_tgt_labels, max_length_labels = pad_instances(tgt_labels)
    padded_tgt_labels = torch.stack(padded_tgt_labels)
    
    # Convert tgt_labels to desired shape [B, M, C]
    desired_labels = torch.zeros(padded_tgt_labels.size(0), num_queries , num_classes)
    for b in range(padded_tgt_labels.size(0)):
        for m in range(max_length_labels):
            label = int(padded_tgt_labels[b, m])  # Convert tensor to integer
            if label >= 0:   # Skip instances with label < 0 (blank list)
                desired_labels[b, m, label] = 1

    def pad_tensors_with_variable_lengths(tensors, padding_value=0):
        max_num_boxes = max(tensor.size(0) for tensor in tensors)
        padded_tensors = []

        for tensor in tensors:
            padded_tensor = torch.full((max_num_boxes,) + tensor.shape[1:], padding_value)
            padded_tensor[:tensor.size(0)] = tensor
            padded_tensors.append(padded_tensor)

        padded_tensors = torch.stack(padded_tensors)
        return padded_tensors


    padded_tgt_bbox = pad_tensors_with_variable_lengths(tgt_bbox)
    # Convert tgt_bbox format from [x_min, y_min, w, h] (COCO dataset) to [center_x, center_y, w, h]
    padded_tgt_bbox = box_xywh_to_cxcywh(padded_tgt_bbox)
    
    # # Convert tgt_bbox to padded tensor
    # padded_tgt_bbox = torch.nn.utils.rnn.pad_sequence([boxes.clone().detach().requires_grad_(True) for boxes in tgt_bbox], batch_first=True)
    
    """
    Removed mask-related computation in original codes
    https://github.com/google-research/scenic/blob/main/scenic/projects/owl_vit/losses.py
    """ 
    ###################################
    #Reshape output's dimension into desired dimensions of written cost function 
    batch_size, num_patches, num_queries = out_logits.size()
    
    # Permute the out_logits dimensions to get [batch_size, num_queries, num_patches]
    permuted_linear_logits = out_logits.permute(0, 2, 1)
    # Create a linear layer to transform to [batch_size, num_queries, num_classes]
    linear_layer = torch.nn.Linear(num_patches, num_classes).to(out_logits.device)
    output_logits = linear_layer(permuted_linear_logits)
    
    
    # Reshape bounding box tensor's dimension
    reshaped_pred_box = out_bbox.view(-1, 4)
    # Repeat reshaped_pred_boxes to match the number of queries
    tiled_pred_boxes = reshaped_pred_box.repeat(num_queries, 1)
    # Reshape back to [batch_size, num_queries, num_patches, 4]
    reshaped_pred_boxes = tiled_pred_boxes.view(batch_size, num_queries, num_patches, 4)
    # Select the appropriate indices to get [batch_size, num_queries, 4]
    output_bbox = reshaped_pred_boxes[:, :, 0, :]
    
    ####################################
    
    # Compute classification loss using sigmoid_loss function.
    loss_class = sigmoid_cost(output_logits,
                             focal_loss=focal_loss,
                             focal_alpha=focal_alpha,
                             focal_gamma=focal_gamma)   # [B, N, C]
    
    desired_labels = desired_labels.to(loss_class.device)
    loss_class = torch.einsum('bnl,bml->bnm', loss_class, desired_labels)
    
    
    # Compute absolute differences between predicted bbox and target bbox.
    padded_tgt_bbox = padded_tgt_bbox.to(output_bbox.device)
    diff = torch.abs(output_bbox[:, :, None] - padded_tgt_bbox[:, None, :])  # [B, N, M, 4]
    
    # Compute bbox loss by summing differences along the last dimension (coordinates).
    loss_bbox = diff.sum(dim=-1) 
    
    
    # Compute generalized IoU (GIoU) loss using specialized function
    loss_giou = -generalized_box_iou(box_cxcywh_to_xyxy(output_bbox),
                                     box_cxcywh_to_xyxy(padded_tgt_bbox),
                                     all_pairs=True)
    
    
    # Combine all the losses
    total_loss = loss_class * class_loss_coef + loss_bbox * bbox_loss_coef + loss_giou * giou_loss_coef
    #print(f"Class loss {loss_class.sum()}, class loss coef {class_loss_coef}, Loss bbox {loss_bbox.sum()}, bbox_loss coef {bbox_loss_coef} Loss GIoU {loss_giou.sum()}, giou_loss_coef {giou_loss_coef}")
    # Return the computed loss tensor and n_cols
    return total_loss