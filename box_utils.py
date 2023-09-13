import torch
from typing import Dict, List, Optional, Tuple, Union
from transformers.utils import TensorType

# Copied from transformers.models.detr.modeling_detr._upcast
def _upcast(t: torch.Tensor) -> torch.Tensor:
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()
    

# Copied from transformers.models.detr.modeling_detr.box_area
def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by its (x1, y1, x2, y2) coordinates.

    Args:
        boxes (`torch.FloatTensor` of shape `(number_of_boxes, 4)`):
            Boxes for which the area will be computed. They are expected to be in (x1, y1, x2, y2) format with `0 <= x1
            < x2` and `0 <= y1 < y2`.

    Returns:
        `torch.FloatTensor`: a tensor containing the area for each box.
    """
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_cxcywh_to_xyxy(x):
    """ Convert boxes from [cx, cy, w, h] format into [x, y, x', y'] format"""
    x_c, y_c, w, h = torch.split(x, 1, dim=-1)  # Split into individual components
    x1 = x_c - 0.5 * w
    y1 = y_c - 0.5 * h
    x2 = x_c + 0.5 * w
    y2 = y_c + 0.5 * h
    return torch.cat([x1, y1, x2, y2], dim=-1)

def box_xywh_to_cxcywh(x):
    """"Convert boxes from [x_min, y_min, w, h] format into [center_x, center_y, w, h] format"""
    x_min, y_min, w, h = torch.split(x, 1, dim = -1)
    center_x = x_min + (w/2)
    center_y = y_min + (w/2)
    return torch.cat([center_x, center_y, w, h], dim = -1)

def box_cxcywh_to_xywh(x):
    """Convert boxes from [center_x, center_y, w, h] format into [x_min, y_min, w, h]"""
    center_x, center_y, w, h = torch.split(x, 1, dim = -1)
    x_min = center_x - (w/2)
    y_min = center_y - (h/2)
    return torch.cat([x_min, y_min, w, h], dim = -1)

def box_xyxy_to_xywh(x):
    """Convert boxes from [top left x, top left y, bottom right x, bottom right y] format into [top left x, top left y, width, height]"""
    tf_x, tf_y, br_x, br_y = torch.split(x, 1, dim = -1)
    width = br_x - tf_x
    height = br_y - tf_y
    return torch.cat([tf_x, tf_y, width, height], dim = -1)

def make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

