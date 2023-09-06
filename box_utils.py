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


def box_iou(boxes1, boxes2, all_pairs=True, eps=1e-6):
  
    """Computes IoU between two sets of boxes.

  Boxes are in [x, y, x', y'] format [x, y] is top-left, [x', y'] is bottom
  right.

  Args:
    boxes1: Predicted bounding-boxes in shape [bs, n, 4].
    boxes2: Target bounding-boxes in shape [bs, m, 4]. Can have a different
      number of boxes if all_pairs is True.
    np_backbone: numpy module: Either the regular numpy package or jax.numpy.
    all_pairs: Whether to compute IoU between all pairs of boxes or not.
    eps: Epsilon for numerical stability.

  Returns:
    If all_pairs == True, returns the pairwise IoU cost matrix of shape
    [bs, n, m]. If all_pairs == False, returns the IoU between corresponding
    boxes. The shape of the return value is then [bs, n].
  """

    # First, compute box areas. These will be used later for computing the union.
    wh1 = boxes1[..., 2:] - boxes1[..., :2]
    area1 = wh1[..., 0] * wh1[..., 1]  # [bs, n]

    wh2 = boxes2[..., 2:] - boxes2[..., :2]
    area2 = wh2[..., 0] * wh2[..., 1]  # [bs, m]

    if all_pairs:
        # Compute pairwise top-left and bottom-right corners of the intersection
        # of the boxes.
        lt = torch.max(boxes1[..., :, None, :2], boxes2[..., None, :, :2])  # [bs, n, m, 2].
        rb = torch.min(boxes1[..., :, None, 2:], boxes2[..., None, :, 2:])  # [bs, n, m, 2].

        # intersection = area of the box defined by [lt, rb]
        wh = (rb - lt).clamp(min=0.0)  # [bs, n, m, 2]
        intersection = wh[..., 0] * wh[..., 1]  # [bs, n, m]

        # union = sum of areas - intersection
        union = area1[..., :, None] + area2[..., None, :] - intersection

        iou = intersection / (union + eps)

    else:
        # Compute top-left and bottom-right corners of the intersection between
        # corresponding boxes.
        assert boxes1.shape[1] == boxes2.shape[1], (
            'Different number of boxes when all_pairs is False')
        lt = torch.max(boxes1[..., :, :2], boxes2[..., :, :2])  # [bs, n, 2]
        rb = torch.min(boxes1[..., :, 2:], boxes2[..., :, 2:])  # [bs, n, 2]

        # intersection = area of the box defined by [lt, rb]
        wh = (rb - lt).clamp(min=0.0)  # [bs, n, 2]
        intersection = wh[..., :, 0] * wh[..., :, 1]  # [bs, n]

        # union = sum of areas - intersection.
        union = area1 + area2 - intersection

        iou = intersection / (union + eps)

    return iou, union


# Copied from transformers.models.detr.modeling_detr.generalized_box_iou
def generalized_box_iou(boxes1: torch.Tensor,
                        boxes2: torch.Tensor,
                        all_pairs: bool = True,
                        eps: float = 1e-6) -> torch.Tensor:
    """Generalized IoU from https://giou.stanford.edu/.
    
    The boxes should be in [x, y, x', y'] format specifying top-left and
    bottom-right corners.
    
    Args:
        boxes1: Predicted bounding-boxes in shape [..., n, 4].
        boxes2: Target bounding-boxes in shape [..., m, 4].
        all_pairs: Whether to compute generalized IoU between all pairs of
            boxes or not. Note that if all_pairs == False, we must have m==n.
        eps: Epsilon for numerical stability.
    
    Returns:
        If all_pairs == True, returns a [bs, n, m] pairwise matrix of generalized
        IoUs. If all_pairs == False, returns a [bs, n] matrix of generalized IoUs.
    """
    # Degenerate boxes give inf/nan results, so do an early check.
    # torch.all((boxes1[..., 2:] >= boxes1[..., :2]))
    # torch.all((boxes2[..., 2:] >= boxes2[..., :2]))
    iou, union = box_iou(
        boxes1, boxes2, all_pairs=all_pairs, eps=eps)
    
    # Generalized IoU has an extra term that takes into account the area of
    # the box containing both of these boxes. The following code is very similar
    # to that for computing the intersection, but the min and max are flipped.
    if all_pairs:
        lt = torch.minimum(boxes1[..., :, None, :2],
                           boxes2[..., None, :, :2])  # [bs, n, m, 2]
        rb = torch.maximum(boxes1[..., :, None, 2:],
                           boxes2[..., None, :, 2:])  # [bs, n, m, 2]
    else:
        lt = torch.minimum(boxes1[..., :, :2],
                           boxes2[..., :, :2])  # [bs, n, 2]
        rb = torch.maximum(boxes1[..., :, 2:], boxes2[..., :,
                                                      2:])  # [bs, n, 2]
    
    # Now, compute the covering box's area.
    wh = (rb - lt).clamp(0.0)  # Either [bs, n, 2] or [bs, n, m, 2].
    area = wh[..., 0] * wh[..., 1]  # Either [bs, n] or [bs, n, m].
    
    # Finally, compute generalized IoU from IoU, union, and area.
    # Somehow the PyTorch implementation does not use eps to avoid 1/0 cases.
    return iou - (area - union) / (area + eps)

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




