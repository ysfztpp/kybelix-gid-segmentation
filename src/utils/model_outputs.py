from typing import Any, Optional, Tuple

import torch


def split_model_outputs(outputs: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Supports:
    - tensor logits
    - {"seg": seg_logits, "edge": edge_logits}
    - tuple/list: (seg_logits, edge_logits)
    """
    if isinstance(outputs, dict):
        if "seg" not in outputs:
            raise KeyError("Model output dict must contain 'seg' logits.")
        return outputs["seg"], outputs.get("edge")

    if isinstance(outputs, (tuple, list)):
        if len(outputs) == 0:
            raise ValueError("Model output tuple/list is empty.")
        edge_logits = outputs[1] if len(outputs) > 1 else None
        return outputs[0], edge_logits

    if not torch.is_tensor(outputs):
        raise TypeError(f"Unsupported model output type: {type(outputs)!r}")

    return outputs, None


def get_segmentation_logits(outputs: Any) -> torch.Tensor:
    seg_logits, _ = split_model_outputs(outputs)
    return seg_logits
