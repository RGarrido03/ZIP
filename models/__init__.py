import os
import torch
from typing import List, Tuple, Optional

from .ebc import _ebc, EBC


def get_model(
    model_info_path: str,
    block_size: Optional[int] = None,
    bins: Optional[List[Tuple[float, float]]] = None,
    bin_centers: Optional[List[float]] = None,
    zero_inflated: Optional[bool] = True,
    vpt_drop: Optional[float] = None,
    input_size: Optional[int] = None,
    norm: str = "none",
    act: str = "none",
) -> EBC:
    if os.path.exists(model_info_path):
        model_info = torch.load(model_info_path, map_location="cpu", weights_only=False)        

        block_size = model_info["config"]["block_size"]
        bins = model_info["config"]["bins"]
        bin_centers = model_info["config"]["bin_centers"]
        zero_inflated = model_info["config"]["zero_inflated"]
        vpt_drop = model_info["config"].get("vpt_drop", None)
        input_size = model_info["config"].get("input_size", None)
        norm = model_info["config"].get("norm", "none")
        act = model_info["config"].get("act", "none")
        weights = model_info["weights"]

    else:
        assert block_size is not None, "block_size should be provided"
        assert bins is not None, "bins should be provided"
        assert bin_centers is not None, "bin_centers should be provided"
        weights = None

    model = _ebc(
        block_size=block_size,
        bins=bins,
        bin_centers=bin_centers,
        zero_inflated=zero_inflated,
        vpt_drop=vpt_drop,
        input_size=input_size,
        norm=norm,
        act=act
    )
    model_config = {
        "block_size": block_size,
        "bins": bins,
        "bin_centers": bin_centers,
        "zero_inflated": zero_inflated,
        "vpt_drop": vpt_drop,
        "input_size": input_size,
        "norm": norm,
        "act": act
    }

    model.config = model_config
    model_info = {"config": model_config, "weights": weights}

    if weights is not None:
        model.load_state_dict(weights)

    if not os.path.exists(model_info_path):
        torch.save(model_info, model_info_path)
    
    return model


__all__ = ["get_model"]
