from typing import List, Tuple, Union

import torch
from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.nets.ProSeg import PRODNet


class ProSegTrainer(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = False
        self.set_deep_supervision_enabled = lambda enabled: None

    def build_network_architecture(
        self,
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        model = PRODNet(
            dim_in=num_input_channels,
            num_classes=num_output_channels,
            depths=[2, 2, 8, 3],
            embed_dims=[24, 48, 96, 192],
            drop=0.1,
        )
        self.configuration_manager.configuration["patch_size"] = (32, 128, 128)
        return model
