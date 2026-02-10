#  Copyright (c) 2024, Salesforce, Inc.
#  SPDX-License-Identifier: Apache-2
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# Modifications Copyright (c) 2026, Abhilash Neog, Virginia Tech.
# Licensed under the Apache License, Version 2.0.

from typing import Optional

import torch
from einops import reduce
from jaxtyping import Bool, Float, Int
from torch import nn

from utils.torch_util import safe_div

    
class PackedScaler(nn.Module):
    def forward(
        self,
        target: torch.Tensor,          # [B, S_]
        mask: Optional[torch.Tensor] = None,  # [B, S_]
        sample_id: Optional[torch.Tensor] = None,      # [B, S_]
        variate_id: Optional[torch.Tensor] = None,     # [B, S_]
        depth_val: Optional[torch.Tensor] = None,      # [B, S_]
    ):
        if mask is None:
            mask = torch.ones_like(target, dtype=torch.bool)
        if sample_id is None:
            sample_id = torch.zeros_like(target, dtype=torch.long, device=target.device)
        if variate_id is None:
            variate_id = torch.zeros_like(target, dtype=torch.long, device=target.device)
        if depth_val is None:
            depth_val = torch.zeros_like(target, dtype=torch.float, device=target.device)

        loc, scale = self._get_loc_scale(
            target.double(), mask, sample_id, variate_id, depth_val
        )
        return loc.float(), scale.float()

    def _get_loc_scale(
        self,
        target: torch.Tensor,
        mask: torch.Tensor,
        sample_id: torch.Tensor,
        variate_id: torch.Tensor,
        depth_val: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

class PackedNOPScaler(PackedScaler):
    def _get_loc_scale(
        self,
        target, mask, sample_id, variate_id, depth_val,
    ):
        return torch.zeros_like(target), torch.ones_like(target)


class PackedStdScaler(PackedScaler):
    def __init__(self, correction=1, minimum_scale=1e-5):
        super().__init__()
        self.correction = correction
        self.minimum_scale = minimum_scale

    def _get_loc_scale(self, target, mask, sample_id, variate_id, depth_val):
        B, S_ = target.shape

        # [B, S_, 1] for broadcasting
        sample_id = sample_id.unsqueeze(-1)
        variate_id = variate_id.unsqueeze(-1)
        depth_val = depth_val.unsqueeze(-1)
        
        id_mask = (
            (sample_id == sample_id.transpose(1, 2)) &
            (variate_id == variate_id.transpose(1, 2)) &
            (depth_val == depth_val.transpose(1, 2))
        )
        mask = mask.squeeze().unsqueeze(1)  # [B, 1, S_]
        target_exp = target.unsqueeze(1)  # [B, 1, S_]
        
        tobs = reduce(id_mask * mask, "b s1 s2 -> b s1 1", "sum")
        
        loc = reduce(id_mask * target_exp * mask, "b s1 s2 -> b s1 1", "sum")
        
        loc = safe_div(loc, tobs)
        
        var = reduce(
            id_mask * ((target_exp - loc) ** 2) * mask,
            "b s1 s2 -> b s1 1", "sum"
        )
        
        var = safe_div(var, (tobs - self.correction))
        scale = torch.sqrt(var + self.minimum_scale)

        group0_mask = sample_id.squeeze(-1) == 0
        loc[group0_mask] = 0
        scale[group0_mask] = 1

        return loc.squeeze(-1), scale.squeeze(-1)
    
class PackedAbsMeanScaler(PackedScaler):
    def _get_loc_scale(
        self,
        target: Float[torch.Tensor, "B S_ 1"],
        observed_mask: Bool[torch.Tensor, "B S_ 1"],
        sample_id: Int[torch.Tensor, "B S_ 1"],
        variate_id: Int[torch.Tensor, "B S_ 1"],
        depth_val: Float[torch.Tensor, "B S_ 1"],  # continuous value
    ) -> tuple[
        Float[torch.Tensor, "B S_ 1"], Float[torch.Tensor, "B S_ 1"]
    ]:
        # (B, S_, S_) boolean mask identifying (sample, variate, depth) groups
        id_mask = (
            (sample_id == sample_id.transpose(1, 2)) &
            (variate_id == variate_id.transpose(1, 2)) &
            (depth_val == depth_val.transpose(1, 2))
        )

        # Expand for broadcasting
        target_exp = target.transpose(1, 2)  # (B, 1, S_)
        mask = observed_mask.transpose(1, 2)  # (B, 1, S_)

        # Number of valid entries per group
        tobs = reduce(id_mask * mask, "b s1 s2 -> b s1 1", "sum")

        # Absolute mean (no centering)
        scale = reduce(id_mask * target_exp.abs() * mask, "b s1 s2 -> b s1 1", "sum")
        scale = safe_div(scale, tobs)

        # Location is always 0 (mean not subtracted)
        loc = torch.zeros_like(scale)

        # For padding/masked sample_id = 0
        loc[sample_id.squeeze(-1) == 0] = 0
        scale[sample_id.squeeze(-1) == 0] = 1

        return loc, scale