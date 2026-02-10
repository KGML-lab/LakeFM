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

import math

import torch
from jaxtyping import Float, Int
from torch import nn
import numpy as np

def get_time_embeddings(dates):
    '''
    return month embeddings
    '''
    # dates : np.ndarray or torch.Tensor of dtype datetime64[D]  shape (B, S)
    dates = np.asarray(dates, dtype='datetime64[D]')   # → ndarray (S,)
    month_idx = (dates.astype('datetime64[M]').astype(np.int64) % 12) # 0 = Jan, …, 11 = Dec
    month_idx = torch.as_tensor(month_idx, dtype=torch.float32)

    month_angle = 2 * math.pi * month_idx / 12.0         # radians
    sin_month = torch.sin(month_angle)                 # (B, S)
    cos_month = torch.cos(month_angle)                 # (B, S)

    time_feat = torch.stack([sin_month, cos_month], dim=-1)   # (B,S,2)
    return time_feat

class SinusoidalPositionEncoding(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        max_len: int,
        normalize: bool = True,
    ):
        """
        Construct a sinusoidal positional embedding module.

        :param width:
            Width of the embedding.
        :param max_len:
            Maximum length of the embedding.
        :param normalize:
            Perform L2 normalization of the embedding.
        """
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, width, 2) * (-math.log(10000.0) / width))

        pe = torch.zeros(max_len, width)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        if normalize:
            l2 = torch.linalg.vector_norm(pe, dim=-1)
            pe /= l2.unsqueeze(-1)

        self.register_buffer("pe", pe, persistent=False)

    def forward(
        self, pos_id: Int[torch.Tensor, "*batch length"]
    ) -> Float[torch.Tensor, "*batch length dim"]:
        return self.pe[pos_id]



class FourierFeatureEncoding(nn.Module):
    def __init__(self, 
                 num_bands: int = 6, 
                 max_resolution: float = 1.0, 
                 include_input: bool = True):
        """
        :param num_bands: Number of frequency bands (controls output dimensionality)
        :param max_resolution: Max value of input used to scale frequencies (e.g., max time or depth)
        :param include_input: Whether to include the raw input itself in the output
        """
        super().__init__()
        self.num_bands = num_bands
        self.include_input = include_input
        self.freq_bands = 2.0 ** torch.linspace(0.0, num_bands - 1, num_bands) * math.pi / max_resolution
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Tensor of shape (..., 1) – typically a scalar per token (e.g. depth, time)
        :return: Fourier features of shape (..., num_bands * 2 [+1 if include_input])
        """
        x = x.unsqueeze(-1)# if x.ndim == 1 else x
        
        invalid_mask = (x == -1)
        
        x_proj = x * self.freq_bands.to(x.device)  # shape (..., num_bands)
        
        sin_enc = torch.sin(x_proj)
        cos_enc = torch.cos(x_proj)
        encoding = [sin_enc, cos_enc]
        if self.include_input:
            encoding = [x] + encoding
        
        encoding = torch.cat(encoding, dim=-1)
        
        return encoding

class LearnedEmbedding(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        max_len: int,
    ):
        super().__init__()
        self.pe = nn.Embedding(
            max_len,
            width,
        )

    def forward(
        self, pos_id: Int[torch.Tensor, "*batch length"]
    ) -> Float[torch.Tensor, "*batch length dim"]:
        return self.pe(pos_id)