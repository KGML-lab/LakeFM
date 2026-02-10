import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import math
from torch.distributions import StudentT
import torch.distributed as dist
from functools import partial
from lakefm.module.packed_scaler import PackedStdScaler, PackedNOPScaler
from lakefm.module.position import (BinaryAttentionBias, 
                                    QueryKeyProjection, 
                                    RotaryProjection, 
                                    FourierFeatureEncoding)
from lakefm.module.norm import RMSNorm
from lakefm.module.ts_embed import MultiInSizeLinear
from lakefm.module.transformer import TransformerEncoder, TransformerDecoder
from utils.torch_util import mask_fill, packed_attention_mask
from omegaconf import DictConfig
from lakefm.module.tokenizer import Tokenizer
from utils.exp_utils import pretty_print
from lakefm.module.position.additive import get_time_embeddings

PAD_VAL=0


def set_token_size(tokenization: str, patch_size: int, add_ctxt_type, use_time_feat):
    """
    Set the token size based on the tokenization method and patch size.
    Args:
        tokenization: The type of tokenization method used.
        patch_size: The size of the patches for patch-based tokenization.
    """
    to_add = 0
    if use_time_feat:
        if add_ctxt_type == "add":
            to_add=2

    if tokenization == "scalar":
        return 1 + to_add
    elif tokenization == "patch":
        return patch_size + to_add
    elif tokenization == "temporal":
        return 1 + to_add
    else:
        raise ValueError(f"Unsupported tokenization method: {tokenization}")

class MeanPoolingMLP(nn.Module):
    def __init__(self, d_model, proj_dim=128):
        super().__init__()
        # self.mlp = nn.Linear(d_model, d_model)
        self.projection_head = nn.Sequential(
            nn.Linear(d_model, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, d_model),
        )
    def forward(self, x, mask):
        masked_output = x * mask.unsqueeze(-1)  # Shape: (batch, S, d_model)
        z = torch.sum(masked_output, dim=1) / torch.sum(mask, dim=1, keepdim=True)  # Shape: (batch, d_model)
        z = self.projection_head(z)  # Shape: (batch, proj_dim)
        return z #self.mlp(weights)  # Shape: (d_model,)

class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.Linear(d_model, 1)
    
    def forward(self, x, mask):
        # x: (batch, S, d_model), mask: (batch, S)
        # Apply mask to attention scores
        attn_scores = self.attn(x)  # Shape: (batch, S, 1)
        attn_scores = attn_scores.masked_fill(mask.unsqueeze(-1) == 0, -1e4)  # Set padded scores to large negative
        weights = torch.softmax(attn_scores, dim=1)  # Shape: (batch, S, 1)
        return torch.sum(x * weights, dim=1)  # Shape: (batch, d_model)

class ForecastHead(nn.Module):
    def __init__(self, d_model: int, 
                 dropout_head: float = 0.0, 
                 additional_forecast_layer: bool = False,
                 seq_len: int = None,
                 pred_len: int = None):
        super().__init__()
        self.additional_forecast_layer = additional_forecast_layer
        
        self.half_ = nn.Linear(d_model, d_model//2)
        if self.additional_forecast_layer:
            self.additional_layer = nn.Linear(d_model//2, d_model//4)
            self.to_scalar = nn.Linear(d_model//4, 1) # project each token from d_model//4 to 1
        else:
            self.to_scalar = nn.Linear(d_model//2, 1) # project each token from d_model//2 to 1
        
        # self.to_forecast = None # project from seq_len to pred_len
        # Initialize to_forecast if seq_len is provided
        if seq_len is not None:
            self.to_forecast = nn.Linear(seq_len, pred_len)
        else:
            self.to_forecast = None
        self.dropout = nn.Dropout(dropout_head)

    def forward(self, x: torch.Tensor, 
                pred_len: int) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len, d_model)
        Returns:
            (B, pred_len, 1)
        """
        _, seq_len, _ = x.shape
        x = self.half_(x) # (B, seq_len, d_model//2)
        x = self.dropout(x)
        if self.additional_forecast_layer:
            x = self.additional_layer(x)
            x = self.dropout(x)
        x = self.to_scalar(x).squeeze()  # (B, seq_len)
        
        if self.to_forecast is None:
            self.to_forecast = nn.Linear(seq_len, pred_len).to(x.device) # lazy init if not initalized
        
        x = self.to_forecast(x) # (B, pred_len)        

        return x

    def forecast(self, x: torch.Tensor,
                 seq_len: int,
                 pred_len: int) -> torch.Tensor:
        """
        Args:
            x: (B, flat_seq, d_model)
        Returns:
            (B, flat_pred_len, 1)
        """
        B, flat_seq, d_model = x.shape
        assert flat_seq % seq_len == 0, f"Expected flat_seq divisible by seq_len, got {flat_seq}"
        num_chunks = flat_seq // seq_len
        
        x = x.view(B * num_chunks, seq_len, d_model)  # (B * chunks, seq_len, d_model)

        out = self(x, pred_len) # (B * chunks, pred_len)

        out = out.view(B, -1) # (B, pred_len * chunks)
        
        return out # (B, pred_len * chunks)

class LakeFMModule(nn.Module):
    
    def __init__(self,
                d_model: int,
                num_layers: int,
                n_heads: int,
                patch_size: int,
                max_seq_len: int,
                attn_dropout_p: float,
                dropout_p: float,
                scaling: bool = True,
                **kwargs):

        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.scaling = scaling
        self.cfg = kwargs
        self.num_bands = self.cfg['num_bands']
        self.max_resolution = self.cfg['max_resolution']
        self.include_input = self.cfg['include_input']
        self.dropout_head = self.cfg['head_dropout']
        self.add_ctxt_type = self.cfg["add_or_concat"]
        self.use_rope = self.cfg['use_rope']
        self.tokenization = self.cfg['tokenization']
        self.use_time_embed = self.cfg['use_time_embed']
        self.cl_proj_type = self.cfg['cl_proj_type']
        self.cl_proj_dim = self.cfg['cl_proj_dim']
        self.time_embed_dim = self.cfg['time_embed_dim']
        self.additional_forecast_layer = self.cfg['additional_forecast_layer']
        self.use_pre_norm = self.cfg['use_pre_norm']
        self.revin = self.cfg['revin']
        self.variate_wise_df = self.cfg.get('variate_wise_df', False)  # Default to False for backward compatibility
        self.shared_variate_embedding_for_df = self.cfg.get('shared_variate_embedding_for_df', False)
        self.stop_grad_on_shared_var_embed_for_df = True

        self.set_token_size = set_token_size(self.tokenization, self.patch_size, self.add_ctxt_type, self.use_time_embed)

        self.mask_encoding = nn.Embedding(num_embeddings=1, embedding_dim=d_model)

        self.tokenizer, self.patch_size = Tokenizer.build(patch_size=self.patch_size, 
                                                          tokenization_type=self.tokenization)

        d_static = self.cfg['static_dim']
        d_temporal = self.cfg['temporal_dim']

        if self.cl_proj_type=='attention_pooling':
            self.cl_proj = AttentionPooling(d_static)  # Use static dimension
        else:
            self.cl_proj = MeanPoolingMLP(d_model=d_static, proj_dim=self.cl_proj_dim)  # Use static dimension

        if self.add_ctxt_type == "concat":
            self.var_embed_dim = self.cfg["var_embed_dim"]
            self.depth_embed_dim = self.cfg["depth_embed_dim"]
            self.inp_embed_dim = self.cfg["inp_embed_dim"]
        else:
            if self.use_time_embed:
                self.inp_embed_dim = self.d_model - 2 # time embed dim is fixed to 2
            else:
                self.inp_embed_dim = self.d_model
            self.var_embed_dim = self.d_model
            self.depth_embed_dim = self.d_model

        if self.use_time_embed:
            self.concat_proj_dim = self.var_embed_dim + self.depth_embed_dim + self.inp_embed_dim + self.time_embed_dim
        else:
            self.concat_proj_dim = self.var_embed_dim + self.depth_embed_dim + self.inp_embed_dim
        self.concat_proj = nn.Linear(self.concat_proj_dim, self.d_model)

        if self.use_rope:
            time_qk_proj_layer = partial(
                QueryKeyProjection,
                proj_layer=RotaryProjection,
                kwargs=dict(max_len=max_seq_len),
                partial_factor=(0.0, 0.5),
            )
        else:
            time_qk_proj_layer = None

        self.encoder = TransformerEncoder(
            d_model,
            num_layers,
            num_heads=n_heads,
            pre_norm=self.use_pre_norm,
            attn_dropout_p=attn_dropout_p,
            dropout_p=dropout_p,
            norm_layer=RMSNorm,
            activation=F.silu,
            use_glu=True,
            use_qk_norm=True,
            var_attn_bias_layer=partial(BinaryAttentionBias),
            time_qk_proj_layer=time_qk_proj_layer,
            shared_var_attn_bias=False,
            shared_time_qk_proj=True,
            d_ff=None,
        )
        
        
        # Decoder works in temporal subspace (d_temporal dimensions)
        self.decoder = TransformerDecoder(
            d_temporal,  # Use temporal dimension, not full d_model
            num_layers,
            num_heads=n_heads,
            pre_norm=self.use_pre_norm,
            attn_dropout_p=attn_dropout_p,
            dropout_p=dropout_p,
            norm_layer=RMSNorm,
            activation=F.silu,
            use_glu=True,
            use_qk_norm=True,
            var_attn_bias_layer=partial(BinaryAttentionBias),
            time_qk_proj_layer=time_qk_proj_layer,
            shared_var_attn_bias=False,
            shared_time_qk_proj=True,
            d_ff=None,
        )
        self.value_proj_a = nn.Linear(1, self.inp_embed_dim)
        self.value_proj_b = nn.Linear(1, self.inp_embed_dim)

        self.in_projector = nn.Linear(self.set_token_size, self.inp_embed_dim)
        self.out_projector = nn.Linear(d_model, self.patch_size)
        
        self.loc_head = nn.Linear(d_temporal, 1)  # Location parameter (mean)
        self.scale_head = nn.Linear(d_temporal, 1)  # Scale parameter (std)
        self.dof_head = nn.Linear(d_temporal, 1)  # Degrees of freedom (learnable)
        
        # init scale head to output values that, after softplus, give reasonable scales
        # We want softplus(scale_head_output) + 1e-3 to be around 0.1-1.0
        # So scale_head_output should be around log(0.1) ≈ -2.3 to log(1.0) = 0
        with torch.no_grad():
            # Scale head: bias around -1.0, small weights
            self.scale_head.bias.fill_(-1.0)
            self.scale_head.weight.normal_(0, 0.1)
            
            # DOF head: bias around 1.0 (after softplus + 2.1 = 3.1), small weights
            self.dof_head.bias.fill_(1.0)
            self.dof_head.weight.normal_(0, 0.1)
            
            # Loc head: bias around 0.0, small weights
            self.loc_head.bias.fill_(0.0)
            self.loc_head.weight.normal_(0, 0.1)

        self.depth_pos_encoder = FourierFeatureEncoding(num_bands=self.num_bands,
                                                        max_resolution=self.max_resolution,
                                                        include_input=self.include_input)
        if self.include_input:
            depth_proj_in_dim = 1 + self.num_bands * 2
        else:
            depth_proj_in_dim = self.num_bands * 2
        self.depth_proj = nn.Linear(depth_proj_in_dim, self.depth_embed_dim)

        self.var_id_embed = nn.Embedding(num_embeddings=self.cfg['max_vars'], 
                                         embedding_dim=self.var_embed_dim,
                                         padding_idx=PAD_VAL)

        # Variate-wise degrees of freedom (optional hybrid approach)
        if self.variate_wise_df:
            if self.shared_variate_embedding_for_df:
                # Use existing var_id_embed projection to a scalar df base
                # Detach optionally to prevent df gradients from updating shared embedding
                self.df_base_projector = nn.Linear(self.var_embed_dim, 1, bias=True)
                with torch.no_grad():
                    # Initialize projector bias so that softplus(bias) + 2.1 ~ 3.1-3.5
                    self.df_base_projector.bias.fill_(1.0)
                print("Variate-wise DF enabled (shared embedding): base = Linear(var_id_embed) [+ detach]")
            else:
                # Each variable gets a learnable base df parameter (separate embedding)
                self.variate_df_embedding = nn.Embedding(
                    num_embeddings=self.cfg['max_vars'], 
                    embedding_dim=1,
                    padding_idx=None  # No padding for df values
                )
                # Initialize: bias around 1.0 (after softplus + 2.1 = 3.1 df)
                with torch.no_grad():
                    nn.init.constant_(self.variate_df_embedding.weight, 1.0)
                print("Variate-wise DF enabled (separate embedding): Hybrid base + adjustment")
        else:
            print("Variate-wise DF disabled: Standard approach (context-only)")

        self.static_proj = nn.Linear(self.d_model, d_static, bias=False)
        self.temporal_proj = nn.Linear(self.d_model, d_temporal, bias=False)
        
        # Project metadata-only query embeddings to temporal dimension for cross-attention
        metadata_dim = self.var_embed_dim + self.depth_embed_dim + (self.time_embed_dim if self.use_time_embed else 0)
        self.query_proj = nn.Linear(metadata_dim, d_temporal, bias=False)

    def _sinusoidal_time_embed(self, time_values: torch.Tensor, embed_dim: int) -> torch.Tensor:
        """
        Build sinusoidal embeddings from per-token numeric time values.
        Expects time_values already normalized to [0, 1] representing day of year.
        """
        # time_values already normalized to [0, 1] at dataset level (day_of_year / 365.25)
        # Just clamp to ensure valid range in case of any edge cases
        phase = torch.clamp(time_values, 0.0, 1.0)
        
        pos = phase.unsqueeze(-1)  # (B, S, 1)
        dim_indices = torch.arange(0, embed_dim, 2, device=time_values.device, dtype=time_values.dtype)
        div_term = torch.exp(-math.log(10000.0) * (dim_indices / embed_dim))  # (embed_dim/2,)
        angles = pos * div_term  # (B, S, embed_dim/2)
        
        sin = torch.sin(2 * math.pi * angles)
        cos = torch.cos(2 * math.pi * angles)
        time_feat = torch.zeros((*sin.shape[:2], embed_dim), device=time_values.device, dtype=time_values.dtype)
        time_feat[..., 0::2] = sin
        time_feat[..., 1::2] = cos
        
        return time_feat

    def apply_RevIn(self, x, seq_len):
        B, L = x.shape
        if L % seq_len == 0 and L > 0:
            num_seqs = L // seq_len
            x_reshaped = x.view(B, num_seqs, seq_len)
            means = x_reshaped.mean(-1, keepdim=True).detach()  # (B, num_seqs, 1)
            x_centered = x_reshaped - means
            stdev = torch.sqrt(torch.var(x_centered, dim=-1, keepdim=True, unbiased=False) + 1e-5)
            x_norm = x_centered / stdev
            x_out = x_norm.view(B, L)
            means_out = means.squeeze(-1)
            stdev_out = stdev.squeeze(-1)
        else:
            means_out = x.mean(dim=1, keepdim=True).detach()  # (B,1)
            x_centered = x - means_out
            stdev_out = torch.sqrt(torch.var(x_centered, dim=1, keepdim=True, unbiased=False) + 1e-5)  # (B,1)
            x_out = x_centered / stdev_out
        return x_out, means_out, stdev_out

    def reverse_revin_apply(self, means, stdev, x, pred_len):
        B, L = x.shape

        if L % pred_len == 0 and L > 0:
            num_seqs = max(1, L // max(1, pred_len))
            x_view = x.view(B, num_seqs, -1)
            if means.dim() == 2 and means.size(1) == num_seqs:
                means_b = means.unsqueeze(-1)
                stdev_b = stdev.unsqueeze(-1)
            else:
                means_b = means.view(B, 1, 1).expand(B, num_seqs, 1)
                stdev_b = stdev.view(B, 1, 1).expand(B, num_seqs, 1)
            x_out = x_view * stdev_b + means_b
            return x_out.view(B, L)
        else:
            if means.dim() == 2:
                means_b = means
                stdev_b = stdev
            else:
                means_b = means.view(B, 1)
                stdev_b = stdev.view(B, 1)
            x_out = x * stdev_b + means_b
            return x_out

    def create_student_t_distribution(self, loc, scale, df):
        """
        Create Student-t distribution with learnable degrees of freedom.
        
        Args:
            loc: Location parameter (mean) - (B, S_t)
            scale: Scale parameter (std) - (B, S_t) 
            df: Degrees of freedom - (B, S_t)
            
        Returns:
            StudentT distribution
        """

        if torch.any(torch.isnan(scale)) or torch.any(torch.isinf(scale)):
            print(f"Warning: Raw scale contains NaN/Inf values. Min: {scale.min().item():.6f}, Max: {scale.max().item():.6f}")
        
        # Ensure positive scale and degrees of freedom with more robust bounds
        scale_raw = scale  # Keep raw values for debugging
        scale = F.softplus(scale) + 1e-3  # Ensure positive scale with larger minimum
        df = F.softplus(df) + 2.1  # Ensure df > 2 (finite variance) with larger minimum
        
        if torch.any(torch.isnan(scale)) or torch.any(torch.isinf(scale)):
            print(f"Warning: Transformed scale contains NaN/Inf. Raw min: {scale_raw.min().item():.6f}, Raw max: {scale_raw.max().item():.6f}")
            print(f"Transformed min: {scale.min().item():.6f}, Transformed max: {scale.max().item():.6f}")
        
        # Additional safety: clamp to reasonable ranges to prevent numerical issues
        # For normalized data (mean≈0, std≈1), scale=2.0 allows 95% CI ≈ ±4σ which is reasonable
        scale = torch.clamp(scale, min=1e-3, max=2.0)  # Scale between 0.001 and 2.0 (good for normalized data)
        df = torch.clamp(df, min=2.1, max=100.0)  # df between 2.1 and 100

        return StudentT(df=df, loc=loc, scale=scale)

    def add_contextual_info_with_value(self, x, var_embed, depth_embed, time_embed):
        if self.add_ctxt_type == "add":
            x = x + var_embed
            x = x + depth_embed
            x = x + time_embed
            return x
        elif self.add_ctxt_type == "concat":
            if self.use_time_embed:
                x = torch.cat((x, var_embed, depth_embed, time_embed), dim=-1)
                x = self.concat_proj(x)
                return x
            else:
                x = torch.cat((x, var_embed, depth_embed), dim=-1)
                x = self.concat_proj(x)
                return x
        else:
            raise ValueError(f"Unsupported add_contextual_info type: {self.add_ctxt_type}")
    
    def add_contextual_info_without_value(self, var_embed, depth_embed, time_embed):
        if self.add_ctxt_type == "add":
            x = var_embed + depth_embed + time_embed
            return x
        elif self.add_ctxt_type == "concat":
            if self.use_time_embed:
                x = torch.cat((var_embed, depth_embed, time_embed), dim=-1)
                return x
            else:
                x = torch.cat((var_embed, depth_embed), dim=-1)
                return x
        else:
            raise ValueError(f"Unsupported add_contextual_info type: {self.add_ctxt_type}")
    
    def token_embed(self, data, depth_values, time_values, variate_ids, include_value: bool = True):
        # TIME EMBEDDING
        if self.use_time_embed:
            if time_values is not None:
                time_feats = self._sinusoidal_time_embed(time_values, self.time_embed_dim)
            else:
                raise RuntimeError("time_values must be provided when use_time_embed is True")
        else:
            time_feats = None

        # VALUE EMBEDDING (SwiGLU for scalar tokens) or zeros for queries
        if include_value:
            data_scalar = data.unsqueeze(-1)  # (B, S, 1)
            a = self.value_proj_a(data_scalar)
            b = self.value_proj_b(data_scalar)
            swish_a = a * torch.sigmoid(a)
            x = swish_a * b  # (B, S, inp_embed_dim)
        else:
            x = None
        
        # VARIABLE ID EMBEDDING
        var_embed = self.var_id_embed(variate_ids)
        
        # DEPTH EMBEDDING
        depth_enc = self.depth_pos_encoder(depth_values)  # (B, L, 2*num_bands + 1)
        depth_embed = self.depth_proj(depth_enc)  # (B, L, d_model)
        
        # COMPLETE TOKEN EMBEDDING
        if include_value:
            x = self.add_contextual_info_with_value(x, var_embed, depth_embed, time_feats)
        else:
            x = self.add_contextual_info_without_value(var_embed, depth_embed, time_feats)
        return x
        
    def forward(
                self,
                data: torch.Tensor,                 # (B, S)
                observed_mask: torch.Tensor,        # (B, S)
                sample_ids: torch.Tensor,           # (B, S)
                variate_ids: torch.Tensor,          # (B, S)
                padding_mask: torch.Tensor,         # (B, S)
                depth_values: torch.Tensor,         # (B, S)
                pred_len: int,
                seq_len: int,
                time_values: torch.Tensor = None,   # (B, S) optional per-token times
                time_ids: torch.Tensor = None,      # optional legacy time ids
                # decoder target metadata
                tgt_variate_ids: torch.Tensor = None,
                tgt_time_values: torch.Tensor = None,
                tgt_time_ids: torch.Tensor = None,
                tgt_depth_values: torch.Tensor = None,
                tgt_padding_mask: torch.Tensor = None
            ) -> torch.Tensor:

        # build a per‐token validity mask, patching it if needed
        valid_mask = observed_mask.bool() & padding_mask  # (B, S)
        # valid_mask = observed_mask.bool()
        if self.tokenization == "patch":
            B, L = valid_mask.shape
            p = self.patch_size
            valid_mask = valid_mask.view(B, L//p, p).all(dim=-1)
        attn_mask_ = valid_mask.unsqueeze(1)# & valid_mask.unsqueeze(2)
        
        if self.revin:
            data, means, stdev = self.apply_RevIn(data, seq_len) # (B, S)

        # token embedding
        x = self.token_embed(data=data,
                             depth_values=depth_values,
                             time_values=time_values,
                             variate_ids=variate_ids,
                             include_value=True)

        x_enc = self.encoder(
            x,
            attn_mask=attn_mask_,
            time_id=time_ids,
            var_id=variate_ids
        )
        # soft-split projections
        z_static_tokens = self.static_proj(x_enc)
        z_temporal_tokens = self.temporal_proj(x_enc)

        # Decoder path (if targets provided)
        if tgt_variate_ids is not None and tgt_time_values is not None and tgt_depth_values is not None:
            B = data.shape[0]
            S_t = tgt_variate_ids.shape[1]
            query_embed = self.token_embed(data=None,
                                           depth_values=tgt_depth_values,
                                           time_values=tgt_time_values,
                                           variate_ids=tgt_variate_ids,
                                           include_value=False)
            query_embed = self.query_proj(query_embed)

            if tgt_padding_mask is not None:
                self_attn_mask = tgt_padding_mask.unsqueeze(1) & tgt_padding_mask.unsqueeze(2)
                cross_attn_mask = tgt_padding_mask.unsqueeze(2) & valid_mask.unsqueeze(1)
            else:
                self_attn_mask = None
                cross_attn_mask = None
            
            dec_out = self.decoder(
                tgt=query_embed,
                memory=z_temporal_tokens,
                self_attn_mask=self_attn_mask,
                cross_attn_mask=cross_attn_mask,
                var_id=tgt_variate_ids,
                time_id=tgt_time_ids,
                mem_var_id=variate_ids,
                mem_time_id=time_ids,
            )

            # Probabilistic forecasting: predict Student-t distribution parameters
            loc_raw = self.loc_head(dec_out).squeeze(-1)  # (B, S_t)
            scale_raw = self.scale_head(dec_out).squeeze(-1)  # (B, S_t)
            
            # Degrees of freedom: hybrid approach if variate_wise_df is enabled
            if self.variate_wise_df:
                # Get base df from variate embedding: shared or separate
                if self.shared_variate_embedding_for_df:
                    var_embed = self.var_id_embed(tgt_variate_ids)  # (B, S_t, var_embed_dim)
                    if self.stop_grad_on_shared_var_embed_for_df:
                        var_embed = var_embed.detach()
                    df_base = self.df_base_projector(var_embed).squeeze(-1)  # (B, S_t)
                else:
                    df_base = self.variate_df_embedding(tgt_variate_ids).squeeze(-1)  # (B, S_t)

                # Get context-aware adjustment from prediction head
                df_adjustment = self.dof_head(dec_out).squeeze(-1)  # (B, S_t)

                # Combine: base + adjustment (hybrid approach)
                df_raw = df_base + df_adjustment  # (B, S_t)

                # Transform base df to positive domain and clamp to stable range for logging/plotting
                df_base_transformed = torch.clamp(F.softplus(df_base) + 2.1, min=2.1, max=100.0)  # (B, S_t)
            else:
                # Standard approach: directly predict df from context
                df_raw = self.dof_head(dec_out).squeeze(-1)  # (B, S_t)
                
            # Create Student-t distribution (this applies softplus + clamping)
            student_t_dist = self.create_student_t_distribution(loc_raw, scale_raw, df_raw)
            
            # Extract the CLAMPED parameters from the distribution for logging/plotting
            # This ensures we return the same values used in the distribution
            loc = student_t_dist.loc  # Same as loc_raw (no transformation)
            scale = student_t_dist.scale  # Clamped scale
            df = student_t_dist.df  # Clamped df
            
            # For training: return distribution parameters and distribution
            # For inference: can sample from distribution or use mean
            forecasts = {
                'loc': loc,
                'scale': scale,  # Now returns CLAMPED scale
                'df': df,  # Now returns CLAMPED df
                'distribution': student_t_dist
            }
            if self.variate_wise_df:
                forecasts['df_base'] = df_base_transformed  # (B, S_t)
            
            if self.revin:
                forecasts['loc'] = self.reverse_revin_apply(means, stdev, forecasts['loc'], pred_len)
        else:
            # fallback to MLP forecast head if no targets given
            forecasts = self.forecast_head.forecast(z_temporal_tokens, seq_len//self.patch_size, pred_len)

            if self.revin:
                forecasts = self.reverse_revin_apply(means, stdev, forecasts, pred_len)
        
        # contrastive representation uses pooled static tokens only
        z = self.cl_proj(z_static_tokens, valid_mask)
        
        return z_temporal_tokens, forecasts, z