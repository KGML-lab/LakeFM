import torch
import torch.nn.functional as F

from abc import ABC, abstractmethod

class Tokenizer(ABC):
    """
    Abstract base tokenizer class for converting input sequences into tokens.
    Subclasses should implement the `tokenize` method.
    """
    @staticmethod
    def build(patch_size, tokenization_type):

        if tokenization_type == "scalar":
            return ScalarTokenizer(), 1
        elif tokenization_type == "patch":
            return PatchTokenizer(patch_size=patch_size), patch_size
        elif tokenization_type == "temporal":
            return TemporalTokenizer(), 1
        else:
            raise ValueError(f"Unsupported tokenization method: {tokenization_type}")

    @abstractmethod
    def tokenize(self, sequence: torch.Tensor, **kwargs) -> torch.Tensor:
        pass

class ScalarTokenizer(Tokenizer):
    """
    Tokenizer that treats the entire sequence as a single token.
    """

    def tokenize(self, sequence: torch.Tensor, 
                 variate_ids: torch.Tensor, 
                 time_ids: torch.Tensor, 
                 depth_values: torch.Tensor,
                 seq_len: int,
                 sample_ids) -> tuple:
        """
        sequence: Tensor of shape (S,)
        variate_id, time_id, depth_values: Tensor of shape (S,)
        """
        return sequence.unsqueeze(-1), variate_ids, time_ids, depth_values, sample_ids
    
class PatchTokenizer(Tokenizer):
    """
    Tokenizer that splits the sequence into patches
    """
    def __init__(self, patch_size: int = 1):
        self.patch_size = patch_size
    
    def patchify_meta_sequence(self, 
                               meta_seq: torch.Tensor,  
                               method: str = 'first') -> torch.Tensor:
        """
        Patchifies metadata (e.g., variate IDs, depth values) by reducing each patch to a single value

        """
        B, S = meta_seq.shape
        pad_len = (self.patch_size - (S % self.patch_size)) % self.patch_size
        if pad_len > 0:
            meta_seq = F.pad(meta_seq, (0, pad_len), value=-1)
        
        meta_seq = meta_seq.view(B, -1, self.patch_size)  # (B, L, P)

        if method == 'first':
            return meta_seq[:, :, 0]
        elif method == 'mode':
            return meta_seq.mode(dim=-1).values
        elif method == 'mean':
            return meta_seq.float().mean(dim=-1)
        else:
            raise ValueError(f"Unsupported method: {method}")

    def patchify_sequence(self, 
                          sequence: torch.Tensor, 
                          pad_value: float = 0.0) -> torch.Tensor:
        """
        Patchify a flattened time-series sequence tensor.
        
        """
        B, S = sequence.shape
        assert S % self.seq_len == 0, "Total sequence length must be divisible by seq_len"

        num_sequences = S // self.seq_len

        # Reshape to (B, num_sequences, seq_len)
        sequence = sequence.view(B, num_sequences, self.seq_len)
        
        # Pad along seq_len if not divisible by patch_size
        pad_len = (self.patch_size - (self.seq_len % self.patch_size)) % self.patch_size
        if pad_len > 0:
            sequence = F.pad(sequence, (0, pad_len), value=pad_value)  # pad last dim

        # New length after padding
        new_seq_len = (self.seq_len + pad_len) // self.patch_size

        # Reshape to patches: (B, num_sequences, new_seq_len, patch_size)
        sequence = sequence.view(B, num_sequences, new_seq_len, self.patch_size)

        # Merge sequences: (B, num_sequences * new_seq_len, patch_size)
        sequence = sequence.view(B, -1, self.patch_size)

        return sequence

    def tokenize(self, 
                 sequence: torch.Tensor, 
                 variate_ids: torch.Tensor, 
                 time_ids: torch.Tensor, 
                 depth_values: torch.Tensor,
                 seq_len: int,
                 sample_ids: torch.Tensor) -> tuple:
        """
        sequence: Tensor of shape (S,)
        variate_id, time_id, depth_values: Tensor of shape (S,)
        Returns tokenized patches of shape (num_patches, patch_size)
        """
        self.seq_len = seq_len
        sequence_patches = self.patchify_sequence(sequence)
        patched_var_ids = self.patchify_meta_sequence(variate_ids)
        patched_depths = self.patchify_meta_sequence(depth_values)
        patched_times = self.patchify_meta_sequence(time_ids)
        patched_samples = self.patchify_meta_sequence(sample_ids)

        return sequence_patches, patched_var_ids, patched_times, patched_depths, patched_samples


class TemporalTokenizer(Tokenizer):
    """
    Tokenizer that treats the entire temporal sequence as a single token (Not yet implemented).
    """
    def tokenize(self, sequence: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError("TemporalTokenizer is not implemented yet.")
    