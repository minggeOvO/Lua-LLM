"""
Masked Linear layers with importance-based sigmoid function for pruning.
This module replaces the external hypernetwork approach with integrated trainable masks.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

def hard_sample(out):
    binary_out = torch.round(out)  
    binary_out = (binary_out - out).detach() + out  
    return binary_out
    
class MaskedLinear(nn.Module):
    """
    Linear layer with integrated learnable masks using importance-based sigmoid function.
    
    This layer combines a regular linear layer with learnable mask parameters that
    determine which weights should be pruned. The mask generation uses an importance-based
    sigmoid function similar to the topk_hypernetwork approach.
    """
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True,
        mask_init_value: float = 0.1
    ):
        """
        Initialize MaskedLinear layer.
        
        Args:
            in_features: Size of each input sample
            out_features: Size of each output sample  
            bias: If set to False, the layer will not learn an additive bias
            mask_init_value: Initial value for mask logits (controls initial sparsity)
        """
        super().__init__()
        
        # Standard linear layer components
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.bias = None
            
        # Mask parameters - learnable logits that determine mask values
        self.mask_logits = nn.Parameter(torch.full((out_features,1), mask_init_value))

    # def get_mask(self) -> torch.Tensor:
    #     """
    #     Generate mask using importance-based sigmoid function.
        
    #     Returns:
    #         Binary-like mask tensor with shape (out_features, in_features)
    #     """
        
    #     # Compute importance-based mask using sigmoid
    #     # Higher importance scores lead to higher probability of being kept
    #     # importance_expanded = importance_scores.unsqueeze(0)  # (1, in_features)
        
    #     # Sigmoid function: higher importance - lower logit = higher probability of being kept
    #     mask_probs = F.sigmoid((self.importance_scores - self.mask_logits.unsqueeze(-1)) * self.in_features)
        
    #     if self.training:
    #         # During training, use soft masks
    #         return mask_probs
    #     else:
    #         # During inference, use hard masks
    #         return (mask_probs > 0.5).float()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with masked weights.
        
        Args:
            input: Input tensor of shape (..., in_features)
            
        Returns:
            Output tensor of shape (..., out_features)
        """
        if not self.training:
            self.tp_out = F.sigmoid((self.importance_scores - self.mask_logits) * self.in_features)
            soft_tp_out = self.tp_out
            self.tp_out = hard_sample(self.tp_out)
            if self.tp_out.sum() == 0:
                self.tp_out[soft_tp_out.argmax()] = 1
        else:
            self.tp_out = F.sigmoid((self.importance_scores - self.mask_logits) * self.in_features)
        
        # Apply mask to weights
        masked_weight = self.weight * self.tp_out
        
        # Perform linear transformation
        if self.bias is None:
            return F.linear(input, masked_weight)
        else:
            return F.linear(input, masked_weight, self.bias)
    
    def get_parameter_count(self) -> int:
        """
        Get effective parameter count after masking.
        
        Returns:
            Number of non-zero parameters
        """
        # with torch.no_grad():
        #     self.tp_out = F.sigmoid((self.importance_scores - self.mask_logits) * self.in_features)
        mask = hard_sample(self.tp_out)
        # return int(mask.sum().item())
        return mask.sum()
        
    def get_full_parameter_count(self) -> int:
        return self.in_features * self.out_features
    
    def set_importance_scores(self, importance_scores: torch.Tensor):
        """
        Set or update importance scores for this layer.
        
        Args:
            importance_scores: Importance scores tensor with shape (in_features,)
        """
        # Ensure dtype/device alignment with weight for mixed precision safety
        if importance_scores.dtype != self.weight.dtype:
            importance_scores = importance_scores.to(self.weight.dtype)
        if importance_scores.device != self.weight.device:
            importance_scores = importance_scores.to(self.weight.device)
        self.register_buffer('importance_scores', importance_scores)


class MaskedLinearGQA(MaskedLinear):
    """
    MaskedLinear variant for Grouped Query Attention layers.
    
    This version handles the special case where attention layers may have 
    grouped query attention patterns that require different parameter counting.
    """
    
    def __init__(self, *args, gqa_groups: int = 1, **kwargs):
        """
        Initialize MaskedLinearGQA layer.
        
        Args:
            gqa_groups: Number of groups for grouped query attention
            *args, **kwargs: Arguments passed to parent class
        """
        super().__init__(*args, **kwargs)
        self.gqa_groups = gqa_groups
    
    def get_parameter_count(self) -> int:
        """
        Get effective parameter count for GQA layers.
        
        Returns:
            Number of non-zero parameters accounting for GQA groups
        """
        mask = hard_sample(self.tp_out)
        return mask.sum() * self.gqa_groups
        
    def get_full_parameter_count(self):
        return self.in_features * self.out_features * self.gqa_groups
    
    def set_importance_scores(self, importance_scores: torch.Tensor):
        """
        Set or update importance scores for this GQA layer.
        
        Args:
            importance_scores: Importance scores tensor with shape (in_features,)
        """
        if importance_scores.dtype != self.weight.dtype:
            importance_scores = importance_scores.to(self.weight.dtype)
        if importance_scores.device != self.weight.device:
            importance_scores = importance_scores.to(self.weight.device)
        self.register_buffer('importance_scores', importance_scores)
