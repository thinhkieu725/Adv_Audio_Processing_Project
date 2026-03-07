import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "unilm/beats"))

from BEATs import BEATs, BEATsConfig
import torch
import torch.nn as nn

def init_BEATs_model(checkpoint_path: str = './BEATs_iter3_plus_AS2M.pt'):
    checkpoint = torch.load(checkpoint_path)

    cfg = BEATsConfig(checkpoint['cfg'])
    BEATs_model = BEATs(cfg)
    BEATs_model.load_state_dict(checkpoint['model'])

    return BEATs_model


class ClassifierModel(nn.Module):
    def __init__(self, 
                 num_parent_classes: int = 5,
                 num_leaf_classes: int = 2,
                 hidden_dim: int = 256,
                 dropout: float = 0.2,
                 BEATs_checkpoint_path: str = './BEATs_iter3_plus_AS2M.pt'):
        super().__init__()
        self.num_parent_classes = num_parent_classes
        self.num_leaf_classes = num_leaf_classes
        self.hidden_dim = hidden_dim
        self.dropout = dropout
    
        self.BEATs_model = init_BEATs_model(BEATs_checkpoint_path)
        self.BEATs_output_dim = 768
        
        self.shared_block = nn.Sequential(
            nn.LayerNorm(self.BEATs_output_dim),
            nn.Linear(self.BEATs_output_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )

        self.parent_classifier = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.num_parent_classes),
            )
        self.leaf_classifier = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.num_leaf_classes),
            )
        
        
    @staticmethod
    def masked_mean_pool(
        x: torch.Tensor,
        padding_mask_token: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Args
        ----
        x : [B, T, D]
            token embeddings

        padding_mask : [B, T]
            True = padded token
            False = valid token

        Returns
        -------
        pooled : [B, D]
        """

        if padding_mask_token is None:
            return x.mean(dim=1)

        valid = (~padding_mask_token).float()

        pooled = (x * valid.unsqueeze(-1)).sum(dim=1)
        pooled = pooled / valid.sum(dim=1, keepdim=True).clamp(min=1)

        return pooled

    def forward(self, 
                x: torch.Tensor, 
                padding_mask: torch.Tensor):
        features, padding_mask_token = self.BEATs_model.extract_features(x, padding_mask=padding_mask)
        pooled_features = self.masked_mean_pool(features, padding_mask_token)
        shared_representation = self.shared_block(pooled_features)
        parent_logits = self.parent_classifier(shared_representation)
        leaf_logits = self.leaf_classifier(shared_representation)

        return parent_logits, leaf_logits

    @torch.no_grad()
    def predict(self, 
                x: torch.Tensor, 
                padding_mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inference method for the model.
        
        Args:
            x: Audio input tensor [B, T]
            padding_mask: Padding mask [B, T], defaults to no padding if None
            
        Returns:
            parent_pred: Predicted parent class indices [B]
            leaf_pred: Predicted leaf class indices [B]
        """
        # Set model to eval mode
        was_training = self.training
        self.eval()
        
        try:
            # If no padding mask provided, assume no padding
            if padding_mask is None:
                padding_mask = torch.zeros_like(x, dtype=torch.bool)
            
            # Forward pass
            parent_logits, leaf_logits = self.forward(x, padding_mask)
            
            # Get predicted classes (argmax)
            parent_pred = parent_logits.argmax(dim=-1)
            leaf_pred = leaf_logits.argmax(dim=-1)
            
            return parent_pred, leaf_pred
        finally:
            # Restore training mode if it was enabled
            if was_training:
                self.train()
        
