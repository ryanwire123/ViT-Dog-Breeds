import torch
from transformers import ViTForImageClassification

def initialize_model(num_labels):
    """Initialize the ViT model for image classification."""
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=num_labels
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model
