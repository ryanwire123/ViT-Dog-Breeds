from transformers import ViTFeatureExtractor

def get_feature_extractor():
    """Initialize the ViT feature extractor."""
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    return feature_extractor

def preprocess_data(dataset, feature_extractor):
    """Preprocess dataset using the feature extractor."""
    def transform(batch):
        inputs = feature_extractor(images=batch['image'], return_tensors="pt")
        batch['pixel_values'] = inputs['pixel_values']
        return batch

    return dataset.map(transform, batched=True)
