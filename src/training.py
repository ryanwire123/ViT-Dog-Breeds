import torch
from sklearn.metrics import accuracy_score

def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    """Train the model for one epoch."""
    model.train()
    for batch in dataloader:
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def validate_model(model, dataloader, device):
    """Validate the model and calculate accuracy."""
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(pixel_values=pixel_values)
            preds = outputs.logits.argmax(dim=-1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    return accuracy
