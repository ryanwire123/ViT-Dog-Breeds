from datasets import load_dataset

def load_dog_breed_dataset():
    """Load the Dogs Breed dataset from Hugging Face."""
    dataset = load_dataset("jhoppanne/Dogs-Breed-Image-Classification-V2")
    return dataset
