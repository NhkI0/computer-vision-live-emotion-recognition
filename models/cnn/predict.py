"""Prediction module for EmotionCNN model."""

import os
from pathlib import Path

import numpy as np
import torch
import cv2
from torchvision import transforms

from .CNN import EmotionCNN

IMG_SIZE = 48
CLASSES = ["negative", "neutral", "positive"]
CLASS_NAMES = ["Negative", "Neutre", "Positive"]


def get_model_path():
    """Get the path to the trained model."""
    return Path(__file__).parent / "best_model.pth"


def get_device():
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_model(device=None):
    """Load the trained EmotionCNN model.

    Args:
        device: torch device to load the model on. If None, auto-detects.

    Returns:
        Loaded EmotionCNN model in eval mode.
    """
    if device is None:
        device = get_device()

    model = EmotionCNN(num_classes=3, dropout=0.3).to(device)
    model_path = get_model_path()

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Please train the model first using: python models/cnn/training.py"
        )

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model


# Global model cache
_model = None
_device = None
_transform = None


def _get_transform():
    """Get or create the preprocessing transform."""
    global _transform
    if _transform is None:
        _transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Grayscale(1),
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
    return _transform


def predict_emotion(image: np.ndarray) -> str:
    """Predict emotion from a face image (BGR format).

    Args:
        image: numpy array of shape (H, W, 3) in BGR format (OpenCV default).
               Should be a cropped face region.

    Returns:
        str: Predicted emotion - "Negative", "Neutre", or "Positive".
    """
    global _model, _device

    # Lazy load model
    if _model is None:
        _device = get_device()
        _model = load_model(_device)
        print(f"Model loaded on {_device}")

    # Preprocess image
    transform = _get_transform()

    # Convert BGR to RGB for PIL
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply transforms
    input_tensor = transform(image_rgb).unsqueeze(0).to(_device)

    # Predict
    with torch.no_grad():
        outputs = _model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    return CLASS_NAMES[predicted_class], confidence


def predict_emotion_simple(image: np.ndarray) -> str:
    """Simple prediction function returning only the emotion string.

    Args:
        image: numpy array of shape (H, W, 3) in BGR format.

    Returns:
        str: "Negative", "Neutre", or "Positive".
    """
    emotion, _ = predict_emotion(image)
    return emotion


if __name__ == "__main__":
    # Test the prediction function
    print("Testing predict_emotion function...")

    # Create a dummy 48x48 image for testing
    test_image = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)

    try:
        emotion, confidence = predict_emotion(test_image)
        print(f"Predicted emotion: {emotion} (confidence: {confidence:.4f})")
    except FileNotFoundError as e:
        print(f"Error: {e}")
