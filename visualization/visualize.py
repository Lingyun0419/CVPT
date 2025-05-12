import timm  # PyTorch image models library
import torch  # PyTorch deep learning library
from skimage import io  # Image processing from scikit-image

from gradcam import GradCam  # Grad-CAM implementation
import numpy as np  # NumPy for numerical operations
import cv2  # OpenCV for image processing
from CVPT import CVPT
from VPT import VPT

# Function to prepare input image for the model
def prepare_input(image):
    image = image.copy()  # Copy the image to avoid modifying the original

    # Normalize the image using the mean and standard deviation
    means = np.array([0.5, 0.5, 0.5])
    stds = np.array([0.5, 0.5, 0.5])
    image -= means
    image /= stds

    # Transpose the image to match the model's expected input format (C, H, W)
    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))
    image = image[np.newaxis, ...]  # Add batch dimension

    return torch.tensor(image, requires_grad=True)  # Convert to PyTorch tensor


# Function to generate a Grad-CAM heatmap
def gen_cam(image, mask):
    # Create a heatmap from the Grad-CAM mask
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    # Superimpose the heatmap on the original image
    cam = (1 - 0.5) * heatmap + 0.5 * image
    cam = cam / np.max(cam)  # Normalize the result
    return np.uint8(255 * cam)  # Convert to 8-bit image


if __name__ == '__main__':
    # Load and preprocess the input image
    img = io.imread("test_pic.png")
    prompt_num = 200
    # CVPT or VPT model
    model_type = 'VPT'
    if img.shape[2] == 4:
        img = img[:, :, :3]
    img = np.float32(cv2.resize(img, (224, 224))) / 255  # Resize and normalize
    inputs = prepare_input(img)  # Prepare the image for the model

    # Create the Vision Transformer model with pretrained weights
    if model_type == 'CVPT':
        model = CVPT(Prompt_num=prompt_num, num_classes= 100)
    elif model_type == 'VPT':
        model = VPT(Prompt_num=prompt_num, num_classes=100)
    else:
        raise ValueError("Choose CVPT or VPT")
    model.load_state_dict(torch.load(f"checkpoints/{model_type}_{prompt_num}.pth"))
    target_layer = model.blocks[-1].norm1  # Specify the target layer for Grad-CAM

    # Initialize Grad-CAM with the model and target layer
    grad_cam = GradCam(model, target_layer)
    mask = grad_cam(inputs)  # Compute the Grad-CAM mask
    result = gen_cam(img, mask)  # Generate the Grad-CAM heatmap

    # Save the result to an image file
    cv2.imwrite(f"pic/{model_type}_{prompt_num}.jpg", result)
