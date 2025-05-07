import cv2
import matplotlib.pyplot as plt
import numpy as np


def display_image(image, title="Image", cmap=None, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    if cmap:
        # Ensure image is 2D for colormaps
        if len(image.shape) > 2:
            image_display = image[
                :, :, 0
            ]  # Display first channel if multi-channel grayscale
        else:
            image_display = image
        plt.imshow(image_display, cmap=cmap)
    else:
        # OpenCV BGR to Matplotlib RGB
        if len(image.shape) == 3:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(image, cmap="gray")  # Grayscale
    plt.title(title)
    plt.axis("off")
    plt.show()


# Helper function to display labeled images with distinct colors
def display_labeled_image(labeled_image, title="Labeled Image", figsize=(6, 6)):
    # Map component labels to hue value, 0 label is background
    label_hue = np.uint8(179 * labeled_image / np.max(labeled_image))
    # Set background to black
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img_colored = cv2.merge([label_hue, blank_ch, blank_ch])
    # Convert from HSV to BGR
    labeled_img_colored = cv2.cvtColor(labeled_img_colored, cv2.COLOR_HSV2BGR)
    # Set background label (0) to black
    labeled_img_colored[label_hue == 0] = 0

    plt.figure(figsize=figsize)
    plt.imshow(
        cv2.cvtColor(labeled_img_colored, cv2.COLOR_BGR2RGB)
    )  # Convert BGR to RGB for display
    plt.title(title)
    plt.axis("off")
    plt.show()
