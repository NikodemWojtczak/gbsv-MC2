import cv2
import matplotlib.pyplot as plt


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
