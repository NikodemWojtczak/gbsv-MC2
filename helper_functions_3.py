import cv2
import numpy as np
import matplotlib.pyplot as plt


def display_image(image, title="Image", cmap=None, figsize=(6, 6)):
    # Ensure image data type is suitable for display (e.g., uint8)
    if image.dtype != np.uint8:
        # Basic normalization/scaling for display if needed (e.g., for float noise)
        if np.min(image) < 0 or np.max(image) > 255:
            img_display = cv2.normalize(
                image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
        else:
            # Assume it's float in [0, 255] or similar, convert safely
            img_display = image.astype(np.uint8)
    else:
        img_display = image

    plt.figure(figsize=figsize)
    if cmap:
        if len(img_display.shape) > 2:
            img_display = img_display[:, :, 0]
        plt.imshow(img_display, cmap=cmap)
    else:
        # OpenCV BGR to Matplotlib RGB
        if len(img_display.shape) == 3:
            plt.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(img_display, cmap="gray")  # Grayscale
    plt.title(title)
    plt.axis("off")
    plt.show()
