import cv2
import numpy as np
import matplotlib.pyplot as plt


# --- Update the display function ---
def display_results_custom(results, figsize=(15, 12)):
    """Displays the original image, preprocessing steps, edges, accumulator slice, and detected circles."""
    params = results["params"]

    # Display each step individually for better visualization

    # Original
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(results["original"], cv2.COLOR_BGR2RGB))
    plt.title("Step 0: Original Image")
    plt.axis("off")
    plt.show()

    # Grayscale
    plt.figure(figsize=(8, 6))
    plt.imshow(results["grayscale"], cmap="gray")
    plt.title("Step 1a: Grayscale")
    plt.axis("off")
    plt.show()

    # Blurred
    plt.figure(figsize=(8, 6))
    plt.imshow(results["blurred"], cmap="gray")
    plt.title("Step 1b: Blurred")
    plt.axis("off")
    plt.show()

    # Edges (Canny)
    plt.figure(figsize=(8, 6))
    if results["edges"] is not None:
        plt.imshow(results["edges"], cmap="gray")
        plt.title(f"Step 2a: Canny Edges (T={params['p1']//2}, {params['p1']})")
    else:
        plt.text(0.5, 0.5, "No Edges Found", ha="center", va="center")
        plt.title("Step 2a: Canny Edges")
    plt.axis("off")
    plt.show()

    # Accumulator (show a slice or projection)
    plt.figure(figsize=(8, 6))
    if results["accumulator"] is not None:
        # Show sum projection along the radius axis for a 2D view of center votes
        acc_proj = np.sum(results["accumulator"], axis=0)
        # Normalize for display
        acc_display = acc_proj / (acc_proj.max() + 1e-6)  # Avoid div by zero
        im = plt.imshow(acc_display, cmap="viridis", aspect="auto")
        plt.title(
            f"Step 2b: Accumulator (Sum Projection)\nThresh={params['acc_thresh']}"
        )
        plt.colorbar(im, fraction=0.046, pad=0.04)
    else:
        plt.text(0.5, 0.5, "No Accumulator", ha="center", va="center")
        plt.title("Step 2b: Accumulator")
    plt.axis("off")
    plt.show()

    # Circles overlaid on original
    plt.figure(figsize=(8, 6))
    output = results["original"].copy()
    num_circles_found = 0
    if results["circles"] is not None:
        # OpenCV returns circles[0] which is the list, our custom returns the list directly
        # Adjusting access based on the typical cv2 output structure [[]]
        circles_to_draw = (
            results["circles"][0]
            if len(results["circles"].shape) == 3
            else results["circles"]
        )
        circles_to_draw = np.uint16(
            np.around(circles_to_draw)
        )  # Convert to int for drawing
        num_circles_found = len(circles_to_draw)
        for i in circles_to_draw:
            center = (i[0], i[1])  # x, y
            radius = i[2]
            # Draw the outer circle (Green)
            cv2.circle(output, center, radius, (0, 255, 0), 2)
            # Draw the center of the circle (Red)
            cv2.circle(output, center, 2, (0, 0, 255), 3)

    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.title(f"Step 2c: Detected Circles: {num_circles_found}")
    plt.axis("off")
    plt.show()


# --- Helper to create dummy example (Identical to your original) ---
def create_dummy_example():
    # Create a blank image (white background)
    width, height = 500, 500
    image = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Draw some circles of different sizes
    centers = [(100, 100), (250, 250), (400, 150), (150, 350), (350, 400)]
    radii = [30, 50, 25, 40, 35]
    colors = [
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
        (255, 255, 0),
        (0, 255, 255),
    ]  # Colors don't matter for detection

    for center, radius, color in zip(centers, radii, colors):
        # Draw outline circles instead of filled for better edge detection
        cv2.circle(image, center, radius, (0, 0, 0), 2)  # Black outline, thickness 2

    print(f"Created dummy image with {len(centers)} circles")
    return image


def display_image(image, title="Image", cmap=None):
    plt.figure(figsize=(8, 6))
    if cmap:
        plt.imshow(image, cmap=cmap)
    else:
        # OpenCV loads images in BGR, Matplotlib expects RGB
        if len(image.shape) == 3:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(image, cmap="gray")  # Display grayscale correctly
    plt.title(title)
    plt.axis("off")
    plt.show()
