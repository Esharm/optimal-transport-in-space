import os
import numpy as np
from PIL import Image
import cv2


# ============================================================
# LOAD IMAGE SEQUENCE (frames -> tensor)
# ============================================================

def load_frames(image_folder, max_frames=30, resize=(128, 128)):
    """
    Loads images from a folder into a tensor [K, H, W]
    """

    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"Folder not found: {image_folder}")

    images = sorted([
        f for f in os.listdir(image_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])[:max_frames]

    if len(images) == 0:
        raise ValueError(f"No images found in {image_folder}")

    frames = []

    for name in images:
        path = os.path.join(image_folder, name)

        img = Image.open(path).convert("L")  # grayscale
        img = img.resize(resize)

        frames.append(np.array(img, dtype=np.float32) / 255.0)

    return np.stack(frames), images


# ============================================================
# SAVE IMAGE SEQUENCE (tensor -> images)
# ============================================================

def save_frames(frames, output_folder, names=None):
    """
    Saves [K, H, W] back to disk as images
    """

    os.makedirs(output_folder, exist_ok=True)

    K = frames.shape[0]

    for k in range(K):

        img = np.clip(frames[k], 0, 1)
        img = (img * 255).astype(np.uint8)

        if names is not None:
            name = f"rec_{names[k]}"
        else:
            name = f"rec_{k:03d}.png"

        Image.fromarray(img).save(os.path.join(output_folder, name))


# ============================================================
# VIDEO EXPORT (replacement for your OpenCV function)
# ============================================================

def images_to_video(image_folder, output_file, fps=30, pause_seconds=1):
    """
    Converts a folder of images into a video.
    """

    images = sorted([
        f for f in os.listdir(image_folder)
        if f.lower().endswith(".png")
    ])

    if len(images) == 0:
        raise ValueError("No PNG images found in folder")

    first = cv2.imread(os.path.join(image_folder, images[0]))

    if first is None:
        raise ValueError("Could not read first image")

    h, w, _ = first.shape

    video = cv2.VideoWriter(
        output_file,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    frames_per_image = max(1, int(pause_seconds * fps))

    for img_name in images:
        path = os.path.join(image_folder, img_name)
        frame = cv2.imread(path)

        if frame is None:
            continue

        for _ in range(frames_per_image):
            video.write(frame)

    video.release()