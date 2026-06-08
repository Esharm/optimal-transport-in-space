import cv2
import os

def images_to_video(image_folder, output_file, fps=30, pause_seconds=1):
    images = sorted([
        f for f in os.listdir(image_folder)
        if f.lower().endswith(".png")
    ])

    if not images:
        raise ValueError("No PNG images found in folder")

    first = cv2.imread(os.path.join(image_folder, images[0]))
    h, w, _ = first.shape

    video = cv2.VideoWriter(
        output_file,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    frames_per_image = int(pause_seconds * fps)

    for img in images:
        path = os.path.join(image_folder, img)
        frame = cv2.imread(path)

        for _ in range(frames_per_image):
            video.write(frame)

    video.release()