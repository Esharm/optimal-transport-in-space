from pathlib import Path
from PIL import Image


input_folder = Path("final_frames_34_png")
output_folder = Path("resized_128")

output_folder.mkdir(parents=True, exist_ok=True)

png_files = sorted(input_folder.glob("*.png"))

if len(png_files) != 15:
    raise ValueError(f"Expected 15 PNG files, but found {len(png_files)}")

for png_file in png_files:
    with Image.open(png_file) as img:
        resized = img.resize((128, 128), Image.Resampling.LANCZOS)
        resized.save(output_folder / png_file.name)

print(f"Resized {len(png_files)} PNGs to 128x128")