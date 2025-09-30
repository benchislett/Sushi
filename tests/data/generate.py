import os
from pathlib import Path

from PIL import Image, ImageDraw


def generate_expected_image() -> None:
    """
    Generates and saves the reference image needed for the unit tests.
    """
    # Define image dimensions and create a blank white canvas
    width, height = 200, 150
    image = Image.new("RGB", (width, height), (255, 255, 255))

    vertices = [(10, 10), (50, 140), (195, 70)]
    color = (200, 55, 79, 192)

    draw = ImageDraw.Draw(image, "RGBA")
    draw.polygon(vertices, fill=color)

    output_dir = Path(__file__).parent
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir / "expected_sample_triangle.png"

    image.convert("RGB").save(output_path)
    print(f"Successfully generated and saved '{output_path}'")


if __name__ == "__main__":
    generate_expected_image()
