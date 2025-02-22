from pathlib import Path
from PIL import Image


"""

THIS DOESN'T WORK

the maximum width displayable on our site is 1600px (really 1650, but what's 50pxs between friends)
This is double resolution for the retina folks - true width is 825 pxs
"""

TARGET_WIDTH = 1600
IMAGE_DIRECTORY = Path("docs/_images")
IMAGE_EXTENSIONS = {'.jpg', '.png', '.gif'}

def resize_image(file_path: str, target_width: int) -> None:
    """
    Resize images for web display, optimizing for balance of quality and file size.
    Handles static images (PNG, JPEG) and animated GIFs.

    Args:
        file_path: Path to the image file
        target_width: Desired width in pixels
    """

    img = Image.open(file_path)

    if img.width < target_width:
        # we don't need to resize
        img.close()
        return
    # Calculate new height maintaining aspect ratio
    aspect_ratio = img.height / img.width
    target_height = int(target_width * aspect_ratio)

    if getattr(img, "is_animated", False):
        img.close()
        return

        # Handle animated GIF
        # frames = []
        # durations = []
        #
        # for i in range(img.n_frames):
        #     img.seek(i)
        #     durations.append(img.info.get('duration', 100))
        #
        #     resized_frame = img.resize(
        #         (target_width, target_height),
        #         resample=Image.Resampling.BILINEAR
        #     )
        #     frames.append(resized_frame.copy())
        #
        # # Save the animated GIF with web optimization
        # frames[0].save(
        #     file_path,
        #     save_all=True,
        #     append_images=frames[1:],
        #     duration=durations,
        #     loop=img.info.get('loop', 0),
        #     optimize=True
        # )

    else:
        # Handle static image
        resized = img.resize(
            (target_width, target_height),
            resample=Image.Resampling.BILINEAR
        )

        # Save with web-optimized settings based on format
        if img.format == 'PNG':
            resized.save(
                file_path,
                format='PNG',
                optimize=True,
                quality=85
            )
        elif img.format == 'GIF':
            resized.save(
                file_path,
                format='GIF',
                optimize=True,
            )
        else:  # JPEG and others
            resized.save(
                file_path,
                format=img.format,
                quality=85,
                optimize=True,
                progressive=True
            )


if __name__ == '__main__':
    image_files = [
        file for file in IMAGE_DIRECTORY.iterdir()
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    i = 0
    for image_file in image_files:

        resize_image(image_file, TARGET_WIDTH)
        i += 1
        if i % 25 == 0:
            print(f'{i} images resized')