import os
from PIL import Image
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
import time
from itertools import islice

MAX_WIDTH = 1600
BATCH_SIZE = 10  # Process images in smaller batches
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBomb warnings


def get_image_info(filepath):
    """Quickly get image dimensions and animation status without loading full image."""
    with Image.open(filepath) as img:
        return {
            'width': img.width,
            'height': img.height,
            'animated': hasattr(img, "n_frames") and img.n_frames > 1,
            'mode': img.mode
        }


def convert_image(args):
    """Combined conversion function for both static and animated images."""
    input_path, output_path, quality = args
    try:
        info = get_image_info(input_path)

        if info['animated']:
            success = convert_animated_gif(input_path, output_path, quality, info)
        else:
            success = convert_static_image(input_path, output_path, quality, info)

        return {
            'status': 'success' if success else 'failed',
            'path': input_path
        }
    except Exception as e:
        return {'status': 'error', 'path': input_path}


def convert_static_image(input_path, output_path, quality, info):
    """Convert a static image to WebP format."""
    with Image.open(input_path) as img:
        # Resize if necessary
        if info['width'] > MAX_WIDTH:
            aspect_ratio = info['height'] / info['width']
            new_width = MAX_WIDTH
            new_height = int(MAX_WIDTH * aspect_ratio)
            img = img.resize((new_width, new_height), Image.BILINEAR)

        # Convert color mode if necessary
        if info['mode'] in ('RGBA', 'LA'):
            background = Image.new('RGBA', img.size, (255, 255, 255, 255))
            background.paste(img, mask=img.split()[-1])
            img = background
        elif info['mode'] != 'RGB':
            img = img.convert('RGB')

        img.save(
            output_path,
            'WEBP',
            quality=quality,
            method=4
        )
        return True


def convert_animated_gif(input_path, output_path, quality, info):
    """Convert an animated GIF to WebP with optimizations."""
    try:
        with Image.open(input_path) as img:
            # Calculate new dimensions once if needed
            need_resize = info['width'] > MAX_WIDTH
            if need_resize:
                aspect_ratio = info['height'] / info['width']
                new_width = MAX_WIDTH
                new_height = int(MAX_WIDTH * aspect_ratio)

            # Get total frames
            n_frames = getattr(img, 'n_frames', 1)

            frames = []
            durations = []

            # Process all frames including first frame
            for frame_idx in range(n_frames):
                img.seek(frame_idx)
                duration = img.info.get('duration', 100)
                frame = img.convert('RGBA' if img.mode == 'P' else img.mode)

                if need_resize:
                    frame = frame.resize((new_width, new_height), Image.BILINEAR)

                frames.append(frame)
                durations.append(duration)

            # Save with optimizations
            if frames:
                frames[0].save(
                    output_path,
                    'WEBP',
                    save_all=True,
                    append_images=frames[1:] if len(frames) > 1 else [],
                    duration=durations,
                    quality=quality,
                    method=4,
                    minimize_size=True,
                    lossless=False
                )

                # Clean up
                for frame in frames:
                    frame.close()

                return True
            return False
    except Exception as e:
        print(f"Error converting {input_path}: {str(e)}")
        return False


def process_batch(batch, pool):
    """Process a batch of images using the process pool."""
    return list(pool.imap_unordered(convert_image, batch))


def batch_convert_to_webp(input_dir, output_dir, num_cores=2, quality=80):
    """Batch convert images using controlled parallel processing."""
    start_time = time.time()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Gather all files
    supported_formats = {'.jpg', '.jpeg', '.png', '.gif'}
    conversion_tasks = []

    for file_path in Path(input_dir).rglob('*'):
        if file_path.suffix.lower() in supported_formats:
            rel_path = file_path.relative_to(input_dir)
            output_path = Path(output_dir) / rel_path.with_suffix('.webp')
            os.makedirs(output_path.parent, exist_ok=True)
            conversion_tasks.append((str(file_path), str(output_path), quality))

    # Process files in batches with controlled parallelism
    with Pool(processes=num_cores) as pool:
        results = []
        task_iterator = iter(conversion_tasks)

        with tqdm(total=len(conversion_tasks), desc="Converting images") as pbar:
            while True:
                # Get next batch of tasks
                batch = list(islice(task_iterator, BATCH_SIZE))
                if not batch:
                    break

                # Process batch
                batch_results = process_batch(batch, pool)
                results.extend(batch_results)
                pbar.update(len(batch))

    # Count results
    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = len(results) - success_count

    # Print summary
    elapsed_time = time.time() - start_time
    print(f"\nConversion completed in {elapsed_time:.1f} seconds!")
    print(f"Successfully converted: {success_count} files")
    print(f"Failed conversions: {error_count} files")
    print(f"Processing speed: {len(results) / elapsed_time:.1f} images/second")


if __name__ == "__main__":
    input_directory = "./docs/_images"
    output_directory = "webp_output"

    # Example usage with 2 cores
    batch_convert_to_webp(
        input_directory,
        output_directory,
        num_cores=2,  # Specify number of cores to use
        quality=80
    )