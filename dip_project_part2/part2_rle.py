import cv2
import numpy as np

def estimate_spatial_redundancy(img):
    height, width = img.shape
    total_pairs = 0
    same_pairs = 0

    for row in range(height):
        for col in range(1, width):
            total_pairs += 1
            if img[row, col] == img[row, col - 1]:
                same_pairs += 1

    if total_pairs == 0:
        return 0.0

    ratio = same_pairs / total_pairs
    return ratio


def rle_encode(img):
    flat = img.flatten()
    if flat.size == 0:
        return []

    encoded = []
    current_value = int(flat[0])
    run_length = 1
    max_run = 65535

    for pixel in flat[1:]:
        pixel = int(pixel)
        if pixel == current_value and run_length < max_run:
            run_length += 1
        else:
            encoded.append((current_value, run_length))
            current_value = pixel
            run_length = 1

    encoded.append((current_value, run_length))
    return encoded


def compute_compressed_size_bits(encoded_data, bits_per_value=8, bits_per_run=16):
    num_pairs = len(encoded_data)
    return num_pairs * (bits_per_value + bits_per_run)


def compute_metrics(original_bits, compressed_bits):
    if compressed_bits == 0:
        return None, None
    cr = original_bits / compressed_bits
    redundancy = 1 - (1 / cr)
    return cr, redundancy


def main():
    image_path = "input.png"
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Error: could not load image.")
        return

    height, width = img.shape
    print("Image size (pixels):", width, "x", height)

    num_pixels = height * width
    bits_per_pixel = 8
    original_size_bits = num_pixels * bits_per_pixel
    original_size_bytes = original_size_bits // 8
    original_size_kb = original_size_bytes / 1024

    print("Original size (raw):", original_size_bits, "bits")
    print("Original size (raw):", original_size_bytes, "bytes")
    print("Original size (raw):", round(original_size_kb, 2), "KB")

    similarity_ratio = estimate_spatial_redundancy(img)
    print("Estimated spatial redundancy (neighbor similarity):", round(similarity_ratio * 100, 2), "%")

    encoded_data = rle_encode(img)
    print("Number of runs (RLE):", len(encoded_data))

    compressed_size_bits = compute_compressed_size_bits(encoded_data)
    compressed_size_bytes = compressed_size_bits // 8
    compressed_size_kb = compressed_size_bytes / 1024

    print("Compressed size (RLE):", compressed_size_bits, "bits")
    print("Compressed size (RLE):", compressed_size_bytes, "bytes")
    print("Compressed size (RLE):", round(compressed_size_kb, 2), "KB")

    cr, redundancy = compute_metrics(original_size_bits, compressed_size_bits)

    if cr is not None:
        print("Compression Ratio (CR):", round(cr, 3))
        print("Redundancy (R):", round(redundancy * 100, 2), "%")
    else:
        print("Error: compressed size is zero.")


if __name__ == "__main__":
    main()
