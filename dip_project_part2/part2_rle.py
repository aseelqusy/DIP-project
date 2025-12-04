import cv2
import numpy as np


def estimate_spatial_redundancy(img):
    """
    تحسب نسبة البكسلات المتجاورة (يمين-يسار) اللي إلها نفس القيمة
    عشان نقدر نحكي في عندنا spatial redundancy أو لا.
    """
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
    """
    تطبق Run-Length Encoding على الصورة
    وترجع list من أزواج (value, run_length)
    """
    flat = img.flatten()
    if flat.size == 0:
        return []

    encoded = []
    current_value = int(flat[0])
    run_length = 1

    # نسمح بحد أقصى لطول الـ run عشان نضمنه ضمن 16 بت
    max_run = 65535

    for pixel in flat[1:]:
        pixel = int(pixel)
        if pixel == current_value and run_length < max_run:
            run_length += 1
        else:
            # نخزن الزوج الحالي
            encoded.append((current_value, run_length))
            current_value = pixel
            run_length = 1

    # آخر run
    encoded.append((current_value, run_length))
    return encoded


def compute_compressed_size_bits(encoded_data, bits_per_value=8, bits_per_run=16):
    """
    نحسب الحجم النظري للبيانات المضغوطة بالبت
    كل زوج (value, run) بياخد bits_per_value + bits_per_run
    """
    num_pairs = len(encoded_data)
    return num_pairs * (bits_per_value + bits_per_run)


def compute_metrics(original_bits, compressed_bits):
    """
    تحسب Compression Ratio و Redundancy
    CR = original / compressed
    R = 1 - 1/CR
    """
    if compressed_bits == 0:
        return None, None

    cr = original_bits / compressed_bits
    redundancy = 1 - (1 / cr)
    return cr, redundancy


def main():
    # 1) قراءة الصورة كـ grayscale
    image_path = "input.png"  # غيّري الاسم لو صورتك اسمها مختلف
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Error: could not load image. تأكدي من اسم الملف ومكانه.")
        return

    height, width = img.shape
    print("Image size (pixels):", width, "x", height)

    # 2) الحجم الأصلي (raw) بالتقريب: M * N * 8 bits
    num_pixels = height * width
    bits_per_pixel = 8
    original_size_bits = num_pixels * bits_per_pixel
    original_size_bytes = original_size_bits // 8
    original_size_kb = original_size_bytes / 1024

    print("Original size (raw):", original_size_bits, "bits")
    print("Original size (raw):", original_size_bytes, "bytes")
    print("Original size (raw):", round(original_size_kb, 2), "KB")

    # 3) تقدير spatial redundancy
    similarity_ratio = estimate_spatial_redundancy(img)
    print("Estimated spatial redundancy (neighbor similarity):",
          round(similarity_ratio * 100, 2), "%")

    # 4) RLE Encoding
    encoded_data = rle_encode(img)
    print("Number of runs (RLE):", len(encoded_data))

    # 5) الحجم بعد الضغط
    compressed_size_bits = compute_compressed_size_bits(encoded_data)
    compressed_size_bytes = compressed_size_bits // 8
    compressed_size_kb = compressed_size_bytes / 1024

    print("Compressed size (RLE):", compressed_size_bits, "bits")
    print("Compressed size (RLE):", compressed_size_bytes, "bytes")
    print("Compressed size (RLE):", round(compressed_size_kb, 2), "KB")

    # 6) حساب CR و Redundancy
    cr, redundancy = compute_metrics(original_size_bits, compressed_size_bits)

    if cr is not None:
        print("Compression Ratio (CR):", round(cr, 3))
        print("Redundancy (R):", round(redundancy * 100, 2), "%")
    else:
        print("Error: compressed size is zero.")


if __name__ == "__main__":
    main()
