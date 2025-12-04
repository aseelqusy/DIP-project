import cv2
import numpy as np

def classify_redundancies(coding_red, spatial_red, spectral_strength):
    # First: determine LOW/MEDIUM/HIGH (for printing)
    levels = {}

    levels["coding"] = "HIGH" if coding_red >= 0.5 else ("MEDIUM" if coding_red >= 0.3 else "LOW")
    levels["spatial"] = "HIGH" if spatial_red >= 0.7 else ("MEDIUM" if spatial_red >= 0.3 else "LOW")
    levels["spectral"] = "HIGH" if spectral_strength >= 0.7 else ("MEDIUM" if spectral_strength >= 0.3 else "LOW")

    # Second: choose actual dominant redundancy (slides logic)
    if spatial_red >= 0.7:
        dominant = "spatial"
    elif coding_red >= 0.5:
        dominant = "coding"
    elif spectral_strength >= 0.7:
        dominant = "spectral"
    else:
        dominant = "none"

    return levels, dominant


def estimate_spatial_redundancy(img_gray):
    height, width = img_gray.shape
    total_pairs = 0
    same_pairs = 0

    for row in range(height):
        for col in range(1, width):
            total_pairs += 1
            if img_gray[row, col] == img_gray[row, col - 1]:
                same_pairs += 1

    if total_pairs == 0:
        return 0.0

    return same_pairs / total_pairs


def analyze_coding_redundancy(img_gray):
    flat = img_gray.flatten()
    num_pixels = flat.size

    if num_pixels == 0:
        return 0.0, 0.0

    hist = np.bincount(flat, minlength=256)
    probs = hist / num_pixels
    non_zero_probs = probs[probs > 0]
    entropy = -np.sum(non_zero_probs * np.log2(non_zero_probs))

    coding_redundancy = 1.0 - (entropy / 8.0)
    return entropy, coding_redundancy


def analyze_spectral_redundancy(img_color):
    if img_color is None or img_color.ndim != 3 or img_color.shape[2] != 3:
        return None, None, None, None

    b, g, r = cv2.split(img_color)
    b = b.flatten().astype(np.float32)
    g = g.flatten().astype(np.float32)
    r = r.flatten().astype(np.float32)

    corr_bg = float(np.corrcoef(b, g)[0, 1])
    corr_gr = float(np.corrcoef(g, r)[0, 1])
    corr_br = float(np.corrcoef(b, r)[0, 1])

    avg_corr = (abs(corr_bg) + abs(corr_gr) + abs(corr_br)) / 3.0
    return corr_bg, corr_gr, corr_br, avg_corr


def choose_compression_method(coding_red, spatial_red, spectral_strength):
    if spatial_red is not None and spatial_red >= 0.7:
        return "RLE", f"High spatial redundancy ({spatial_red * 100:.2f}%)"
    if spectral_strength is not None and spectral_strength >= 0.9:
        return "Color transform / spectral compression", f"High spectral redundancy (avg corr ≈ {spectral_strength:.3f})"
    if coding_red is not None and coding_red >= 0.5:
        return "Entropy coding (e.g., Huffman)", f"High coding redundancy ({coding_red * 100:.2f}%)"
    return "No specific compression advantage", "Redundancy levels are relatively low"


def rle_encode(img_gray):
    flat = img_gray.flatten()
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
    return len(encoded_data) * (bits_per_value + bits_per_run)


def compute_metrics(original_bits, compressed_bits):
    if compressed_bits == 0:
        return None, None
    cr = original_bits / compressed_bits
    redundancy = 1 - (1 / cr)
    return cr, redundancy


def main():
    image_path = "L_Spec_Red.png"

    img_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_color is None:
        print("Error: could not load image.")
        return

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    height, width = img_gray.shape
    print("Image size (pixels):", width, "x", height)

    num_pixels = height * width
    original_size_bits = num_pixels * 8
    original_size_bytes = original_size_bits // 8
    original_size_kb = original_size_bytes / 1024

    print("Original size (raw, grayscale):", original_size_bits, "bits")
    print("Original size (raw, grayscale):", original_size_bytes, "bytes")
    print("Original size (raw, grayscale):", round(original_size_kb, 2), "KB")

    entropy, coding_red = analyze_coding_redundancy(img_gray)
    coding_red = max(0.0, min(1.0, coding_red))
    print(f"Entropy (H): {entropy:.4f} bits/pixel")
    print(f"Estimated coding redundancy: {coding_red * 100:.2f} %")

    similarity_ratio = estimate_spatial_redundancy(img_gray)
    print("Estimated spatial redundancy (neighbor similarity):", round(similarity_ratio * 100, 2), "%")

    corr_bg, corr_gr, corr_br, avg_corr = analyze_spectral_redundancy(img_color)
    if avg_corr is not None:
        print(f"Spectral redundancy correlations (B-G, G-R, B-R): {corr_bg:.3f}, {corr_gr:.3f}, {corr_br:.3f}")
        print(f"Average spectral redundancy (|corr|): {avg_corr:.3f}")
    else:
        print("Spectral redundancy: not applicable (grayscale image).")

    levels, dominant = classify_redundancies(coding_red, similarity_ratio, avg_corr)
    print("Redundancy levels:")
    print("  Coding :", levels["coding"])
    print("  Spatial:", levels["spatial"])
    print("  Spectral:", levels["spectral"])
    print("Dominant redundancy type:", dominant)

    method, reason = choose_compression_method(coding_red, similarity_ratio, avg_corr)
    print("Chosen compression method:", method)
    print("Reason:", reason)

    encoded_data = rle_encode(img_gray)
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
