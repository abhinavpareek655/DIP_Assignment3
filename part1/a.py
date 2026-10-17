import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((128,128))
    return np.asarray(image, dtype=np.float64)

def feature_representation(image):
    h, w, c = image.shape
    features = np.array([
        [j, i, *image[i, j]]
        for i in range(h)
        for j in range(w)
    ])
    return features

def calculate_weights(current, features, hs, hr):
    x, y = current[:2]
    r, g, b = current[2:]
    spatial_features = features[:, :2]
    rgb_features = features[:, 2:]
    ds = (spatial_features[:, 0]-x)**2 + (spatial_features[:, 1]-y)**2
    dr = (rgb_features[:, 0]-r)**2 + (rgb_features[:, 1]-g)**2 + (rgb_features[:, 2]-b)**2
    ws = np.exp(-ds / (2 * hs**2))
    wr = np.exp(-dr / (2 * hr**2))
    return ws * wr

def mean_shift(image, hs, hr, min_shift=1e-3, max_iter=100):
    features = feature_representation(image)
    new_features = np.zeros_like(features)
    for idx, feature in enumerate(features):
        current = feature.astype(float).copy()
        for _ in range(max_iter):
            weights = calculate_weights(current, features, hs, hr)
            weighted_sum = np.sum(weights[:, np.newaxis] * features,axis=0)
            new_feature = weighted_sum / np.sum(weights)
            shift = np.linalg.norm(new_feature - current)
            current = new_feature
            if shift < min_shift:
                break
        new_features[idx] = current
    return new_features

def create_segmentation(shifted_features, image_shape, spatial_threshold=10, color_threshold=15):
    """Cluster converged mean-shift features and return labels and segment colors."""
    n = shifted_features.shape[0]
    labels = np.zeros(n, dtype=np.int32)
    modes = []
    bins = {}
    spatial_threshold = float(spatial_threshold)
    color_threshold = float(color_threshold)

    for index, feature in enumerate(shifted_features):
        key = tuple(np.floor(feature[:2] / spatial_threshold).astype(int))  
        label = None
        for bx in range(key[0] - 1, key[0] + 2):
            for by in range(key[1] - 1, key[1] + 2):
                for candidate in bins.get((bx, by), []):
                    mode = modes[candidate]
                    if (np.linalg.norm(feature[:2] - mode[:2]) <= spatial_threshold
                            and np.linalg.norm(feature[2:] - mode[2:]) <= color_threshold):
                        label = candidate
                        break
                if label is not None:
                    break
            if label is not None:
                break
        if label is None:
            label = len(modes)
            modes.append(feature.copy())
            bins.setdefault(key, []).append(label)
        labels[index] = label + 1

    labels = labels.reshape(image_shape[:2])
    modes = np.asarray(modes)
    segmented_image = np.clip(
        modes[labels.ravel() - 1, 2:].reshape(image_shape[:2] + (3,)), 0, 255
    ).astype(np.uint8)
    return labels, segmented_image

def create_segment_boundaries(image, labels):
    """Return the RGB image with two-pixel-thick black segment boundaries."""
    result = np.asarray(image, dtype=np.uint8).copy()
    boundaries = np.zeros(labels.shape, dtype=bool)
    boundaries[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundaries[1:, :] |= labels[1:, :] != labels[:-1, :]
    thick = boundaries.copy()
    thick[1:, :] |= boundaries[:-1, :]
    thick[:, 1:] |= boundaries[:, :-1]
    result[thick] = 0
    return result

def save_outputs(image, segmented_image, labels, image_path, out_dir, alpha=0.3):
    os.makedirs(out_dir, exist_ok=True)
    original_u8 = image.astype(np.uint8)
    Image.fromarray(segmented_image).save(os.path.join(out_dir, "segmented.png"))
    Image.fromarray(original_u8).save(os.path.join(out_dir, "original.png"))
    np.save(os.path.join(out_dir, "labels.npy"), labels)
    overlay = (alpha * segmented_image + (1 - alpha) * original_u8).astype(np.uint8)
    Image.fromarray(overlay).save(os.path.join(out_dir, "overlay.png"))
    print(f"Saved outputs to '{out_dir}/'")
    return overlay

def main(image_path):
    image = load_image(image_path)
    hs = 0.2*min(image.shape[:2])
    hr = 40
    shifted_features = mean_shift(image, hs, hr)

    labels, segmented_image = create_segmentation(
        shifted_features,
        image.shape,
        spatial_threshold=10,
        color_threshold=15
    )
    boundaries = create_segment_boundaries(image, labels)
    output_dir = "outputs/part1/" + image_path.split('/')[-1].split('.')[0]
    save_outputs(image, segmented_image, labels, image_path, output_dir)
    segmented_gray = np.mean(segmented_image, axis=2)
    binary_segmentation = ((segmented_gray > segmented_gray.mean()) * 255).astype(np.uint8)
    Image.fromarray(binary_segmentation).save(
        os.path.join(output_dir, "binary_segmentation.png")
    )
    Image.fromarray(boundaries).save(os.path.join(output_dir, "boundaries.png"))
    print("Number of segments:", len(np.unique(labels)))

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image.astype(np.uint8))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(binary_segmentation, cmap="gray", vmin=0, vmax=255)
    plt.title("Binary Segmentation")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "input_and_segmentation.png"),
                bbox_inches="tight", dpi=150)
    plt.show()

    plt.figure(figsize=(5, 5))
    plt.imshow(boundaries)
    plt.title("Segment Boundaries on Input")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "boundaries_display.png"),
                bbox_inches="tight", dpi=150)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python a.py <image_path>")
        sys.exit(1)
    main(sys.argv[1])