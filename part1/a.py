import sys

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
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
    h, w, c = image_shape
    n = len(shifted_features)
    labels = np.full(n, -1, dtype=int)
    modes = []
    for i in range(n):
        point = shifted_features[i].astype(float)
        assigned = False

        for label, mode in enumerate(modes):
            spatial_distance = np.linalg.norm(point[:2] - mode[:2])
            color_distance = np.linalg.norm(point[2:] - mode[2:])

            if (spatial_distance <= spatial_threshold and
                    color_distance <= color_threshold):
                labels[i] = label
                assigned = True
                break

        if not assigned:
            labels[i] = len(modes)
            modes.append(point.copy())

    label_map = labels.reshape(h, w)

    rng = np.random.default_rng(42)
    num_segments = len(modes)
    colors = rng.integers(
        0, 256, size=(num_segments, 3), dtype=np.uint8
    )
    segmented_image = colors[label_map]

    return label_map, segmented_image

def main(image_path):
    image = load_image(image_path)
    hs = 10
    hr = 15
    shifted_features = mean_shift(image, hs, hr)

    labels, segmented_image = create_segmentation(
        shifted_features,
        image.shape,
        spatial_threshold=10,
        color_threshold=15
    )

    print("Number of segments:", len(np.unique(labels)))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image.astype(np.uint8))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image)
    plt.title("Mean-Shift Segmentation")
    plt.axis("off")

    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python a.py <image_path>")
        sys.exit(1)
    main(sys.argv[1])