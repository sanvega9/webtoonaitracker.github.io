# Manual image processing pipeline for diabetic retinopathy detection
import numpy as np
import matplotlib.pyplot as plt
import imageio
import cv2

# --- Utility Functions ---
def load_and_resize_image(path, size=(512, 512)):
    img = imageio.v2.imread(path)
    img = img[:size[0], :size[1], :3]  # Crop to size
    return img

def rgb2gray(image):
    return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

def normalize(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)

# --- Manual CLAHE ---
def manual_clahe(img, clip_limit=0.01, tile_grid=(8, 8)):
    img = (img * 255).astype(np.uint8)
    h, w = img.shape
    tile_h, tile_w = h // tile_grid[0], w // tile_grid[1]
    result = np.zeros_like(img, dtype=np.float32)

    for i in range(tile_grid[0]):
        for j in range(tile_grid[1]):
            x0, y0 = i * tile_h, j * tile_w
            x1, y1 = min((i + 1) * tile_h, h), min((j + 1) * tile_w, w)
            tile = img[x0:x1, y0:y1]
            hist, _ = np.histogram(tile.flatten(), bins=256, range=(0, 255))
            cdf = np.cumsum(hist)
            clip_value = clip_limit * tile.size
            cdf = np.clip(cdf, 0, clip_value)
            cdf = np.cumsum(cdf)
            cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min() + 1e-8) * 255
            cdf = cdf.astype(np.uint8)
            result[x0:x1, y0:y1] = cdf[tile]

    return normalize(result)

# --- Manual Morphological Filters ---
def tophat_filter(img, kernel_size=15):
    pad = kernel_size // 2
    padded = np.pad(img, pad, mode='reflect')
    result = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i + kernel_size, j:j + kernel_size]
            min_val = np.min(region)
            result[i, j] = img[i, j] - min_val
    return normalize(result)

def blackhat_filter(img, kernel_size=15):
    pad = kernel_size // 2
    padded = np.pad(img, pad, mode='reflect')
    result = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i + kernel_size, j:j + kernel_size]
            max_val = np.max(region)
            result[i, j] = max_val - img[i, j]
    return normalize(result)

# --- Region Detection (manual BFS-based blob detection) ---
def detect_regions(image, threshold=0.6, min_area=5, max_area=100):
    binary = image > threshold
    visited = np.zeros_like(binary, dtype=bool)
    boxes = []

    def bfs(x, y):
        queue = [(x, y)]
        coords = []
        while queue:
            cx, cy = queue.pop(0)
            if 0 <= cx < image.shape[0] and 0 <= cy < image.shape[1]:
                if binary[cx, cy] and not visited[cx, cy]:
                    visited[cx, cy] = True
                    coords.append((cx, cy))
                    queue.extend([(cx + dx, cy + dy) for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]])
        return coords

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if binary[i, j] and not visited[i, j]:
                region = bfs(i, j)
                if min_area <= len(region) <= max_area:
                    xs, ys = zip(*region)
                    boxes.append((min(xs), min(ys), max(xs), max(ys)))
    return boxes

# --- Draw Bounding Boxes ---
def draw_boxes(image, boxes, color=(255, 0, 0)):
    overlay = image.copy()
    for x1, y1, x2, y2 in boxes:
        overlay[x1:x2 + 1, y1] = color
        overlay[x1:x2 + 1, y2] = color
        overlay[x1, y1:y2 + 1] = color
        overlay[x2, y1:y2 + 1] = color
    return overlay

# --- Main Execution ---
if __name__ == "__main__":
    path = "archive/retino/train/DR/1a7e3356b39c_png.rf.bd52d78b9d4b1ab6551912357977fcbd.jpg"
    image = load_and_resize_image(path)
    gray = rgb2gray(image)
    gray_norm = normalize(gray)

    clahe_img = manual_clahe(gray_norm, clip_limit=0.02)
    tophat_img = tophat_filter(clahe_img, kernel_size=15)
    blackhat_img = blackhat_filter(clahe_img, kernel_size=15)

    bright_boxes = detect_regions(tophat_img, threshold=0.7, min_area=8, max_area=150)
    dark_boxes = detect_regions(blackhat_img, threshold=0.5, min_area=8, max_area=150)

    img_with_bright = draw_boxes(image.copy(), bright_boxes, color=(255, 255, 0))
    img_with_dark = draw_boxes(img_with_bright, dark_boxes, color=(0, 255, 255))

    # Visualization
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1); plt.imshow(image); plt.title("Original"); plt.axis('off')
    plt.subplot(2, 3, 2); plt.imshow(clahe_img, cmap='gray'); plt.title("Manual CLAHE"); plt.axis('off')
    plt.subplot(2, 3, 3); plt.imshow(tophat_img, cmap='gray'); plt.title("Top-Hat"); plt.axis('off')
    plt.subplot(2, 3, 4); plt.imshow(blackhat_img, cmap='gray'); plt.title("Black-Hat"); plt.axis('off')
    plt.subplot(2, 3, 5); plt.imshow(img_with_bright); plt.title("Bright Lesions"); plt.axis('off')
    plt.subplot(2, 3, 6); plt.imshow(img_with_dark); plt.title("Final Detection"); plt.axis('off')
    plt.tight_layout()
    plt.show()


# Load the retinal image
image = cv2.imread('archive/retino/valid/DR/f6f433f3306f_png.rf.379af08f32752a80302e616e10c0b700.jpg')  # Replace with your image file path


# Load the retinal image
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 1: Preprocessing (Convert to Green Channel)
green_channel = image_rgb[:, :, 1]

# Step 2: Adaptive Thresholding (to isolate blood vessels, exudates, etc.)
# Using Gaussian adaptive thresholding
adaptive_thresh = cv2.adaptiveThreshold(green_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

# Step 3: Edge Detection (Canny Edge Detection)
edges = cv2.Canny(green_channel, 100, 200)

# Step 4: Watershed Segmentation (for Optic Disc Segmentation)
# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a Gaussian blur to smooth out the image
blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Perform edge detection using Canny (to help with watershed initialization)
edges_for_watershed = cv2.Canny(blurred, 100, 200)

# Find contours to initialize the markers for watershed
contours, _ = cv2.findContours(edges_for_watershed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a marker image (initialize with zeros)
markers = np.zeros_like(gray_image, dtype=np.int32)  # Ensure dtype is CV_32SC1

# Draw the contours (markers for watershed)
cv2.drawContours(markers, contours, -1, (255), 1)  # Set contours as markers

# Mark the region of interest (optional)
markers[markers == 0] = -1  # Set the background marker to -1 for watershed
markers[edges_for_watershed == 255] = 1  # Set edge region markers to 1

# Convert to 8-bit for visualization (as OpenCV requires 8-bit for displaying)
image_display = image.copy()

# Apply watershed algorithm
cv2.watershed(image_display, markers)

# Mark the boundaries found by watershed
image_display[markers == -1] = [255, 0, 0]  # Red boundary for watershed result

# Step 5: Displaying Results
# Create a plot to visualize the results
plt.figure(figsize=(12, 8))

plt.subplot(231)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis('off')

plt.subplot(232)
plt.imshow(adaptive_thresh, cmap='gray')
plt.title("Adaptive Thresholding")
plt.axis('off')

plt.subplot(233)
plt.imshow(edges, cmap='gray')
plt.title("Canny Edge Detection")
plt.axis('off')

plt.subplot(234)
plt.imshow(image_display)
plt.title("Watershed Segmentation")
plt.axis('off')

plt.show()


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the uploaded fundus image
img_path = "archive/Diagnosis of Diabetic Retinopathy/valid/DR/46d3316c4857_png.rf.97514151b3d9982f4026534206390af6.jpg"
original = cv2.imread(img_path)
original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_gray = clahe.apply(gray)

# Microaneurysms
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 5
params.maxArea = 100
params.filterByCircularity = True
params.minCircularity = 0.5
params.filterByConvexity = False
params.filterByInertia = False
params.blobColor = 255
detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(clahe_gray)

micro_mask = np.zeros_like(gray)
for kp in keypoints:
    x, y = int(kp.pt[0]), int(kp.pt[1])
    cv2.circle(micro_mask, (x, y), int(kp.size / 2), 255, -1)

# --- Better Exudate Detection using HSV ---
hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
lower_yellow = np.array([20, 60, 150])  # tuned for yellowish exudates
upper_yellow = np.array([40, 255, 255])
exudate_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
exudate_mask = cv2.medianBlur(exudate_mask, 5)

# Optional: remove small blobs
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
exudate_mask = cv2.morphologyEx(exudate_mask, cv2.MORPH_OPEN, kernel, iterations=1)

# --- Better Hemorrhage Detection using LAB ---
lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
lower_hemo = np.array([20, 130, 120])   # tuned for dark red lesions
upper_hemo = np.array([90, 170, 150])
hemo_mask = cv2.inRange(lab, lower_hemo, upper_hemo)
hemo_mask = cv2.medianBlur(hemo_mask, 5)
hemo_mask = cv2.morphologyEx(hemo_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

# Overlay
overlay = original_rgb.copy()
overlay[hemo_mask > 0] = [255, 0, 0]     # Red: Hemorrhages
overlay[micro_mask > 0] = [0, 255, 0]    # Green: Microaneurysms
overlay[exudate_mask > 0] = [255, 255, 0]# Yellow: Exudates

# Plotting
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
titles = ["CLAHE Preprocessed", "Microaneurysms", "Exudates", "Hemorrhages", "Overlay"]
images = [clahe_gray, micro_mask, exudate_mask, hemo_mask, overlay]

for ax, title, img in zip(axes, titles, images):
    ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.show()



import numpy as np
import tkinter as tk
from tkinter import filedialog
import imageio
import cv2
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from skimage import filters, morphology, exposure,feature,color
from skimage.filters import frangi
from skimage.morphology import white_tophat, disk
from skimage.util import img_as_ubyte
from skimage.morphology import skeletonize
import deep_learning_DR 

# ----------------- Image Processing Utilities -----------------

def rgb2gray(image):
    return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])

def normalize(image):
    return (image - image.min()) / (image.max() - image.min())

def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def vessel_segmentation(img):
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    frangi_img = frangi(blurred / 255.0)
    frangi_ubyte = img_as_ubyte(frangi_img)
    thresholded = cv2.adaptiveThreshold(frangi_ubyte, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, 15, -2)
    skeleton = skeletonize(thresholded > 0)
    return img_as_ubyte(skeleton)

def detect_microaneurysms(img):
    tophat = white_tophat(img, disk(8))
    blur = cv2.GaussianBlur(tophat, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    return cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
def detect_exudates(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # Use morphological operations to suppress vessels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    closed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)

    # Subtract enhanced image from the closed image to highlight bright lesions
    exudates = cv2.subtract(closed, enhanced)

    # Threshold to get binary mask
    _, mask = cv2.threshold(exudates, 30, 255, cv2.THRESH_BINARY)

    return mask

def show_fft(gray_img):
    f = np.fft.fft2(gray_img)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1e-8)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(gray_img, cmap='gray')
    plt.title("Grayscale Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(magnitude, cmap='gray')
    plt.title("FFT Magnitude Spectrum")
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    return magnitude


def manual_clahe_optimized(img, clip_limit=2.0, tile_grid_size=(8,8)):
    if len(img.shape) != 2:
        raise ValueError("Input must be a grayscale image.")
    h, w = img.shape
    n_tiles_y, n_tiles_x = tile_grid_size
    tile_h = h // n_tiles_y
    tile_w = w // n_tiles_x
    n_bins = 256
    tile_size = tile_h * tile_w
    clip_limit_pixels = int((clip_limit * tile_size) / n_bins)
    clip_limit_pixels = max(clip_limit_pixels, 1)

    lut_tiles = np.zeros((n_tiles_y, n_tiles_x, n_bins), dtype=np.float32)

    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            y0 = ty * tile_h
            x0 = tx * tile_w
            y1 = y0 + tile_h if (ty < n_tiles_y - 1) else h
            x1 = x0 + tile_w if (tx < n_tiles_x - 1) else w
            tile = img[y0:y1, x0:x1]
            hist = np.bincount(tile.ravel(), minlength=n_bins).astype(np.float32)
            excess = hist - clip_limit_pixels
            excess = np.maximum(excess, 0)
            excess_total = np.sum(excess)
            hist = np.minimum(hist, clip_limit_pixels)
            hist += excess_total / n_bins
            hist = hist / np.sum(hist)
            cdf = np.cumsum(hist)
            cdf = np.clip(cdf, 0, 1)
            lut = (cdf * 255).astype(np.uint8)
            lut_tiles[ty, tx] = lut

    out_img = np.zeros_like(img)
    grid_y = np.linspace(0, h, n_tiles_y + 1, dtype=int)
    grid_x = np.linspace(0, w, n_tiles_x + 1, dtype=int)

    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            y0, y1 = grid_y[ty], grid_y[ty+1]
            x0, x1 = grid_x[tx], grid_x[tx+1]
            y_indices = np.linspace(0, 1, y1 - y0, endpoint=False)[:, None]
            x_indices = np.linspace(0, 1, x1 - x0, endpoint=False)[None, :]
            lut_tl = lut_tiles[ty, tx]
            lut_tr = lut_tiles[ty, tx+1] if tx < n_tiles_x-1 else lut_tiles[ty, tx]
            lut_bl = lut_tiles[ty+1, tx] if ty < n_tiles_y-1 else lut_tiles[ty, tx]
            lut_br = lut_tiles[ty+1, tx+1] if (tx < n_tiles_x-1 and ty < n_tiles_y-1) else lut_tiles[ty, tx]
            tile_region = img[y0:y1, x0:x1]
            mapped_tl = lut_tl[tile_region]
            mapped_tr = lut_tr[tile_region]
            mapped_bl = lut_bl[tile_region]
            mapped_br = lut_br[tile_region]
            top = mapped_tl * (1 - x_indices) + mapped_tr * x_indices
            bottom = mapped_bl * (1 - x_indices) + mapped_br * x_indices
            interpolated = top * (1 - y_indices) + bottom * y_indices
            out_img[y0:y1, x0:x1] = interpolated.astype(np.uint8)

    return out_img

def manual_clahe_color(img, clip_limit=2.0, tile_grid_size=(8,8)):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_clahe = manual_clahe_optimized(l, clip_limit, tile_grid_size)
    lab_clahe = cv2.merge((l_clahe, a, b))
    img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    return img_clahe

def opencv_clahe_color(img, clip_limit=2.0, tile_grid_size=(8,8)):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    return img_clahe

def tophat_filter(img, kernel_size=15):
    pad = kernel_size // 2
    padded = np.pad(img, pad, mode='reflect')
    result = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i+kernel_size, j:j+kernel_size]
            min_val = np.min(region)
            result[i, j] = img[i, j] - min_val
    return normalize(result)

def blackhat_filter(img, kernel_size=15):
    pad = kernel_size // 2
    padded = np.pad(img, pad, mode='reflect')
    result = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i+kernel_size, j:j+kernel_size]
            max_val = np.max(region)
            result[i, j] = max_val - img[i, j]
    return normalize(result)

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

# ----------------- GUI Class -----------------

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Manual Image Processing App")
        self.original_image = None
        self.processed_image = None

        # Buttons
        button_texts = [
            ("Load Image", self.load_image),
            ("Manual CLAHE (Color)", self.apply_manual_clahe),
            ("OpenCV CLAHE (Color)", self.apply_opencv_clahe),
            ("Top-Hat Transform ", self.apply_tophat),
            ("Black-Hat Dark Lesions", self.apply_blackhat),
            ("Grayscale", self.to_grayscale),
            ("Manual Gaussian Blur", self.gaussian_blur),
            ("Histogram Equalization", self.histogram_equalization),
            ("Edge Detection", self.sobel_edge_detection),
            ("Threshold", self.apply_threshold),
            ("Show Red Channel", self.show_red_channel),
            ("Show Green Channel", self.show_green_channel),
            ("Show Blue Channel", self.show_blue_channel),
            ("Gray Red Channel", self.gray_red_channel),
            ("Gray Green Channel", self.gray_green_channel),
            ("Gray Blue Channel", self.gray_blue_channel),
            ("Compare MSE", self.compare_mse),
            ("Show Histogram", self.show_histogram),
            ("CLAHE + Vessel Segmentation", self.process_vessels),     
            ("Detect Microaneurysms",self.process_microaneurysms),
            ("Detect Exudates", self.process_exudates), 
            ("Deep Learning",self.image_detection_button) 
            ]

        for text, cmd in button_texts:
            tk.Button(root, text=text, command=cmd).pack()

        self.threshold_value = tk.IntVar(value=128)
        self.threshold_slider = tk.Scale(root, from_=0, to=255, orient="horizontal",
                                         variable=self.threshold_value, label="Adjust Threshold",
                                         command=self.update_threshold)
        self.threshold_slider.pack()
        self.threshold_slider.pack_forget()

        self.orig_label = tk.Label(root)
        self.orig_label.pack(side="left", padx=10, pady=10)
        self.proc_label = tk.Label(root)
        self.proc_label.pack(side="right", padx=10, pady=10)

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            img = imageio.v2.imread(file_path)
            img = img[:512, :512, :]
            self.original_image = img
            self.processed_image = img
            self.display_image(self.original_image, self.orig_label)


    def display_image(self, array, label):
        if isinstance(array, Image.Image):
            img = np.array(array)
        else:
            img = array
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        image = Image.fromarray(img)
        image = image.resize((400, 400))
        image = ImageTk.PhotoImage(image)
        label.configure(image=image)
        label.image = image
        
    def process_vessels(self):
        if self.original_image is None:
            return

        # 1) convert original RGB to 8‑bit gray
        gray = self.to_gray_array()               # shape (H, W), uint8

        # 2) equalize contrast on that gray
        clahe_img = apply_clahe(gray)             # still (H, W), uint8

        # 3) do your Frangi + adaptive threshold + skel
        skeleton = vessel_segmentation(clahe_img)  # (H, W), uint8

        # 4) store for potential MSE, etc.
        self.processed_image = skeleton

        # 5) expand to 3‑channel so PIL/Tkinter can show it correctly
        gray_rgb = np.stack([gray]*3, axis=-1)
        skel_rgb = np.stack([skeleton]*3, axis=-1)

        # 6) display side‑by‑side
        self.display_image(gray_rgb, self.orig_label)
        self.display_image(skel_rgb, self.proc_label)
    def process_fft(self):
        if self.original_image is None:
            return

        # Convert to grayscale for FFT
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

        # Apply FFT
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)  # +1 to avoid log(0)

        # Normalize and convert to displayable image
        mag_img = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        fft_img = cv2.cvtColor(mag_img, cv2.COLOR_GRAY2BGR)

        self.display_image(fft_img, self.proc_label)


    def process_microaneurysms(self):
        if self.original_image is not None:
            gray = self.to_gray_array()
            mask = detect_microaneurysms(gray)
            self.processed_image = np.stack([mask]*3, axis=-1)
            self.display_image(self.processed_image, self.proc_label)
                
    def process_exudates(self):
        if self.original_image is None:
            return

        img = self.original_image.copy()
        green_channel = img[:, :, 1]  # Extract green channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_green = clahe.apply(green_channel)

        # Top-hat to extract bright lesions (exudates)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        tophat = cv2.morphologyEx(enhanced_green, cv2.MORPH_TOPHAT, kernel)

        # Threshold to isolate bright regions
        _, exudate_mask = cv2.threshold(tophat, 30, 255, cv2.THRESH_BINARY)

        # Remove small noise and thin vessels
        cleaned_mask = cv2.morphologyEx(exudate_mask, cv2.MORPH_OPEN,
                                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        cleaned_mask = cv2.dilate(cleaned_mask,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)

        # Filter by area (remove small components)
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_mask = np.zeros_like(cleaned_mask)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 50:  # Keep only large enough areas
                cv2.drawContours(final_mask, [cnt], -1, 255, -1)

        # Highlight exudates in green
        exudate_overlay = img.copy()
        exudate_overlay[final_mask == 255] = [0, 255, 0]
        output = cv2.addWeighted(img, 0.7, exudate_overlay, 0.3, 0)

        self.display_image(output, self.proc_label)

    def to_gray_array(self):
        r, g, b = self.original_image[:,:,0], self.original_image[:,:,1], self.original_image[:,:,2]
        gray = (0.299 * r + 0.587 * g + 0.114 * b)
        return gray.astype(np.uint8)

    def to_grayscale(self):
        gray = self.to_gray_array()
        self.processed_image = np.stack([gray]*3, axis=-1)
        self.display_image(self.processed_image, self.proc_label)

    def gaussian_kernel(self, size=5, sigma=1.0):
        k = size // 2
        x, y = np.mgrid[-k:k+1, -k:k+1]
        kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        return kernel / kernel.sum()

    def apply_filter(self, image, kernel):
        pad = kernel.shape[0] // 2
        padded = np.pad(image, pad, mode='reflect')
        result = np.zeros_like(image)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded[i:i+kernel.shape[0], j:j+kernel.shape[1]]
                result[i, j] = np.sum(region * kernel)
        return result

    def gaussian_blur(self):
        gray = self.to_gray_array()
        kernel = self.gaussian_kernel(5, 1)
        blurred = self.apply_filter(gray, kernel).astype(np.uint8)
        self.processed_image = np.stack([blurred]*3, axis=-1)
        self.display_image(self.processed_image, self.proc_label)

    def histogram_equalization(self):
        gray = self.to_gray_array()
        hist, _ = np.histogram(gray.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_min = cdf[cdf > 0].min()
        cdf_normalized = (cdf - cdf_min) * 255 / (cdf.max() - cdf_min)
        cdf_normalized = np.clip(cdf_normalized, 0, 255).astype(np.uint8)
        equalized = cdf_normalized[gray]
        self.processed_image = np.stack([equalized]*3, axis=-1)
        self.display_image(self.processed_image, self.proc_label)

    def show_red_channel(self):
        if self.original_image is not None:
            red = self.original_image.copy()
            red[:,:,1] = 0
            red[:,:,2] = 0
            self.processed_image = red
            self.display_image(red, self.proc_label)

    def show_green_channel(self):
        if self.original_image is not None:
            green = self.original_image.copy()
            green[:,:,0] = 0
            green[:,:,2] = 0
            self.processed_image = green
            self.display_image(green, self.proc_label)

    def show_blue_channel(self):
        if self.original_image is not None:
            blue = self.original_image.copy()
            blue[:,:,0] = 0
            blue[:,:,1] = 0
            self.processed_image = blue
            self.display_image(blue, self.proc_label)

    def apply_manual_clahe(self):
        if self.original_image is not None:
            processed = manual_clahe_color(self.original_image)
            self.processed_image = processed
            self.display_image(processed, self.proc_label)

    def apply_opencv_clahe(self):
        if self.original_image is not None:
            processed = opencv_clahe_color(self.original_image)
            self.processed_image = processed
            self.display_image(processed, self.proc_label)

    def apply_tophat(self):
        if self.original_image is not None:
            gray = rgb2gray(self.original_image)
            gray_norm = normalize(gray)
            tophat_img = tophat_filter(gray_norm, kernel_size=15)
            self.processed_image = (tophat_img * 255).astype(np.uint8)
            self.processed_image = np.stack([self.processed_image]*3, axis=-1)
            self.display_image(self.processed_image, self.proc_label)

    def apply_blackhat(self):
        if self.original_image is not None:
            gray = rgb2gray(self.original_image)
            gray_norm = normalize(gray)
            blackhat_img = blackhat_filter(gray_norm, kernel_size=15)
            self.processed_image = (blackhat_img * 255).astype(np.uint8)
            self.processed_image = np.stack([self.processed_image]*3, axis=-1)
            self.display_image(self.processed_image, self.proc_label)

    def sobel_edge_detection(self):
        gray = self.to_gray_array()
        dx = filters.sobel_h(gray)
        dy = filters.sobel_v(gray)
        edges = np.hypot(dx, dy)
        edges = (normalize(edges) * 255).astype(np.uint8)
        self.processed_image = np.stack([edges]*3, axis=-1)
        self.display_image(self.processed_image, self.proc_label)

    def apply_threshold(self):
        gray = self.to_gray_array()
        thresh = self.threshold_value.get()
        binary = (gray > thresh) * 255
        binary = binary.astype(np.uint8)
        self.processed_image = np.stack([binary]*3, axis=-1)
        self.display_image(self.processed_image, self.proc_label)
        self.threshold_slider.pack()

    def update_threshold(self, val):
        self.apply_threshold()

    def gray_red_channel(self):
        red = self.original_image[:,:,0]
        gray = np.stack((red,)*3, axis=-1)
        self.processed_image = gray
        self.display_image(gray, self.proc_label)

    def gray_green_channel(self):
        green = self.original_image[:,:,1]
        gray = np.stack((green,)*3, axis=-1)
        self.processed_image = gray
        self.display_image(gray, self.proc_label)

    def gray_blue_channel(self):
        blue = self.original_image[:,:,2]
        gray = np.stack((blue,)*3, axis=-1)
        self.processed_image = gray
        self.display_image(gray, self.proc_label)

    def compare_mse(self):
        if self.original_image is not None and self.processed_image is not None:
            gray_orig = rgb2gray(self.original_image)
            gray_proc = rgb2gray(self.processed_image)
            error = mse(gray_orig, gray_proc)
            print(f"MSE: {error:.2f}")

    def show_histogram(self):
        if self.original_image is not None:
            gray = self.to_gray_array()
            plt.hist(gray.flatten(), bins=256, range=[0,256], color='blue', alpha=0.7)
            plt.title('Histogram')
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')
            plt.show()
    def image_detection_button(self):
        deep_learning_DR.run_gui(parent=root)
    # root = tk.Tk()
    # root.mainloop()
                
    # def overlay_and_save(self):
    #     if self.original_image is None or self.processed_image is None:
    #         clahe_img = apply_clahe(self.processed_image)
    #         vessel_mask = vessel_segmentation(clahe_img)
    #         micro_mask = detect_microaneurysms(self.processed_image)
    #         exudate_mask = detect_exudates(self.original_image)
    #         overlay_img = self.original_image.copy()
    #         overlay_img[vessel_mask > 0] = [0, 255, 0]        # Green
    #         overlay_img[micro_mask > 0] = [255, 0, 0]         # Red
    #         overlay_img[exudate_mask > 0] = [0, 255, 255] 
    #         self.display_image(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))

    #         # Save
    #         file_path = filedialog.asksaveasfilename(defaultextension=".png",
    #                                                 filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
    #         if file_path:
    #             cv2.imwrite(file_path, overlay_img)
        # Convert processed_image to grayscale if it's not already binary mask
        # if len(self.processed_image.shape) == 3 and self.processed_image.shape[2] == 3:
        #     processed_gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
        # else:
        #     processed_gray = self.processed_image.copy()

        # # Create binary mask from processed image
        # _, binary_mask = cv2.threshold(processed_gray, 25, 255, cv2.THRESH_BINARY)

        # # Create a color overlay (e.g., red) where the mask is present
        # overlay = self.original_image.copy()
        # overlay[binary_mask == 255] = [255, 0, 0]  # Red overlay

        # # Blend original and overlay for visualization
        # blended = cv2.addWeighted(self.original_image, 0.7, overlay, 0.3, 0)

        # # Show result
        # self.display_image(blended, self.proc_label)

        # Ask user for save location
        # file_path = filedialog.asksaveasfilename(defaultextension=".png",
        #                                         filetypes=[("PNG Image", "*.png"),
        #                                                     ("JPEG Image", "*.jpg"),
        #                                                     ("All Files", "*.*")])
        # if file_path:
        #     cv2.imwrite(file_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
        #     print(f"Overlay image saved to {file_path}")





# ----------------- Main -----------------

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
