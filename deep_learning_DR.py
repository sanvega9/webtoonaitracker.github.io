# run cnn.py CNN model trained and saved and view SVM and report
#run python image_processing_DR.py connect with these file code
import cv2
import numpy as np
from skimage import filters, morphology, feature, exposure
from skimage.filters import frangi
from skimage.morphology import skeletonize
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt

def apply_clahe(gray_img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_img)

def gaussian_blur(img, kernel_size=5):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def apply_frangi(img):
    img_norm = img / 255.0
    vessel = frangi(img_norm)
    return img_as_ubyte(vessel)

def adaptive_threshold(img):
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, 15, -2)

def extract_skeleton(binary_img):
    skeleton = skeletonize(binary_img > 0)
    return img_as_ubyte(skeleton)

def detect_lesions(clahe_img):
    # Exudates (bright spots)
    exudate_thresh = filters.threshold_otsu(clahe_img)
    exudates = (clahe_img > exudate_thresh + 40).astype(np.uint8) * 255

    # Hemorrhages (dark round blobs)
    inv = cv2.bitwise_not(clahe_img)
    hemorrhages = cv2.medianBlur(inv, 5)
    hemorrhages = cv2.adaptiveThreshold(hemorrhages, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)

    # Microaneurysms (small bright spots)
    laplacian = cv2.Laplacian(clahe_img, cv2.CV_64F)
    microaneurysms = np.uint8((laplacian > np.percentile(laplacian, 99)) * 255)

    return exudates, hemorrhages, microaneurysms

def overlay_colored_lesions(original, exudates, hemorrhages, microaneurysms, skeleton):
    color_image = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

    color_image[exudates > 0] = [0, 255, 255]       # Yellow for exudates
    color_image[hemorrhages > 0] = [0, 0, 255]      # Red for hemorrhages
    color_image[microaneurysms > 0] = [255, 0, 0]   # Blue for microaneurysms
    color_image[skeleton > 0] = [255, 255, 255]     # White for skeleton vessels

    return color_image

# === DRIVER FUNCTION ===

def process_retinal_image(image_path):
    # Load and convert to grayscale
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe_img = apply_clahe(gray)
    blurred = gaussian_blur(clahe_img)
    vessel_enhanced = apply_frangi(blurred)
    thresholded = adaptive_threshold(vessel_enhanced)
    skeleton = extract_skeleton(thresholded)

    exudates, hemorrhages, microaneurysms = detect_lesions(clahe_img)

    result = overlay_colored_lesions(gray, exudates, hemorrhages, microaneurysms, skeleton)

    # Show results
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original with Lesions & Skeleton")
    plt.imshow(result)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("CLAHE + Frangi + Skeleton")
    plt.imshow(skeleton, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Example usage
process_retinal_image('archive/Diagnosis of Diabetic Retinopathy/valid/DR/0f882877bf13_png.rf.f328da55e3d0a3754e285039a2a22b0d.jpg')  # replace with your image path



import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load grayscale image
img = Image.open('archive/Diagnosis of Diabetic Retinopathy/valid/DR/0f882877bf13_png.rf.f328da55e3d0a3754e285039a2a22b0d.jpg').convert('L')
img_np = np.array(img)

# Compute 2D FFT and shift
f = np.fft.fft2(img_np)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(1 + np.abs(fshift))

# Show original and spectrum
plt.subplot(1, 2, 1)
plt.imshow(img_np, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum')
plt.show()
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import frangi
from skimage.morphology import skeletonize
from skimage.util import img_as_ubyte
from skimage.morphology import white_tophat, disk

# ========== Image Processing Functions ==========

def apply_clahe(gray_img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_img)

def vessel_segmentation(img):
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    frangi_img = frangi(blurred / 255.0)
    frangi_ubyte = img_as_ubyte(frangi_img)
    thresholded = cv2.adaptiveThreshold(frangi_ubyte, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, 15, -2)
    skeleton = skeletonize(thresholded > 0)
    return img_as_ubyte(skeleton)

def detect_microaneurysms(gray_img):
    tophat = white_tophat(gray_img, disk(8))
    blur = cv2.GaussianBlur(tophat, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    return cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

def detect_exudates(color_img):
    hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (20, 40, 80), (45, 255, 255))
    v_mask = cv2.inRange(hsv[:, :, 2], 200, 255)
    combined = cv2.bitwise_and(mask, v_mask)
    return cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

def show_fft(gray_img):
    f = np.fft.fft2(gray_img)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(1 + np.abs(fshift))
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(gray_img, cmap='gray')
    plt.title("Grayscale Image")
    plt.subplot(1, 2, 2)
    plt.imshow(magnitude, cmap='gray')
    plt.title("FFT Magnitude Spectrum")
    plt.tight_layout()
    plt.show()

# ========== GUI Application ==========

class DRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Diabetic Retinopathy Analyzer")
        self.image = None
        self.gray = None

        # Buttons
        tk.Button(root, text="Load Image", command=self.load_image).pack()
        tk.Button(root, text="CLAHE + Vessel Segmentation", command=self.process_vessels).pack()
        tk.Button(root, text="Detect Microaneurysms", command=self.process_microaneurysms).pack()
        tk.Button(root, text="Detect Exudates", command=self.process_exudates).pack()
        tk.Button(root, text="Show FFT", command=self.process_fft).pack()

        # Canvas
        self.canvas = tk.Label(root)
        self.canvas.pack()

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            img = cv2.imread(path)
            self.image = img
            self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.show_image(self.gray)

    def show_image(self, img):
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)))
        self.canvas.imgtk = imgtk
        self.canvas.config(image=imgtk)

    def process_vessels(self):
        if self.gray is not None:
            clahe_img = apply_clahe(self.gray)
            skeleton = vessel_segmentation(clahe_img)
            self.show_image(skeleton)

    def process_microaneurysms(self):
        if self.gray is not None:
            mask = detect_microaneurysms(self.gray)
            self.show_image(mask)

    def process_exudates(self):
        if self.image is not None:
            mask = detect_exudates(self.image)
            self.show_image(mask)

    def process_fft(self):
        if self.gray is not None:
            show_fft(self.gray)

# ========== Run Application ==========

if __name__ == "__main__":
    root = tk.Tk()
    app = DRApp(root)
    root.mainloop()
    
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import frangi
from skimage.morphology import skeletonize, white_tophat, disk
from skimage.util import img_as_ubyte
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os


# ========== Image Processing Functions ==========
model = load_model('dr_detector_mobilenetv2.h5')  # Update to your model path
NUM_CLASSES = model.output_shape[-1]
severity_labels = ["DR", "No DR"] if NUM_CLASSES == 1 else ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

def apply_clahe(gray_img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_img)

def vessel_segmentation(img):
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    frangi_img = frangi(blurred / 255.0)
    frangi_ubyte = img_as_ubyte(frangi_img)
    thresholded = cv2.adaptiveThreshold(frangi_ubyte, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, 15, -2)
    skeleton = skeletonize(thresholded > 0)
    return img_as_ubyte(skeleton)

def detect_microaneurysms(gray_img):
    tophat = white_tophat(gray_img, disk(8))
    blur = cv2.GaussianBlur(tophat, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    return cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

def detect_exudates(color_img):
    hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (20, 40, 80), (45, 255, 255))
    v_mask = cv2.inRange(hsv[:, :, 2], 200, 255)
    combined = cv2.bitwise_and(mask, v_mask)
    return cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

def show_fft(gray_img):
    f = np.fft.fft2(gray_img)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(1 + np.abs(fshift))
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(gray_img, cmap='gray')
    plt.title("Grayscale Image")
    plt.subplot(1, 2, 2)
    plt.imshow(magnitude, cmap='gray')
    plt.title("FFT Magnitude Spectrum")
    plt.tight_layout()
    plt.show()

def manual_haar_wavelet(gray_img):
    # Downsample image to even dimensions
    h, w = gray_img.shape
    h -= h % 2
    w -= w % 2
    gray_img = gray_img[:h, :w].astype(np.float32)

    # Split into even/odd rows and columns
    LL = (gray_img[0::2, 0::2] + gray_img[0::2, 1::2] + gray_img[1::2, 0::2] + gray_img[1::2, 1::2]) / 4
    LH = (gray_img[0::2, 0::2] - gray_img[0::2, 1::2] + gray_img[1::2, 0::2] - gray_img[1::2, 1::2]) / 4
    HL = (gray_img[0::2, 0::2] + gray_img[0::2, 1::2] - gray_img[1::2, 0::2] - gray_img[1::2, 1::2]) / 4
    HH = (gray_img[0::2, 0::2] - gray_img[0::2, 1::2] - gray_img[1::2, 0::2] + gray_img[1::2, 1::2]) / 4

    # Normalize for display
    def norm(img):
        return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return norm(LL), norm(LH), norm(HL), norm(HH)
def segment_vessels(img_np):
    green_channel = img_np[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green_channel)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    frangi_img = frangi(blurred / 255.0)
    vessel_mask = (frangi_img > 0.1).astype(np.uint8)
    skeleton = skeletonize(vessel_mask).astype(np.uint8) * 255
    return skeleton

def detect_microaneurysms_DL(img_np):
    green = img_np[:, :, 1]
    tophat = cv2.morphologyEx(green, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    _, th = cv2.threshold(tophat, 20, 255, cv2.THRESH_BINARY)
    return th

def detect_exudates_DL(img_np):
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    _, thresh = cv2.threshold(l, 180, 255, cv2.THRESH_BINARY)
    return thresh

def create_overlay(original_img, vessels, micro, exudates):
    overlay = original_img.copy()
    overlay[vessels > 0] = [255, 0, 0]       # Red for vessels
    overlay[micro > 0] = [0, 255, 0]         # Green for microaneurysms
    overlay[exudates > 0] = [255, 255, 0]    # Yellow for exudates
    return overlay
# ========== GUI Application ==========

class DRApp:
    def __init__(self, master):
        self.root = master
        self.root.title("Diabetic Retinopathy Analyzer")
        self.image = None
        self.gray = None

        # Buttons
        tk.Button(self.root, text="Load Image", command=self.load_image).pack()
        tk.Button(self.root, text="CLAHE + Vessel Segmentation", command=self.process_vessels).pack()
        tk.Button(self.root, text="Detect Microaneurysms", command=self.process_microaneurysms).pack()
        tk.Button(self.root, text="Detect Exudates", command=self.process_exudates).pack()
        tk.Button(self.root, text="Show FFT", command=self.process_fft).pack()
        tk.Button(self.root, text="Overlay & Save Result", command=self.overlay_and_save).pack()
        # tk.Button(self.root, text="Classify DR vs No DR", command=self.classify_dr).pack()
        tk.Button(self.root, text="Show Processing Graph", command=self.show_processing_pipeline).pack()
        self.load_button = tk.Button(self.root, text="Select Images", command=self.predict_batch)
        self.load_button.pack()
        self.canvas = tk.Canvas(self.root)
        self.scroll_y = tk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scroll_y.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scroll_y.pack(side="right", fill="y")


        # Image canvas
        self.image_label = tk.Label(self.root)
        self.image_label.pack()

        self.imgtk = None  # Hold reference to PhotoImage

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            img = cv2.imread(path)
            self.image = img
            self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.show_image(self.gray)

    def show_image(self, img):
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        imgtk = ImageTk.PhotoImage(image=pil_img)
        self.imgtk = imgtk  # Prevent garbage collection
        self.image_label.config(image=imgtk)

    def process_vessels(self):
        if self.gray is not None:
            clahe_img = apply_clahe(self.gray)
            skeleton = vessel_segmentation(clahe_img)
            self.show_image(skeleton)

    def process_microaneurysms(self):
        if self.gray is not None:
            mask = detect_microaneurysms(self.gray)
            self.show_image(mask)

    def process_exudates(self):
        if self.image is not None:
            mask = detect_exudates(self.image)
            self.show_image(mask)

    def process_fft(self):
        if self.gray is not None:
            show_fft(self.gray)

    def overlay_and_save(self):
        if self.image is None or self.gray is None:
            return

        clahe_img = apply_clahe(self.gray)
        vessel_mask = vessel_segmentation(clahe_img)
        micro_mask = detect_microaneurysms(self.gray)
        exudate_mask = detect_exudates(self.image)

        overlay_img = self.image.copy()
        overlay_img[vessel_mask > 0] = [0, 255, 0]        # Green
        overlay_img[micro_mask > 0] = [255, 0, 0]         # Red
        overlay_img[exudate_mask > 0] = [0, 255, 255]     # Yellow

        self.show_image(overlay_img)

        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if file_path:
            cv2.imwrite(file_path, overlay_img)
    def load_and_predict(self, file_path):
        img = Image.open(file_path).resize((224, 224))
        tk_img = ImageTk.PhotoImage(img)
        img_np = np.array(img)

        # Model prediction
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)

        if NUM_CLASSES == 1:
            score = pred[0][0]
            label = "No Diabetic Retinopathy" if score > 0.5 else "Diabetic Retinopathy"
            confidence = score if score > 0.5 else 1 - score
        else:
            class_idx = np.argmax(pred[0])
            confidence = np.max(pred[0])
            label = severity_labels[class_idx]

        # Segmentation
        vessels = segment_vessels(img_np)
        micro = detect_microaneurysms_DL(img_np)
        exudates = detect_exudates_DL(img_np)
        overlay_img = create_overlay(img_np, vessels, micro, exudates)
        overlay_pil = Image.fromarray(overlay_img).resize((224, 224))
        overlay_tk = ImageTk.PhotoImage(overlay_pil)

        return {
            "filename": os.path.basename(file_path),
            "tk_img": tk_img,
            "overlay_img": overlay_tk,
            "label": label,
            "confidence": confidence
        }

    def predict_batch(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        for path in file_paths:
            img_data = self.load_and_predict(path)
            self.display_result(img_data)

    def display_result(self, img):
        name_label = tk.Label(self.scrollable_frame, text=f"{img['filename']}: {img['label']} ({img['confidence']*100:.2f}%)")
        name_label.pack()

        img_label = tk.Label(self.scrollable_frame, image=img['tk_img'])
        img_label.image = img['tk_img']
        img_label.pack()

        overlay_label = tk.Label(self.scrollable_frame, image=img['overlay_img'])
        overlay_label.image = img['overlay_img']
        overlay_label.pack()

    
    def show_processing_pipeline(self):
        if self.gray is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        img = self.gray.astype(np.uint8)
        kernel_size = 5
        pad = kernel_size // 2
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

        # Manual Erosion
        eroded = np.zeros_like(img)
        padded_img = np.pad(img, pad, mode='constant', constant_values=255)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                region = padded_img[i:i+kernel_size, j:j+kernel_size]
                eroded[i, j] = np.min(region[kernel == 1])

        # Manual Dilation
        dilated = np.zeros_like(img)
        padded_img = np.pad(img, pad, mode='constant', constant_values=0)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                region = padded_img[i:i+kernel_size, j:j+kernel_size]
                dilated[i, j] = np.max(region[kernel == 1])

        row = img.shape[0] // 2
        orig_profile = img[row, :]
        eroded_profile = eroded[row, :]
        dilated_profile = dilated[row, :]

        # Visualization with matplotlib
        plt.figure(figsize=(12, 6))

        plt.subplot(2, 2, 1)
        plt.title("Original Image")
        plt.imshow(img, cmap='gray')
        plt.axhline(y=row, color='r', linestyle='--')

        plt.subplot(2, 2, 2)
        plt.title("Manually Eroded Image")
        plt.imshow(eroded, cmap='gray')
        plt.axhline(y=row, color='r', linestyle='--')

        plt.subplot(2, 2, 3)
        plt.title("Manually Dilated Image")
        plt.imshow(dilated, cmap='gray')
        plt.axhline(y=row, color='r', linestyle='--')

        plt.subplot(2, 2, 4)
        plt.title(f"Row Intensity Profiles at Y={row}")
        plt.plot(orig_profile, label="Original", color='black')
        plt.plot(eroded_profile, label="Eroded", color='blue')
        plt.plot(dilated_profile, label="Dilated", color='green')
        plt.xlabel("X-axis (columns)")
        plt.ylabel("Intensity")
        plt.legend()

        plt.tight_layout()
        plt.show()


# ========== Run Application ==========

def run_gui(parent=None):
    if parent:
        window = tk.Toplevel(parent)
    else:
        window = tk.Tk()
    app = DRApp(window)
    if not parent:
        window.mainloop()

# ========== Main Entry ==========

if __name__ == "__main__":
    run_gui()
