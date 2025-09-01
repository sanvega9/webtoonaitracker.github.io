# from PIL import Image
# import numpy as np
# #load image path 
# def load_image_grayscale(path):
#     return Image.open(path).convert("L")

# def load_image_color(path):
#     return Image.open(path)

# # ------------------------------
# # 1. Down-Sampling (64x64)
# # ------------------------------
# def downsample(image, factor=4):
#     img_array = np.array(image)
#     h, w = img_array.shape
#     downsampled = img_array[::factor, ::factor]
#     return Image.fromarray(downsampled)

# # ------------------------------
# # 2. Nearest-Neighbor Interpolation
# # ------------------------------
# def nearest_neighbor(image, scale=4):
#     img_array = np.array(image)
#     h, w = img_array.shape
#     new_h, new_w = h * scale, w * scale
#     upsampled = np.zeros((new_h, new_w), dtype=np.uint8)
#     for i in range(new_h):
#         for j in range(new_w):
#             upsampled[i, j] = img_array[i // scale, j // scale]
#     return Image.fromarray(upsampled)

# # ------------------------------
# # 3.  Bilinear Interpolation
# # ------------------------------
# def bilinear_interpolation(image, scale=4):
#     img_array = np.array(image)
#     h, w = img_array.shape
#     new_h, new_w = h * scale, w * scale
#     upsampled = np.zeros((new_h, new_w), dtype=np.uint8)
    
#     for i in range(new_h):
#         for j in range(new_w):
#             x, y = i / scale, j / scale
#             x0, y0 = int(x), int(y)
#             x1, y1 = min(x0 + 1, h - 1), min(y0 + 1, w - 1)
#             a, b = x - x0, y - y0
            
#             interpolated = (1 - a) * (1 - b) * img_array[x0, y0] + \
#                            a * (1 - b) * img_array[x1, y0] + \
#                            (1 - a) * b * img_array[x0, y1] + \
#                            a * b * img_array[x1, y1]
#             upsampled[i, j] = int(interpolated)
    
#     return Image.fromarray(upsampled)

# # ------------------------------
# # 4. Compute MSE
# # ------------------------------
# def compute_mse(img1, img2):
#     img1 = np.array(img1.resize((256, 256)))
#     img2 = np.array(img2.resize((256, 256)))
#     return np.mean((img1 - img2) ** 2)

# # ------------------------------
# # 5. Quantization (4-bit)
# # ------------------------------
# def quantize_4bit(image):
#     img_array = np.array(image)
#     levels = 16
#     quantized = (img_array / 255 * (levels - 1)).round() * (255 // (levels - 1))
#     return Image.fromarray(quantized.astype(np.uint8))

# # ------------------------------
# # 6. Grayscale Conversion
# # ------------------------------
# def grayscale_conversion(image):
#     img_array = np.array(image)
#     r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
#     grayscale = (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.uint8)
#     return Image.fromarray(grayscale)

# # ------------------------------
# # 7. Extract RGB Channels
# # ------------------------------
# def extract_rgb_channels(image):
#     img_array = np.array(image)
#     return Image.fromarray(img_array[:, :, 0]), Image.fromarray(img_array[:, :, 1]), Image.fromarray(img_array[:, :, 2])

# # Run Processing Steps
# lenna_gray = load_image_grayscale("lenna.jpg")
# lenna_color = load_image_color("lenna_color.tiff")

# # Downsample and reconstruct
# lenna_downsampled = downsample(lenna_gray)
# lenna_nearest = nearest_neighbor(lenna_downsampled)
# lenna_bilinear = bilinear_interpolation(lenna_downsampled)

# # Compute MSE
# mse_nearest = compute_mse(lenna_gray, lenna_nearest)
# mse_bilinear = compute_mse(lenna_gray, lenna_bilinear)

# # Quantization
# lenna_quantized = quantize_4bit(lenna_gray)
# mse_quantization = compute_mse(lenna_gray, lenna_quantized)

# # Convert color image to grayscale
# lenna_manual_gray = grayscale_conversion(lenna_color)
# mse_grayscale = compute_mse(lenna_gray, lenna_manual_gray)

# # Extract RGB channels
# red_channel, green_channel, blue_channel = extract_rgb_channels(lenna_color)
# mse_red = compute_mse(lenna_gray, red_channel)
# mse_green = compute_mse(lenna_gray, green_channel)
# mse_blue = compute_mse(lenna_gray, blue_channel)

# # Compute Average Errors
# average_mse_interpolation = (mse_nearest + mse_bilinear) / 2
# average_mse_color_related = (mse_quantization + mse_grayscale + mse_red + mse_green + mse_blue) / 5

# # Print Results
# print(f"Nearest-Neighbor MSE: {mse_nearest:.4f}")
# print(f"Bilinear MSE: {mse_bilinear:.4f}")
# print(f"Quantization MSE (B=4): {mse_quantization:.4f}")
# print(f"Grayscale MSE: {mse_grayscale:.4f}")
# print(f"Red Channel MSE: {mse_red:.4f}")
# print(f"Green Channel MSE: {mse_green:.4f}")
# print(f"Blue Channel MSE: {mse_blue:.4f}")

# print("\n------------------------------")
# print(f"Average Interpolation MSE: {average_mse_interpolation:.4f}")
# print(f"Average Color-Related MSE: {average_mse_color_related:.4f}")

# # Save Processed Images
# lenna_downsampled.save("lenna_downsampled.jpg")
# lenna_nearest.save("lenna_nearest.jpg")
# lenna_bilinear.save("lenna_bilinear.jpg")
# lenna_quantized.save("lenna_quantized.jpg")
# lenna_manual_gray.save("lenna_manual_gray.jpg")
# red_channel.save("lenna_red_channel.jpg")
# green_channel.save("lenna_green_channel.jpg")
# blue_channel.save("lenna_blue_channel.jpg")

import numpy as np # recommended convention
x = np.array([1,1,1],dtype=np.float32)
x.itemsize
print(x.itemset)