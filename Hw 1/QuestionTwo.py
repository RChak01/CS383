from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import cv2

toPath = 'yalefaces'
SEPARATOR = os.sep
data = []

def standardize(arr):
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    new_arr = (arr - mean) / std
    return new_arr

def PComponents(data, k):
    mean = np.mean(data, axis=0)
    centered_data = data - mean
    _, _, v = np.linalg.svd(centered_data, full_matrices=False)
    max_vectors = v[:k]
    return max_vectors

files = os.listdir(toPath)
subject_files = sorted([f for f in files if not f.endswith('.txt')])

for filename in subject_files:
    im = Image.open(os.path.join(toPath, filename)).resize((40, 40))
    arr = np.array(im).flatten()
    data.append(arr)

data = np.array(data)
std_data = standardize(data)

k_values = range(1, data.shape[0] + 1)
video_frames = []

reconstructed_images = np.zeros((len(k_values), 40, 40), dtype=np.uint8)

for i, k in enumerate(k_values):
    subject_img_path = subject_files[i]
    subject_img = Image.open(os.path.join(toPath, subject_img_path)).resize((40, 40))
    subject_arr = np.array(subject_img).flatten()
    subject_std_arr = standardize(subject_arr)

    w = PComponents(std_data, k)
    z = np.dot(subject_std_arr, w.T)

    reconstructed_arr = np.sum(w * z[:, np.newaxis], axis=0) + np.mean(data, axis=0)
    reconstructed_arr = (reconstructed_arr * std_data.std(ddof=1)) + std_data.mean()

    reconstructed_img = Image.fromarray(reconstructed_arr.astype(np.uint8).reshape((40, 40)))

    img_with_text = reconstructed_img.copy()
    draw = ImageDraw.Draw(img_with_text)
    font = ImageFont.load_default()
    draw.text((0, 0), f'k={k}', font=font, fill=255)

    video_frames.append(img_with_text)
    reconstructed_images[k - 1] = reconstructed_arr.reshape((40, 40))

output_video_path = 'pca_reconstruction_video.mp4'
frame_width, frame_height = video_frames[0].size
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(output_video_path, fourcc, 10.0, (frame_width, frame_height))

for frame in video_frames:
    video_writer.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))

video_writer.release()
