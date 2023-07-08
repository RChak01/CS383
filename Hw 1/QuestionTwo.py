from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
from numpy import linalg as LA
import matplotlib.pyplot as plt
import cv2

DIR = 'yalefaces'
SEPARATOR = os.sep
data = []

def standardize(arr):
    mean = arr.mean()
    std = arr.std(ddof=1)
    new_arr = (arr - mean) / std
    return new_arr

def get_k_principal_components(data, k):
    cov = np.cov(data.T)
    d, v = LA.eig(cov)
    indices = np.argsort(d)[::-1][:k]
    max_vectors = v[:, indices]
    return max_vectors

files = os.listdir(DIR)
subject_files = sorted([f for f in files if not f.endswith('.txt')])

# Load data
for filename in subject_files:
    im = Image.open(os.path.join(DIR, filename)).resize((40, 40))
    arr = np.array(im).flatten()
    data.append(arr)

data = np.array(data)
std_data = standardize(data)

# Perform PCA and reconstruction for each k
k_values = range(data.shape[1])
video_frames = []
for k in k_values:
    w = get_k_principal_components(std_data, k)

    # Project subject02.centerlight onto k principal components
    subject_img_path = os.path.join(DIR, 'subject02.centerlight')
    subject_img = Image.open(subject_img_path).resize((40, 40))
    subject_arr = np.array(subject_img).flatten()
    subject_std_arr = standardize(subject_arr)
    z = np.dot(w.T, subject_std_arr)

    # Reconstruct the person using k principal components
    reconstructed_arr = np.dot(w, z)
    # Undo standardization
    reconstructed_arr = (reconstructed_arr * subject_std_arr.std(ddof=1)) + subject_std_arr.mean()
    # Reshape to a 40x40 image/matrix
    reconstructed_img = Image.fromarray(reconstructed_arr.astype(np.uint8).reshape((40, 40)))

    # Superimpose current k value on the image
    img_with_text = reconstructed_img.copy()
    draw = ImageDraw.Draw(img_with_text)
    font = ImageFont.load_default()
    draw.text((0, 0), f'k={k}', font=font, fill=255)

    # Append the frame to the video frames list
    video_frames.append(img_with_text)

# Save the frames as a video
output_video_path = 'pca_reconstruction_video.mp4'
frame_width, frame_height = video_frames[0].size
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(output_video_path, fourcc, 10.0, (frame_width, frame_height))

for frame in video_frames:
    video_writer.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))

video_writer.release()
