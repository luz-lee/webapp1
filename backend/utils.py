import numpy as np
import cv2
import random
from concurrent.futures import ThreadPoolExecutor
import os

# 공통 유틸리티 함수
def resize_image(img, shape):
    return cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)

def text_to_image(text, img_shape, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=3, thickness=5):
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (img_shape[1] - text_size[0]) // 2
    text_y = (img_shape[0] + text_size[1]) // 2

    img_wm = np.zeros(img_shape, dtype=np.uint8)
    cv2.putText(img_wm, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

    return img_wm

# 이미지 워터마크 함수
def apply_watermark(img, text_or_image_path, alpha=5, use_image=False):
    height, width = img.shape[:2]
    if use_image and os.path.exists(text_or_image_path):
        img_wm = cv2.imread(text_or_image_path, cv2.IMREAD_GRAYSCALE)
        img_wm = resize_image(img_wm, (height, width))
    else:
        img_wm = text_to_image(text_or_image_path, (height, width))

    img_f = np.fft.fft2(img)
    
    y_random_indices, x_random_indices = list(range(height)), list(range(width))
    random.seed(2021)
    random.shuffle(x_random_indices)
    random.shuffle(y_random_indices)
    
    random_wm = np.zeros(img.shape, dtype=np.uint8)
    
    for y in range(height):
        for x in range(width):
            random_wm[y_random_indices[y], x_random_indices[x]] = img_wm[y, x]
    
    result_f = img_f + alpha * random_wm
    
    result = np.fft.ifft2(result_f)
    result = np.real(result)
    result = result.astype(np.uint8)
    
    return result

def extract_watermark(img_ori, img_input, alpha=5):
    height, width = img_ori.shape[:2]
    
    img_input = resize_image(img_input, (height, width))
    
    img_ori_f = np.fft.fft2(img_ori)
    img_input_f = np.fft.fft2(img_input)
    
    watermark = (img_input_f - img_ori_f) / alpha
    watermark = np.real(watermark).astype(np.uint8)
    
    y_random_indices, x_random_indices = list(range(height)), list(range(width))
    random.seed(2021)
    random.shuffle(x_random_indices)
    random.shuffle(y_random_indices)
    
    result2 = np.zeros(watermark.shape, dtype=np.uint8)
    
    for y in range(height):
        for x in range(width):
            result2[y, x] = watermark[y_random_indices[y], x_random_indices[x]]
    
    return watermark, result2

# 비디오 워터마크 함수
def process_video(input_path, output_path, process_function, *args, frame_skip=1):
    try:
        cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        with ThreadPoolExecutor() as executor:
            futures = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_skip == 0:
                    futures.append(executor.submit(process_function, frame, *args))
                frame_idx += 1

            for future in futures:
                processed_frame = future.result()
                out.write(processed_frame)

        cap.release()
        out.release()
    except Exception as e:
        print(f"Error processing video: {e}")

def apply_watermark_to_video(input_path, output_path, text_or_image_path, frame_skip=1, use_image=False):
    process_video(input_path, output_path, apply_watermark, text_or_image_path, frame_skip, use_image)

def extract_frame_watermark(frame, watermarked_frame, alpha=5):
    _, extracted_frame = extract_watermark(frame, watermarked_frame, alpha)
    return extracted_frame

def extract_watermark_from_video(input_path, watermarked_path, output_path, alpha=5, frame_skip=1):
    try:
        cap = cv2.VideoCapture(watermarked_path)
        watermarked_frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            watermarked_frames.append(frame)
        cap.release()

        cap = cv2.VideoCapture(input_path)
        frame_list = []
        frame_idx = 0
        with ThreadPoolExecutor() as executor:
            futures = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_skip == 0:
                    watermarked_frame = watermarked_frames[frame_idx]
                    futures.append(executor.submit(extract_frame_watermark, frame, watermarked_frame, alpha))
                frame_idx += 1

            for future in futures:
                processed_frame = future.result()
                frame_list.append(processed_frame)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frame_list:
            out.write(frame)

        out.release()
    except Exception as e:
        print(f"Error extracting watermark: {e}")
