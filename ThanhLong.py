import cv2
import streamlit as st
import numpy as np
from PIL import Image
from io import BytesIO


def brighten_image(image, amount):
    img_bright = cv2.convertScaleAbs(image, beta=amount)
    return img_bright

def contrast(img, amount):
    img_contrast = cv2.convertScaleAbs(img, alpha = amount)
    return img_contrast

def blur_image(image, amount):
    blur_img = cv2.GaussianBlur(image, (11, 11), amount)
    return blur_img


def enhance_details(img):
    hdr = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
    return hdr

def sharpening(img, amount):
    kernel = np.array([[ 0, -1,  0],
                   [-1,  5, -1],
                   [ 0, -1,  0]])

    sharped = cv2.filter2D(img, ddepth = amount, kernel = kernel)
    return sharped

def bw_filter(img):
    img_gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    return img_gray

def vignette(img, level):
    
    height, width = img.shape[:2]  

    X_resultant_kernel = cv2.getGaussianKernel(width, width/level)
    Y_resultant_kernel = cv2.getGaussianKernel(height, height/level)
        
    kernel = Y_resultant_kernel * X_resultant_kernel.T 
    mask = kernel / kernel.max()
    
    img_vignette = np.copy(img)

    for i in range(3):
        img_vignette[:,:,i] = img_vignette[:,:,i] * mask

    return img_vignette

def main_loop():
    st.title("Thành Long Image Processing App")
    st.header("Author: Nguyen Thanh Long - 210085 with the help of Streamlit")

    blur_rate = st.sidebar.slider("Blurring", min_value=0.001, max_value=100.001)
    brightness_amount = st.sidebar.slider("Brightness", min_value=-50, max_value=50, value=0)
    contrast_amount = st.sidebar.slider("Contrast", min_value=0.001, max_value=1.999, value=1.001)
    vignette_amount = st.sidebar.slider("Vignette", min_value=0.001, max_value=2.999, value=0.001)
    #sharpen_amount = st.sidebar.slider("Sharpening", min_value=-50, max_value=50, value=0)
    apply_enhancement_filter = st.sidebar.checkbox('Enhance Details')
    apply_BW_filter = st.sidebar.checkbox('B&W')

    image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])
    if not image_file:
        return None

    original_image = Image.open(image_file)
    original_image = np.array(original_image)

    processed_image = blur_image(original_image, blur_rate)
    processed_image = brighten_image(processed_image, brightness_amount)
    processed_image = vignette(processed_image, vignette_amount)
    processed_image = contrast(processed_image, contrast_amount)
    #Processed_image = sharpening(processed_image, sharpen_amount)
    
    if apply_enhancement_filter:
        processed_image = enhance_details(processed_image)
    
    if apply_BW_filter:
        processed_image = bw_filter(processed_image)
    
    st.text("Original Image vs Processed Image")
    st.image([original_image, processed_image]) 
    
    st.header("DOWNLOAD INSTRUCTION")
    st.subheader('RIGHT click at the processed image and then CLICK _:blue["save image as"]_ to choose your own directory :sunglasses:')

if __name__ == '__main__':
    main_loop()