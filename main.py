import cv2 as cv
import numpy as np

# Loading the Image
image_path = "Enhancer\\assets\\shell.png"  
image = cv.imread(image_path)

# Denoise
def median(image, ksize=5):
    if not ksize % 2 == 1:
        raise ValueError("Ksize cannot be even")
    img = cv.medianBlur(image, ksize)
    return img

# LUT
def lut(image):
    lut_in = [0, 255, 150]    #[0, 255, 100]  
    lut_out = [0, 0, 255]     #[0, 80, 255]

    model = np.arange(0, 256)

    lut_8u = np.interp(model, lut_in, lut_out).astype(np.uint8)
    result = cv.LUT(image, lut_8u)

    return result

# Smoothen
def gaussian(image, iter=5):
    if not 1 <= iter <= 10:
        raise ValueError("Iterations can't be more than 10")
    img = image.copy()
    for _ in range(iter):
        ksize = [2 * _ + 1, 2 * _ + 1]
        img = cv.GaussianBlur(img, ksize, 2)
    return img

# Bicubic Interpolation
def upscale(image, scale=2):
    new_w = image.shape[1] * scale
    new_h = image.shape[0] * scale
    upscaled_image = cv.resize(image, (int(new_w), int(new_h)), interpolation=cv.INTER_CUBIC)
    return upscaled_image

# Applying effects
img=lut(gaussian(median(upscale(image,scale=3),ksize=3),iter=1))
#img = upscale(median(image,ksize=3))

# Resizing to Compare the Original and Upscaled Images
new_width = 300
new_height = int((new_width / img.shape[1]) * img.shape[0])
preview_img = cv.resize(img, (new_width, new_height))
preview_image = cv.resize(image, (new_width, new_height))

cv.imshow('Original vs Enhanced', np.hstack((preview_image, preview_img)))

# Saving the Upscaled Image
upscaled_image_path = image_path.split('\\')[0] + '\\enhanced_' + image_path.split('\\')[-1]
cv.imwrite(upscaled_image_path, img)

cv.waitKey(0)
cv.destroyAllWindows()