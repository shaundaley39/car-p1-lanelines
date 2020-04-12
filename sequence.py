import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    lcount, rcount, lgrad, rgrad, loffset, roffset = (0, 0, 0, 0, 0, 0)
    for line in lines:
        for x1, y1, x2, y2 in line:
            try:
                grad = (x2-x1)/(y2-y1)
            except (Exception):
                pass
            offset = x1 - grad * y1
            if grad < -0.5 and grad > -3 and offset < 1000:
                # left lane
                lcount += 1
                lgrad += grad
                loffset += offset
            elif grad > 0.5 and grad < 3 and offset > -200:
                # right lane
                rcount += 1
                rgrad += grad
                roffset += offset
    if lcount > 0:
        lgrad /= lcount
        loffset /= lcount
        y1_left = img.shape[1]
        x1_left = int((y1_left * lgrad + loffset))
        x2_left = int(img.shape[1]*0.48)
        y2_left = int((x2_left - loffset) / lgrad)
        try:
            cv2.line(img, (x1_left, y1_left), (x2_left, y2_left), color,
                     thickness)
        except (Exception):
            pass
    if rcount > 0:
        rgrad /= rcount
        roffset /= rcount
        y1_right = img.shape[1]
        x1_right = int((y1_right * rgrad + roffset))
        x2_right = int(img.shape[1]*0.52)
        y2_right = int((x2_right - roffset) / rgrad)
        try:
            cv2.line(img, (x1_right, y1_right), (x2_right, y2_right), color,
                     thickness)
        except (Exception):
            pass


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


def process_image(image, out_dir, image_name):
    img = grayscale(image)

    # gaussian blur - we care about image regions larger than individual pixels
    img = gaussian_blur(img, 5)
    mpimg.imsave(out_dir + "blurred_" + image_name, img)

    # Canny transform - select only the points in the image with sufficiently high gradient (likely on edge)
    img = canny(img, 50, 150)
    mpimg.imsave(out_dir + "edges_" + image_name, img)

    # mask - only a small part of the image should be processed for road lines
    imshape = img.shape
    vertices = np.array([[(0, imshape[0]), (0.45*imshape[1], 0.55*imshape[0]),
                          (0.55*imshape[1], 0.55*imshape[0]),
                          (imshape[1], imshape[0])]], dtype=np.int32)
    img = region_of_interest(img, vertices)
    mpimg.imsave(out_dir + "roi_" + image_name, img)

    # hough transform and line extraction
    rho = 2
    threshold = 64
    theta = np.pi/180
    min_line_len = 64
    max_line_gap = 128
    img = hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap)
    mpimg.imsave(out_dir + "hough_lines_" + image_name, img)

    # overlay on original image
    img = weighted_img(img, image)
    mpimg.imsave(out_dir + "final_" + image_name, img)
    return img

test_dir = "test_images/"
out_dir = "test_images_intermediate/"

for fname in os.listdir(test_dir):
    in_img = mpimg.imread(test_dir + fname)
    process_image(in_img, out_dir, fname)
