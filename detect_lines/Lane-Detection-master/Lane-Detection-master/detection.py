from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import argparse
import math
import cv2


def region_of_interest(img, vertices):
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


def draw_lines(img, lines, color=(255, 0, 0), thickness=10):
    # reshape lines to a 2d matrix
    print(lines.shape)
    lines = lines.reshape(lines.shape[0], lines.shape[2])
    print(lines.shape)
    # create array of slopes
    slopes = (lines[:,3] - lines[:,1]) /(lines[:,2] - lines[:,0])
    # remove junk from lists

    lines = lines[~np.isnan(lines) & ~np.isinf(lines)]
    slopes = slopes[~np.isnan(slopes) & ~np.isinf(slopes)]
    # convert lines into list of points
    lines.shape = (lines.shape[0]//2,2)

    # Right lane
    # move all points with negative slopes into right "lane"
    right_slopes = slopes[slopes < 0]
    right_lines = np.array(list(filter(lambda x: x[0] > (img.shape[1]/2), lines)))
    max_right_x, max_right_y = right_lines.max(axis=0)
    min_right_x, min_right_y = right_lines.min(axis=0)

    # Left lane
    # all positive  slopes go into left "lane"
    left_slopes = slopes[slopes > 0]
    left_lines = np.array(list(filter(lambda x: x[0] < (img.shape[1]/2), lines)))
    max_left_x, max_left_y = left_lines.max(axis=0)
    min_left_x, min_left_y = left_lines.min(axis=0)

    # Curve fitting approach
    # calculate polynomial fit for the points in right lane
    right_curve = np.poly1d(np.polyfit(right_lines[:,1], right_lines[:,0], 2))
    left_curve  = np.poly1d(np.polyfit(left_lines[:,1], left_lines[:,0], 2))

    # shared ceiling on the horizon for both lines
    min_y = min(min_left_y, min_right_y)

    # use new curve function f(y) to calculate x values
    max_right_x = int(right_curve(img.shape[0]))
    min_right_x = int(right_curve(min_right_y))

    min_left_x = int(left_curve(img.shape[0]))

    r1 = (min_right_x, min_y)
    r2 = (max_right_x, img.shape[0])
    print('Right points r1 and r2,', r1, r2)
    cv2.line(img, r1, r2, color, thickness)

    l1 = (max_left_x, min_y)
    l2 = (min_left_x, img.shape[0])
    print('Left points l1 and l2,', l1, l2)
    cv2.line(img, l1, l2, color, thickness)



def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    if lines is not None:
        draw_lines(line_img, lines)
    return line_img

# Takes in a single frame or an image and returns a marked image
def mark_lanes(image):
    if image is None: raise ValueError("no image given to mark_lanes")
    # grayscale the image to make finding gradients clearer
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)

    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    edges_img = cv2.Canny(np.uint8(blur_gray), low_threshold, high_threshold)

    h = image.shape[0]
    w = image.shape[1]

    # левая нижняя точка области интереса
    bottom_left = (250, h)
    # левая верхняя точка области интереса
    top_left = (300, h / 2)
    # правая верхняя точка области интереса
    top_right = (650, h / 2)
    # правая нижняя точка области интереса
    bottom_right = (750, h)

    # создание массива, определяющего область интереса на изображении
    vertices = np.array([[bottom_left,
                          top_left,
                          top_right,
                          bottom_right]],
                        dtype=np.int32)

    masked_edges = region_of_interest(edges_img, vertices )


    # Define the Hough transform parameters
    # аргументы ро и тета задают требуемую разрешающую способность
    # для прямыых (т.е. квантизацию аккумуляторной плоскости)
    rho             = 4           # distance resolution in pixels of the Hough grid
    theta           = np.pi/180   # angular resolution in radians of the Hough grid
    threshold       = 30       # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 20       # minimum number of pixels making up a line
    max_line_gap    = 20       # maximum gap in pixels between connectable line segments

    line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)

    # Draw the lines on the edge image
    # initial_img * α + img * β + λ
    lines_edges = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    #cv2.imshow('img', lines_edges)
    #cv2.waitKey(0)
    return lines_edges


def read_image_for_marking(img_filepath):
    # read in the image
    image = cv2.imread(img_filepath)
    print('Reading image :', img_filepath, '\nDimensions:', image.shape)

    marked_lanes = mark_lanes(image)

    # show the image to plotter and then save it to a file
    plt.imshow(marked_lanes)
    plt.savefig(img_filepath[:-4] + '_output.png')


if __name__ == "__main__":
    # set up parser
    #read_image_for_marking('D:\\PythonProjects\\detect_lines\\Lane-Detection-master\\Lane-Detection-master\\test_images\\solidWhiteCurve.jpg')
    #read_image_for_marking('D:\\PythonProjects\\detect_lines\\Lane-Detection-master\\Lane-Detection-master\\test_images\\road3.jpg')

    clip = VideoFileClip('solidWhiteRight.mp4')
    clip = clip.fl_image(mark_lanes)
    clip.write_videofile('output_' + 'solidWhiteRight.mp4', audio=False)


