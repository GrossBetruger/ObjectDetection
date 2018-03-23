from __future__ import division
import cv2
import numpy as np
import pyautogui as ag
import time

green = (0, 255, 0)

# def show(image):
#     # Figure size in inches
#     plt.figure(figsize=(10, 10))
#
#     # Show image, with nearest neighbour interpolation
#     plt.imshow(image, interpolation='nearest')


def overlay_mask(mask, image):
	#make the mask rgb
    rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    #calculates the weightes sum of two arrays. in our case image arrays
    #input, how much to weight each. 
    #optional depth value set to 0 no need
    img = cv2.addWeighted(rgb_mask, 0.5, image, 0.5, 0)
    return img

def find_biggest_contour(image):
    # Copy
    image = image.copy()

    im, contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Isolate largest contour
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, [biggest_contour], -1, 255, -1)
    return biggest_contour, mask


def find_contours(image):
    # Copy
    image = image.copy()

    im, contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    sorted_contours = sorted([(cv2.contourArea(contour), contour) for contour in contours], key=lambda x : x[0])
    return [x[1] for x in list(reversed(sorted_contours))]


def circle_contour(image, contour):
    # Bounding ellipse
    image_with_ellipse = image.copy()
    #easy function
    ellipse = cv2.fitEllipse(contour)
    #add it
    cv2.ellipse(image_with_ellipse, ellipse, green, 2, cv2.CV_8S)
    return image_with_ellipse


def move_around(point):
    from random import randint
    for _ in range(7):
        point = (point[0]+randint(-5, 5), point[1]+randint(-5, 5))
        ag.moveTo(*point)
        time.sleep(.3)


def find_red_objects(image):
    #RGB stands for Red Green Blue. Most often, an RGB color is stored 
    #in a structure or unsigned integer with Blue occupying the least 
    #significant  area  (a byte in 32-bit and 24-bit formats), Green the 
    #second least, and Red the third least. BGR is the same, except the 
    #order of areas is reversed. Red occupies the least significant area,
    # Green the second (still), and Blue the third.
    # we'll be manipulating pixels directly
    #most compatible for the transofrmations we're about to do
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    #eliminate noise from our image. clean. smooth colors without dots
    #Blurs an image using a Gaussian filter. input, kernel size, how much to filter, empty)
    image_blur = cv2.GaussianBlur(image, (7, 7), 0)
    # unlike RGB, HSV separates luma, or the image intensity, from
    # chroma or the color information.
    #just want to focus on color, segmentation
    image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)

    # Filter by colour
    # 0-10 hue
    #minimum red amount, max red amount
    min_red = np.array([0, 100, 80])
    max_red = np.array([10, 256, 256])
    #layer
    mask1 = cv2.inRange(image_blur_hsv, min_red, max_red)

    #birghtness of a color is hue
    # 170-180 hue
    min_red2 = np.array([170, 100, 80])
    max_red2 = np.array([180, 256, 256])
    mask2 = cv2.inRange(image_blur_hsv, min_red2, max_red2)

    #looking for what is in both ranges
    # Combine masks
    mask = mask1 + mask2

    # Clean up
    #we want to circle our strawberry so we'll circle it with an ellipse
    #with a shape of 15x15
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    #morph the image. closing operation Dilation followed by Erosion. 
    #It is useful in closing small holes inside the foreground objects, 
    #or small black points on the object.
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    #erosion followed by dilation. It is useful in removing noise
    mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

    contours = find_contours(mask_clean)

    for con in contours:
        M = cv2.moments(con)
        centroid = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))
        # move_around(centroid)
        print "CENTROID:", centroid
        epsilon = 0.1 * cv2.arcLength(con, True)
        approx = cv2.approxPolyDP(con, epsilon, True)
        print "SHAPE_APPROXIMATION:"
        print approx
        print "CONTOUR_AREA:", cv2.contourArea(con)
        print "###"
    return


if __name__ == "__main__":
    import sys
    #read the image
    image = cv2.imread(sys.argv[1])
    #detect red obejcts!!
    result = find_red_objects(image)
    #write the new image
    # cv2.imwrite(sys.argv[2], result)

