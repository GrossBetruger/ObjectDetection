from __future__ import division

import json

import cv2
import numpy as np
import math

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

    height, width, channels = image.shape

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
    final_result = []
    for con in contours:
        M = cv2.moments(con)
        centroid = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))
        x, y = centroid
        normalized_centriod = normalize(x, y, width, height)
        # move_around(centroid)
        # print "CENTROID:", centroid
        epsilon = 0.1 * cv2.arcLength(con, True)
        approx = cv2.approxPolyDP(con, epsilon, True).tolist()
        # print "SHAPE_APPROXIMATION:"
        # print approx
        contour_area = cv2.contourArea(con)
        # print "CONTOUR_AREA:", contour_area
        # print "###"
        jsn_res = {"centroid": centroid, "normalized_centroid": normalized_centriod,
                   "shape": approx, "contour_area": contour_area}
        print json.dumps(jsn_res)
        final_result.append(jsn_res)

    # print json.dumps(final_result)
    return final_result


def distance(x, y):
    return math.fabs(x - y)


def normalize(x, y, width, height):
    return float(x)/width, float(y)/height


def get_most_centered(results, width, height):

    distances = []
    for res in results:
        x, y = res["centroid"]
        nx, ny = normalize(x, y, width, height)
        distances.append([distance(nx, ny), (x, y), (nx, ny)])
    return sorted(distances, key=lambda a: -a[0])

if __name__ == "__main__":
    import sys
    #read the image
    image = cv2.imread(sys.argv[1])
    height, width, channels = image.shape
    print "dimensions", width, height
    #detect red obejcts!!
    result = find_red_objects(image)
    # print get_most_centered(result, width, height)

    #write the new image
    # cv2.imwrite(sys.argv[2], result)

