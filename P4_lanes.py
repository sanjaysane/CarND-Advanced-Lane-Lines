from scipy import signal
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
import os
import pickle
from scipy.misc import imresize, imread
import matplotlib.image as mpimg

### --------------   CAMERA CALIBRATION --------------   ####

def camera_calibrate(image_path, num_corners, display=False):
    """
     This function expects images for calibration,  
     number of corners to calibrate over
     Returns calibration parameters as a dict{mtx, dist}
    """
    calibrate_path = os.path.join(image_path,'calibration.p')
    if not os.path.exists(calibrate_path):
        glob_pattern = os.path.join(image_path, 'calibration*.jpg')

        obj_points = []
        img_points = []
        image_size = None
        objp = np.zeros((num_corners[0] * num_corners[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:num_corners[1], 0:num_corners[0]].T.reshape(-1, 2)
        images = glob(glob_pattern)
        for idx, fname in enumerate(images):
            img = imread(fname)
            image_size = (img.shape[1], img.shape[0])
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (num_corners[1], num_corners[0]), None)
            if ret:
                obj_points.append(objp)
                img_points.append(corners)
                
                if display == True:
                    cv2.drawChessboardCorners(img, (num_corners[1], num_corners[0]), corners, ret)
                    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
                    ax1.imshow(cv2.cvtColor(mpimg.imread(fname), cv2.COLOR_BGR2RGB))
                    ax1.set_title('Original Image', fontsize=18)
                    ax2.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    ax2.set_title('With Corners', fontsize=18)

        print("Processed(%s) out of (%s) total calibration images" % (len(obj_points), len(images)))
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, image_size, None, None)
        assert ret, 'Camera calibration failed'
        calibration = {'mtx': mtx, 'dist': dist}
    
        with open(calibrate_path, 'wb') as f:
            pickle.dump(calibration, file=f)
    else:
        with open(calibrate_path, "rb") as f:
            calibration = pickle.load(f)

    return calibration

### --------------   IMAGE PROCESSING --------------   ####

def undistort(img, calibration, display = False):
    """
    input calibration parameters as a dict{mtx, dist}
    returns undistorted image
    """
    mtx = calibration['mtx']
    dist = calibration['dist']
    result = cv2.undistort(img, mtx, dist, None, mtx)
    if display == True:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(9,6))
        ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Image', fontsize=20)
        ax2.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        ax2.set_title('Undistorted Image', fontsize=20)
    return result

def get_thresholded_img(img, y_window=0):
    """
    create binary thresholded image
    :param y_window: limit vertical area
    :return: binary mask
    """
    window = img[y_window:, :, :]
    yuv = cv2.cvtColor(window, cv2.COLOR_RGB2YUV)
    yuv = 255 - yuv
    hls = cv2.cvtColor(window, cv2.COLOR_RGB2HLS)
    chs = np.stack((yuv[:, :, 1], yuv[:, :, 2], hls[:, :, 2]), axis=2)
    gray = np.mean(chs, 2)

    # sobel
    axis = (1, 0) 
    s_x = np.absolute(cv2.Sobel(gray, -1, *axis, ksize=3))
    axis = (0, 1)
    s_y = np.absolute(cv2.Sobel(gray, -1, *axis, ksize=3))

    #gradient direction
    with np.errstate(divide='ignore', invalid='ignore'):
        abs_grad_dir = np.absolute(np.arctan(s_y / s_x))
        abs_grad_dir[np.isnan(abs_grad_dir)] = np.pi / 2
    grad_dir = abs_grad_dir.astype(np.float32)

    #gradient magnitude
    abs_grad_mag = np.sqrt(s_x ** 2 + s_y ** 2)
    grad_mag = abs_grad_mag.astype(np.uint16)

    # yellow mask
    hsv = cv2.cvtColor(window, cv2.COLOR_RGB2HSV)
    yellow = cv2.inRange(hsv, (20, 50, 150), (40, 255, 255))

    # highlights
    high_img = window[:, :, 0]
    p = int(np.percentile(high_img, 99.9) - 30)
    highlights = cv2.inRange(high_img, p, 255)

    mask = np.zeros(img.shape[:-1], dtype=np.uint8)

    mask[y_window:, :][((s_x >= 25) & (s_x <= 255) &
                        (s_y >= 25) & (s_y <= 255)) |
                       ((grad_mag >= 30) & (grad_mag <= 512) &
                        (grad_dir >= 0.2) & (grad_dir <= 1.)) |
                       (yellow == 255) |
                       (highlights == 255)] = 1

    # noise reduction by applying a filter over neighbors
    k = np.array([[1, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    nb_neighbors = cv2.filter2D(mask, ddepth=-1, kernel=k)
    mask[nb_neighbors < 4] = 0

    return mask

def get_birds_eye_ROI(img):
    """
    returns the src, dst and ROI set of vertices that can 
    be used for birds_eye transform
    
    img_shape = img.shape
    src = np.float32([
                    (img_shape[1] * 0.0, img_shape[0] * 1.0),
                    (img_shape[1] * 0.4, img_shape[0] * 0.65),
                    (img_shape[1] * 0.6, img_shape[0] * 0.65),
                    (img_shape[1] * 1.0, img_shape[0] * 1.0)])
    dst = np.float32([
                    (img_shape[1] * 0.0, img_shape[0] * 1.0),
                    (img_shape[1] * 0.0, img_shape[0] * 0.0),
                    (img_shape[1] * 1.0, img_shape[0] * 0.0),
                    (img_shape[1] * 1.0, img_shape[0] * 1.0)])
    """
    x_offset = 250
    src = np.float32([
                    (132, 703),
                    (540, 466),
                    (740, 466),
                    (1147, 703)])

    dst = np.float32([
                    (src[0][0] + x_offset, 720),
                    (src[0][0] + x_offset, 0),
                    (src[-1][0] - x_offset, 0),
                    (src[-1][0] - x_offset, 720)])
    ROI = src
    return src, dst, ROI

### --------------   LANE DETECTION --------------   ####

def detect_lanes(img, num_steps, search_area, window):
    """
    detect lane line pixels by applying a sliding histogram.
    :param img: input binary image
    :param num_steps: number of steps used for sliding histogram
    :param search_area: Area to limit horizontal search space.
    :param window: window size for histogram median smoothing
    :return: x, y of pixels that belong to lanes
    """
    all_x = []
    all_y = []
    masked_img = img[:, search_area[0]:search_area[1]]
    pixels_per_step = img.shape[0] // num_steps

    for i in range(num_steps):
        start = masked_img.shape[0] - (i * pixels_per_step)
        end = start - pixels_per_step
        histogram = np.sum(masked_img[end:start, :], axis=0)
        histogram_median = signal.medfilt(histogram, window)
        peaks = np.array(signal.find_peaks_cwt(histogram_median, np.arange(1, 5)))

        top_peak = top_n_peaks(histogram_median, peaks, n=1, threshold=5)
        if len(top_peak) == 1:
            top_peak = top_peak[0]
            center = (start + end) // 2
            x, y = get_pixel_in_area(masked_img, top_peak, center, pixels_per_step)
            all_x.extend(x)
            all_y.extend(y)

    all_x = np.array(all_x) + search_area[0]
    all_y = np.array(all_y)

    return all_x, all_y

def top_n_peaks(histogram, peaks, n=2, threshold=0):
    """
    return the top n peaks of a histogram above a given threshold.
    :param histogram: input histogram
    :param peaks: input list of peak indexes
    :param n: top n peaks to select
    :param threshold: cutoff point
    :return:
    """
    if len(peaks) == 0:
        return []

    peak_list = [(peak, histogram[peak]) for peak in peaks if histogram[peak] > threshold]
    peak_list = sorted(peak_list, key=lambda x: x[1], reverse=True)

    if len(peak_list) == 0:
        return []

    x, y = zip(*peak_list)
    x = list(x)

    if len(peak_list) < n:
        return x

    return x[:n]

def get_pixel_in_area(img, x_center, y_center, size):
    """
    return selected pixel inside the search area.
    :param img: input binary image
    :param x_center: x coordinate of the search area center
    :param y_center: y coordinate of the search area center
    :param size: size of the search area in pixels
    :return: x, y of detected pixels
    """
    half_size = size // 2
    window = img[y_center - half_size:y_center + half_size, x_center - half_size:x_center + half_size]

    x, y = (window.T == 1).nonzero()

    x = x + x_center - half_size
    y = y + y_center - half_size

    return x, y

def detect_lane_along_poly(img, poly, num_steps):
    """
    slide a window along a polynomial and select all pixels inside.
    :param img: binary image
    :param poly: polynomial to follow
    :param num_steps: number of steps for the sliding window
    :return: x, y of detected pixels
    """
    pixels_per_step = img.shape[0] // num_steps
    all_x = []
    all_y = []

    for i in range(num_steps):
        start = img.shape[0] - (i * pixels_per_step)
        end = start - pixels_per_step

        center = (start + end) // 2
        x = poly(center)

        x, y = get_pixel_in_area(img, x, center, pixels_per_step)

        all_x.extend(x)
        all_y.extend(y)

    return all_x, all_y

 
### --------------   Line Processing --------------   ####


class Line:
    def __init__(self, num_frames=1, x=None, y=None):
        """
        Object for each detected line
        :param num_frames: number of frames for smoothing
        :param x: initial x coordinates
        :param y: initial y coordinates
        """
        # last frames saved in object
        self.num_frames = num_frames
        self.detected = False
        self.pixels_per_frame = []
        
    # x values of the last frames
        self.last_x = []
        self.avg_x = None
        
    # polynomial fit over last num frames
        self.current_polyfit = None
        self.avg_polyfit = None
        self.current_poly1d = None
        self.avg_poly1d = None
        
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        self.radius_of_curvature = None
        
    # x and y values for all line pixels
        self.allx = None
        self.ally = None

        if x is not None:
            self.update(x, y)

    def update(self, x, y):
        """
        update the line representation.
        :param x: list of x values
        :param y: list of y values
        """
        self.allx = x
        self.ally = y
        self.pixels_per_frame.append(len(self.allx))
        self.last_x.extend(self.allx)

        if len(self.pixels_per_frame) > self.num_frames:
            n_x_to_remove = self.pixels_per_frame.pop(0)
            self.last_x = self.last_x[n_x_to_remove:]
        self.avg_x = np.mean(self.last_x)
        self.current_polyfit = np.polyfit(self.allx, self.ally, 2)
        if self.avg_polyfit is None:
            self.avg_polyfit = self.current_polyfit
        else:
            self.avg_polyfit = (self.avg_polyfit * (self.num_frames - 1) + self.current_polyfit) / self.num_frames
        self.current_poly1d = np.poly1d(self.current_polyfit)
        self.avg_poly1d = np.poly1d(self.avg_polyfit)
    

def get_lane_area(lines, area_height, num_steps):
    """
    get the area between two lines as a set of points
    :param lanes: list of lines. 
    :param area_height:
    :param num_steps:
    :return:
    """
    points_left = np.zeros((num_steps + 1, 2))
    points_right = np.zeros((num_steps + 1, 2))
    for i in range(num_steps + 1):
        pixels_per_step = area_height // num_steps
        start = area_height - i * pixels_per_step
        points_left[i] = [lines[0].avg_poly1d(start), start]
        points_right[i] = [lines[1].avg_poly1d(start), start]
    return np.concatenate((points_left, points_right[::-1]), axis=0)


### --------------   UTILS and Drawing --------------   ####

def outlier_removal(x, y, q=5):
    """
    remove horizontal outliers based on a given percentile.
    :param x: x coordinates of pixels
    :param y: y coordinates of pixels
    :param q: percentile
    :return: cleaned coordinates (x, y)
    """
    if len(x) == 0 or len(y) == 0:
        return x, y

    x = np.array(x)
    y = np.array(y)

    lower_bound = np.percentile(x, q)
    upper_bound = np.percentile(x, 100 - q)
    selection = (x >= lower_bound) & (x <= upper_bound)
    return x[selection], y[selection]

def draw_area(img, area, num_steps, color):
    """
    draw a polygonal area onto an image - lines on sides of lane area
    :param img: 
    :param area: set of points of the area
    :param num_steps: number of points
    :param color: color of the lines to be drawn 
    :return: img: which has the area drawn over it
    """
    img_height = img.shape[0]
    pixels_per_step = img_height // num_steps
    for i in range(num_steps):
        start = i * pixels_per_step
        end = start + pixels_per_step
        start_point = (int(area(start)), start)
        end_point = (int(area(end)), end)
        img = cv2.line(img, end_point, start_point, color, 10)
    return img

def get_curvature(coeffs):
    """
    get the curvature of a line in meters
    :param polyfit: polyfit coefficients
    :return: radius of curvature in meters
    """
    # conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meteres per pixel in x dimension
    y = np.array(np.linspace(0, 719, num=10))
    x = np.array([coeffs(x) for x in y])
    y_eval = np.max(y)
    polyfit = np.polyfit(y * ym_per_pix, x * xm_per_pix, 2)
    curvature = ((1 + (2 * polyfit[0] * y_eval + polyfit[1]) ** 2) ** 1.5) / np.absolute(2 * polyfit[0])

    return curvature

def draw_2_images(images_left, images_right, title_left, title_right, filename):
    """
    plot 2 set of images next to each other, also save file
    """
    plt.rcParams.update({'font.size': 8})
    plt.figure(figsize=(10, 5))
    num_rows = len(images_left)
    image_num = 1
    for i in range(num_rows):
        plt.subplot(num_rows, 2, image_num)
        plt.imshow(images_left[i])
        plt.axis('on')
        if i == 0: plt.title(title_left)
        image_num += 1
        plt.subplot(num_rows, 2, image_num)
        plt.imshow(images_right[i])
        plt.axis('on')
        if i == 0:  plt.title(title_right)
        image_num += 1
    plt.savefig(filename, bbox_inches='tight')

def draw_dashboard(img, curvature, car_offset):
    """
    display dashboard with information like lane curvature and car's center offset. 
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "Radius of curvature is {:.2f}m".format(curvature)
    cv2.putText(img, text, (50, 50), font, 1, (255, 255, 0), 3)
    left_or_right = 'left' if car_offset < 0 else 'right'
    text = "Car is {:.2f}m {} of center".format(np.abs(car_offset), left_or_right)
    cv2.putText(img, text, (50, 120), font, 1, (255, 255, 0), 3)

def draw_lane(img, left_line, right_line):
    """
    draw the lane area onto the image
    """
    overlay = np.zeros([img.shape[0], img.shape[1], img.shape[2]])
    mask = np.zeros([img.shape[0], img.shape[1]])

    # lane area
    lane_area = get_lane_area((left_line, right_line), img.shape[0], 20)
    mask = cv2.fillPoly(mask, np.int32([lane_area]), 1)
    src, dst, ROI= get_birds_eye_ROI(img)
    M_inv = cv2.getPerspectiveTransform(dst,src) 
    mask = cv2.warpPerspective(mask, M_inv, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_LINEAR)

    overlay[mask == 1] = (128, 255, 0)
    selection = (overlay != 0)
    img[selection] = img[selection] * 0.5 + overlay[selection] * 0.5

    # side lines 
    mask[:] = 0
    mask = draw_area(mask, left_line.avg_poly1d, 5, 255)
    mask = draw_area(mask, right_line.avg_poly1d, 5, 255)
    mask = cv2.warpPerspective(mask, M_inv, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_LINEAR)
    img[mask == 255] = (255, 200, 2)


### --------------   Lane Process --------------   ####
    
class LaneProcess:

    def __init__(self):
        self.num_frames = 7
        self.line_segments = 10
        self.image_offset = 250
        self.left_line = None
        self.right_line = None
        self.center_poly1d = None
        self.curvature = 0.0
        self.car_offset = 0.0
        self.dists = []
        self.calibration = camera_calibrate('camera_cal', (9, 6), display=True)

    def is_line_plausible(self, left, right):
        """
            determine if pixels describing two line are plausible lane lines based on curvature and distance.
            :param left: Tuple of arrays containing the coordinates of detected pixels
            :param right: Tuple of arrays containing the coordinates of detected pixels
            :return:
            """
        parallel_threshold=(0.0003, 0.55)
        dist_threshold=(350, 460)

        if len(left[0]) < 3 or len(right[0]) < 3:
            return False
        else:
            # check for parallelism of two lines and distance
            line1 = Line(y=left[0], x=left[1])
            line2 = Line(y=right[0], x=right[1])
            diff1 = np.abs(line1.current_polyfit[0] - line2.current_polyfit[0])
            diff2 = np.abs(line1.current_polyfit[1] - line2.current_polyfit[1])
            is_parallel = diff1 < parallel_threshold[0] and diff2 < parallel_threshold[1]
            distance = np.abs(line1.current_poly1d(719) - line2.current_poly1d(719))
            is_plausible_dist = dist_threshold[0] < distance < dist_threshold[1]
            return is_parallel & is_plausible_dist

    def validate_lines(self, left_x, left_y, right_x, right_y):
        """
        compare two line to each other and to their last prediction.
        :param left_x:
        :param left_y:
        :param right_x:
        :param right_y:
        :return: boolean tuple (left_detected, right_detected)
        """
        left_detected = False
        right_detected = False

        if self.is_line_plausible((left_x, left_y), (right_x, right_y)):
            left_detected = True
            right_detected = True
        elif self.left_line is not None and self.right_line is not None:
            if self.is_line_plausible((left_x, left_y), (self.left_line.allx, self.left_line.ally)):
                left_detected = True
            if self.is_line_plausible((right_x, right_y), (self.right_line.ally, self.right_line.allx)):
                right_detected = True
        return left_detected, right_detected        
        
        
    def process_image(self,img):

        orig_img = np.copy(img)

        # undistort frame
        img = undistort(img, self.calibration, display=False)

        # apply sobel and color transforms to create a thresholded binary image.
        img = get_thresholded_img(img, 400)

        # apply perspective transform to get birds-eye view
        src, dst, ROI= get_birds_eye_ROI(orig_img)
        M = cv2.getPerspectiveTransform(src,dst)
        img = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

        left_detected = right_detected = False
        left_x = left_y = right_x = right_y = []

        # if lanes were already found in previous images, search around that area
        if self.left_line is not None and self.right_line is not None:
            left_x, left_y = detect_lane_along_poly(img, self.left_line.avg_poly1d, self.line_segments)
            right_x, right_y = detect_lane_along_poly(img, self.right_line.avg_poly1d, self.line_segments)
            left_detected, right_detected = self.validate_lines(left_x, left_y, right_x, right_y)

        # if no lanes were found a histogram search is performed
        if not left_detected:
            left_x, left_y = detect_lanes(img, self.line_segments, 
                    (self.image_offset, img.shape[1] // 2), window=7)
            left_x, left_y = outlier_removal(left_x, left_y)

        if not right_detected:
            right_x, right_y = detect_lanes(img, self.line_segments, 
                    (img.shape[1] // 2, img.shape[1] - self.image_offset), window=7)
            right_x, right_y = outlier_removal(right_x, right_y)

        if not left_detected or not right_detected:
            left_detected, right_detected = self.validate_lines(left_x, left_y, right_x, right_y)

        # updated left lane information
        if left_detected:
            # switch x and y since lines are almost vertical
            if self.left_line is not None:
                self.left_line.update(y=left_x, x=left_y)
            else:
                self.left_line = Line(self.num_frames, left_y, left_x)

        # updated right lane information.
        if right_detected:
            # switch x and y since lines are almost vertical
            if self.right_line is not None:
                self.right_line.update(y=right_x, x=right_y)
            else:
                self.right_line = Line(self.num_frames, right_y, right_x)

        # add calculated information onto the frame
        if self.left_line is not None and self.right_line is not None:
            self.dists.append(np.abs(self.left_line.avg_poly1d(719) - self.right_line.avg_poly1d(719)))
            self.center_poly1d = (self.left_line.avg_poly1d + self.right_line.avg_poly1d) / 2
            self.curvature = get_curvature(self.center_poly1d)
            self.car_offset = (img.shape[1] / 2 - self.center_poly1d(719)) * 3.7 / 700
            draw_lane(orig_img,self.left_line,self.right_line)
            draw_dashboard(orig_img,self.curvature,self.car_offset)

        return orig_img
    
