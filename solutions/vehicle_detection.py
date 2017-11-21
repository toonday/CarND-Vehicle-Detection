import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from skimage.feature import hog
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import pickle
import os
import json
from scipy.ndimage.measurements import label
import random

class Settings():
    if not os.path.isfile('car_detection_settings.ini'):
        raise Exception('Settings file does not exist, create one.')
        
    with open('car_detection_settings.ini', 'r') as configfile:
        settings_str = configfile.read()
    config_settings = json.loads(settings_str)
    
    color_space = config_settings['color_space']
    spatial_size = tuple(config_settings['spatial_size'])
    hist_bin = config_settings['hist_bin']
    hog_resize = tuple(config_settings['hog_resize'])
    hist_range = tuple(config_settings['hist_range'])
    orient = config_settings['orient']
    pix_per_cell = tuple(config_settings['pix_per_cell'])
    cell_per_block = tuple(config_settings['cell_per_block'])
    hog_channel = config_settings['hog_channel']
    x_start_stop = config_settings['x_start_stop']
    y_start_stop = config_settings['y_start_stop']
    xy_window = tuple(config_settings['xy_window'])
    xy_overlap = tuple(config_settings['xy_overlap'])
    cells_per_step = config_settings['cells_per_step']

def draw_boxes(draw_img, bboxes, color=(0, 0, 255), thick=6):
    for bbox in bboxes:
        cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)
    return draw_img

def color_hist(img, nbins=32, bins_range=(0, 256), debug=False):
    rhist = np.histogram(img[:,:,0], bins=nbins)#, range=bins_range)
    ghist = np.histogram(img[:,:,1], bins=nbins)#, range=bins_range)
    bhist = np.histogram(img[:,:,2], bins=nbins)#, range=bins_range)
    
    bin_edges = rhist[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    if debug:
        return rhist, ghist, bhist, bin_centers, hist_features
    else:
        return hist_features

#def bin_spatial(img, color_space='RGB', size=(32, 32)):
def change_color_space(img, color_space):
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = img
    return feature_image

def resize_and_flatten(img, size):
    flattened_img = cv2.resize(img, size).ravel()
    return flattened_img

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=pix_per_cell,
                                  cells_per_block=cell_per_block, transform_sqrt=False, 
                                  visualise=True, feature_vector=False)
        return hog_image, features
    else:
        ###print(" ******************** ", "img.shape: ", img.shape)
        features = hog(img, orientations=orient, pixels_per_cell=pix_per_cell,
                       cells_per_block=cell_per_block, transform_sqrt=False, 
                       visualise=False, feature_vector=feature_vec)
        ###print(" %%%%%%%%%%%%%%%%%%%% ", "features.shape: ", features.shape)
        return features

def extract_features(imgs, cspace, \
                     spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256), \
                     orient=9, pix_per_cell=(8, 8), cell_per_block=(2, 2), hog_channel=0, hog_resize=(32, 32), zero_mean_image=False):
    features = []
    count = -1
    for file in imgs:
        if type(file) == str:
            image = mpimg.imread(file).astype(np.float32)
        elif type(file) == np.ndarray:
            image = file
        else:
            raise ValueError("imgs should be a list of numpy arrays or file names:: "+str(type(file)))
            
        feature_image = image.copy()
        # Normalize Image
        if zero_mean_image:
            feature_image = (feature_image - feature_image.mean()) / (feature_image.max() - feature_image.min())
        else:
            feature_image = (feature_image - feature_image.mean()) / (feature_image.max() - feature_image.min())
            ###### feature_image = feature_image / 255
            ###### feature_image = (feature_image - 128) / 255
        
        # (HOC) Color and positional features
        feature_image = change_color_space(feature_image, cspace)
        spatial_features = resize_and_flatten(feature_image, size=spatial_size)
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
        #features.append(np.concatenate((spatial_features, hist_features)))
        
        # resize image to deal with memory issues
        ###### resized_img = cv2.resize(image, dsize=(0,0), fx=0.5, fy=0.5)
        ###### feature_image = resized_img
        # HOG features
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)        
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        features.append(np.concatenate((spatial_features, hist_features, hog_features)))
    return features

def slide_window(img_shape, x_start_stop=(None, None), y_start_stop=(None, None), 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    x_start_stop = list(x_start_stop)
    y_start_stop = list(y_start_stop)
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img_shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img_shape[0]

    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)
    
    window_list = []
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate each window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
        
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    return window_list

def search_windows(img, windows, clf, color_space, scaler=None, transformer=None, 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, hog_resize=(32, 32), spatial_feat=True, 
                    hist_feat=True, hog_feat=True):
    on_windows = []
    off_windows = []
    heatmap = np.zeros_like(img[:,:,0]).astype(np.float32)
    for window in windows:
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        
        features = extract_features([test_img], cspace=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, hist_range=hist_range,
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, hog_resize=hog_resize)
        
        if scaler is not None:
            test_features = scaler.transform(np.array(features[0]).reshape(1, -1))
        else:
            test_features = np.array(features[0]).reshape(1, -1)
        
        if transformer is not None:
            test_features = transformer.transform(test_features)
            
        prediction = clf.predict(test_features)
        if prediction == 1:
            on_windows.append(window)
            heatmap[window[0][1]:window[1][1], window[0][0]:window[1][0]] += 1
        else:
            off_windows.append(window)
    return on_windows, heatmap, off_windows

def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, color_space, hog_resize, cells_per_step):
    
    draw_img = np.copy(img)
    on_windows = []
    heatmap = np.zeros_like(img[:,:,0]).astype(np.float32)
    
    # Normalize image
    img = img / 255
    #img = (img - img.mean()) / (img.max() - img.min())
    ###### img = (img - 128) / 255
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = change_color_space(img_tosearch, color_space=color_space)
    
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
    
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell[1]) - cell_per_block[1] + 1
    nyblocks = (ch1.shape[0] // pix_per_cell[0]) - cell_per_block[0] + 1 
    nfeat_per_block = orient*cell_per_block[1]**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell[1]) - cell_per_block[1] + 1
    #cells_per_step = 4  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    bboxes = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            
            '''
            hog_feat1 = hog_feat1.ravel().reshape((42, 42))
            hog_feat2 = hog_feat2.ravel().reshape((42, 42))
            hog_feat3 = hog_feat3.ravel().reshape((42, 42))
            '''
            
            '''
            hog_feat1 = cv2.resize(hog_feat1, (18,18)).ravel()
            hog_feat2 = cv2.resize(hog_feat2, (18,18)).ravel()
            hog_feat3 = cv2.resize(hog_feat3, (18,18)).ravel()
            '''
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell[0]
            ytop = ypos*pix_per_cell[1]

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
            
            # Get color features
            spatial_features = resize_and_flatten(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)
            
            '''
            resized_img = cv2.resize(subimg, dsize=(0,0), fx=0.5, fy=0.5)
            ch1 = resized_img[:,:,0]
            ch2 = resized_img[:,:,1]
            ch3 = resized_img[:,:,2]
            hog_feat1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=True)
            hog_feat2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=True)
            hog_feat3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=True)
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            '''

            # Scale features and make a prediction
            if X_scaler is not None:
                test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
                #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            else:
                test_features = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)
                
                bbox = np.zeros((2,2)).astype(np.uint16)
                bbox[0][0] = xbox_left
                bbox[0][1] = ytop_draw+ystart
                bbox[1][0] = xbox_left+win_draw
                bbox[1][1] = ytop_draw+win_draw+ystart
                heatmap[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]] += 1
                
                on_windows.append(bbox)
            '''
            else:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,255,255),2)
            '''
    return draw_img, heatmap, on_windows

def add_heat(heatmap, bbox_list):
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap
    
def apply_threshold(heatmap, threshold):
    heatmap[heatmap <= threshold] = 0
    return heatmap

def draw_labeled_bboxes(img, labels):
    bboxes = []
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        bboxes.append(bbox)
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    return img, bboxes

class FrameHeatMap():
    def __init__(self, shape):
        self.heatmap = np.zeros(shape).astype(np.float32)
        self.counter = 0  
        self.start_val = 3.0
        self.max = 5.0
        self.valid_slope = (0.5, 1.5)
        self.valid_area = (6500, 65000)
        
    def update(self, image, heat_map, car_labels, threshold=3):
        for car_number in range(1, car_labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (car_labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            
            x1 = np.min(nonzerox)
            y1 = np.min(nonzeroy)
            x2 = np.max(nonzerox)
            y2 = np.max(nonzeroy)
            #patch = self.heatmap[y1:y2, x1:x2]
            
            area = (x2-x1) * (y2-y1)
            slope = (y2-y1) / (x2-x1)
            ### print("patch area: ", area)
            ### print("patch slope: ", slope)
            
            if area < self.valid_area[0]:# or area > self.valid_area[1]:
                ### print("ignoring patch area: ", area)
                #cv2.rectangle(img, bbox[0], bbox[1], (0,255,255), 6)
                heat_map[y1:y2, x1:x2] = 0.0
                continue
                
            if slope < self.valid_slope[0] or slope > self.valid_slope[1]:
                ### print("ignoring patch slope: ", slope)
                #cv2.rectangle(img, bbox[0], bbox[1], (255,0,255), 6)
                heat_map[y1:y2, x1:x2] = 0.0
                continue
            
            
        self.heatmap[(heat_map > 0) & (self.heatmap > 0.0)] += 2.0
        self.heatmap[(heat_map > 0) & (self.heatmap <= 0.0)] += self.start_val
        self.heatmap[(self.heatmap > self.max)] = self.max
        self.heatmap[self.heatmap > 0.0] -= 1.0
        
        heatmap_update = apply_threshold(self.heatmap.copy(), threshold)
        labels = label(heatmap_update)
        draw_img, _ = self.draw_labeled_bboxes(image.copy(), labels)
        
        self.heatmap[self.heatmap < 0.0] = 0.0
        self.counter += 1
        return draw_img

    def draw_labeled_bboxes(self, img, labels):
        bboxes = []
        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            
            x1 = np.min(nonzerox)
            y1 = np.min(nonzeroy)
            x2 = np.max(nonzerox)
            y2 = np.max(nonzeroy)
            # Define a bounding box based on min/max x and y
            bbox = ((x1, y1), (x2, y2))
            
            area = (x2-x1) * (y2-y1)
            slope = (y2-y1) / (x2-x1)
            ### print("area: ", area)
            ### print("slope: ", slope)
            
            if area < self.valid_area[0]:# or area > self.valid_area[1]:
                ### print("ignoring area: ", area)
                ### cv2.rectangle(img, bbox[0], bbox[1], (0,255,255), 6)
                self.heatmap[y1:y2, x1:x2] -= 1.0
                continue
                
            if slope < self.valid_slope[0] or slope > self.valid_slope[1]:
                ### print("ignoring slope: ", slope)
                ### cv2.rectangle(img, bbox[0], bbox[1], (255,0,255), 6)
                self.heatmap[y1:y2, x1:x2] -= 1.0
                continue
                
                
            bboxes.append(bbox)
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        return img, bboxes
            
        