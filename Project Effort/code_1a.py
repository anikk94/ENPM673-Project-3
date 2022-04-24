from turtle import left, right
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from copy import deepcopy
from tqdm import tqdm
# from math import pow    

def correlation_correspondence(left_image, right_image):
    '''
    inputs are left and right image after rectfication
    match cost is sum of squared difference
    '''
    # vertical difference between corresponding epilines
    v_error = 5
    depth_map = np.zeros_like(right_image)
    # input image parameters
    row, col = left_image.shape[:2]
    # match window size
    size_w = 7
    size_w_long = size_w*10
    step_size = 3
    for v in tqdm(range(250, 330, step_size)):
        for w in range(0, col-size_w_long, step_size):
            row_number = v
            col_number = w
            ssd = 0
            min_ssd = 100000000
            match_location = None
            ssd_arr = []
            ssd_indx = []
            search_window = left_image[v_error+v : v_error+v+size_w , w : w+size_w]
            cv2.imshow("sw", search_window)
            for t in range(0, w+size_w_long-1):
                ssd = 0
                for i in range(size_w):
                    for j in range(size_w):
                        ssd += pow((int(search_window[i,j]) - int(right_image[row_number+i, t+j])),2)
                ssd_arr.append(ssd)
                ssd_indx.append((row_number, t))        
                ssd = 0
            match_location = ssd_indx[ssd_arr.index(min(ssd_arr))]
            match_window = right_image[row_number:row_number+size_w, match_location[1]:match_location[1]+size_w_long]
            cv2.imshow("mw", match_window)
            # disparity
            print(w-match_location[1])
            cv2.waitKey(0)
            # print((row_number, col_number), match_location, "\t", col_number-match_location[1])
            # cv2.imshow("asdf", search_window)
            # cv2.waitKey(0)
            # block wise depth colouring
            disparity = abs(w-match_location[1])
            if disparity == 0:
                # depth_map[v-size_w//2:v+size_w//2, w-size_w//2:w+size_w//2] = 0
                # depth_map[v:v+size_w, w:w+size_w] = 0
                depth_map[v:v+step_size, w:w+step_size] = 0
                # depth_map[v, w] = 0
            else:
                # depth_map[v-size_w//2:v+size_w//2, w-size_w//2:w+size_w//2] = baseline*f/(abs(disparity))
                # depth_map[v:v+size_w, w:w+size_w] = baseline*f/(abs(disparity))
                depth_map[v:v+step_size, w:w+step_size] = baseline*f/((disparity))
                # depth_map[v, w] = baseline*f/(abs(disparity))
            # depth_map[v:v+size_w, w:w+size_w] = calculate_depth(w-match_location[1])
            # depth_map[v:v+size_w, w:w+size_w] = abs(w-match_location[1])
            # pixel wise depth colouring
            # depth_map[v, w] = calculate_depth(w-match_location[1])
        depth_map = np.multiply(depth_map, 255/depth_map.max(), out=depth_map, casting="unsafe")
        cv2.imshow("depth map", depth_map)
        cv2.waitKey(1)
    depth_map = np.multiply(depth_map, 255/depth_map.max(), out=depth_map, casting="unsafe")
    cv2.imshow("depth map", depth_map)
    cv2.waitKey(0)


def feature_correspondence(left_image, right_image, match_point):
    '''
    inputs are left image, right image after rectfication
    and homography transformed point from left image
    '''
    # point 
    p = match_point
    print(p)
    # return
    # if 
 
    # template window size 
    size_w = 11

    # start and end point for drawing a rectangle
    startpt = p[0]-size_w//2, p[1]-size_w//2
    endpt = 1+p[0]+size_w//2, 1+p[1]+size_w//2
    if startpt[0] < 0 or startpt[1] < 0:
        print("startpt out of bounds")
        return
    if endpt[0] < 0 or endpt[1] < 0:
        print("endpt out of bounds")
        return
    # print(startpt)
    # print(endpt)

    # search image
    image = deepcopy(right_image)
    # print("image shape")
    # print(image.shape)

    # template defination using switched x and y for row and col convention 1>0, 0>1
    print(p[1]-size_w//2, 1+p[1]+size_w//2, p[0]-size_w//2, 1+p[0]+size_w//2)
    match_win = left_image[p[1]-size_w//2: 1+p[1]+size_w//2, p[0]-size_w//2: 1+p[0]+size_w//2]

    template = deepcopy(match_win)
    # cv2.imshow("match_win", match_win)
    # cv2.waitKey(0)

    # show image with template highlighted 
    cv2.rectangle(left_image, startpt, endpt, (0,0,255), 1)
    # cv2.imshow("left image", left_image)
    # cv2.resizeWindow('left image', w//3, h//3)
    # cv2.waitKey(0)

    position = p
    # template = match_win
    template_row = template.shape[0]
    template_col = template.shape[1]

    # specific point row referrence for cross correlation
    # centered on top left of kernel window
    # cross correlation value
    cc = 0
    high_cc = 0
    ssd = 0
    low_ssd = 255*255*255*255
    matching_point = None
    # row to be scanned is the y value of the match feature point
    # hence position[1] not position[0]
    img_row = position[1]

    # comparison operation
    # ssd, cc, or something else
    for t in range(20, left_image.shape[1]-size_w):
    # for t in range(size_w-1, size_w):
    # for t in range(100, 102):
        # col being scanned
        img_col = t
        # shift origin
        prcc = img_row - template_row//2
        pccc = img_col - template_col//2
        # imgr_rectified[img_row, img_col] = 0
        for i in range(template_row):
            for j in range(template_col):
                # print(i, j, prcc+i, pccc+j)

                # cross correlation
                # cc += int(template[i, j])*int(image[prcc+i, pccc+j])

                # SSD
                # print(i, j, int(template[i, j]), int(image[prcc+i, pccc+j]),'\t', abs(int(template[i, j]) - int(image[prcc+i, pccc+j])))
                ssd += np.power((int(template[i, j]) - int(image[prcc+i, pccc+j])),2)

        # SSD
        if img_col < position[0]+20 and img_col > position[0]-20:
            if ssd < low_ssd:
                low_ssd = ssd
                matching_point = [img_row, img_col]
                # print(f"{t} update")

        cc = 0
        ssd = 0
        # image = deepcopy(right_image)

    # print(matching_point)
    cv2.circle(image, [matching_point[1], matching_point[0]], 10, (0, 0, 0), 1)

def drawlines(img1,img2,lines,ptsl,ptsr):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,ptsl,ptsr):
        # color = tuple(np.random.randint(0,255,3).tolist())
        line_color = (0,255,0)
        point_color = (255,0,0)
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), line_color,1)
        img1 = cv2.circle(img1,tuple(pt1),2,point_color,-1)
        img2 = cv2.circle(img2,tuple(pt2),2,point_color,-1)
    return img1,img2

def draw_rectangles():
    for i in range(10):
        p = ptsl[i]
        print(p)
        p = np.append(p, 1)
        p = H1.dot(p)
        p = p/p[-1]
        p = p.astype(int)
        print(p)
        startpt = p[0]-10,p[1]-10 
        endpt = p[0]+10,p[1]+10 
        # print(startpt)
        # print(endpt)
        image = cv2.rectangle(imgl_rectified, startpt, endpt, (0,0,255), 2)
    cv2.imshow("asdf", image)

def calculate_depth(disparity):
    if disparity == 0:
        return 0
    return baseline*f/(abs(disparity))


#                 ____    ____       _       _____  ____  _____  
#                |_   \  /   _|     / \     |_   _||_  \\|_   _| 
#                  |  \\//  |      //_\\      |||    | \\\ | |   
#                  | |\\//| |     / ___ \     |||    | |\\\| |   
#                 _| |_\/_| |_  _///   \\\_  _|||_  _| |_\\  |_  
#                |_____||_____||____| |____||_____||_____|\____| 


# input = "curule"
# input = "octagon"
input = "pendulum"
imgl = cv2.imread("Questions and Inputs/" + input + "/im0.png", 0)
imgr = cv2.imread("Questions and Inputs/" + input + "/im1.png", 0)
h, w = imgl.shape[:2]
scale_factor = 3
imgl = cv2.resize(imgl, (w//scale_factor, h//scale_factor))
imgr = cv2.resize(imgr, (w//scale_factor, h//scale_factor))

if input == "curule":
    # curule
    # cam0=[1758.23 0 977.42; 0 1758.23 552.15; 0 0 1]
    # cam1=[1758.23 0 977.42; 0 1758.23 552.15; 0 0 1]
    doffs=0
    baseline=88.39
    width=1920
    height=1080
    ndisp=220
    vmin=55
    vmax=195
    f = 1758.23

if input == "octagon":
    # octagon
    # cam0=[1742.11 0 804.90; 0 1742.11 541.22; 0 0 1]
    # cam1=[1742.11 0 804.90; 0 1742.11 541.22; 0 0 1]
    doffs=0
    baseline=221.76
    width=1920
    height=1080
    ndisp=100
    vmin=29
    vmax=61
    f = 1742.11

if input == "pendulum":
    # pendulum
    # cam0=[1729.05 0 -364.24; 0 1729.05 552.22; 0 0 1]
    # cam1=[1729.05 0 -364.24; 0 1729.05 552.22; 0 0 1]
    doffs=0
    baseline=537.75
    width=1920
    height=1080
    ndisp=180
    vmin=25
    f = 1729.05
    vmax=150

# sift
sift = cv2.SIFT_create()
# get sift features


kp1, des1 = sift.detectAndCompute(imgl,None)
kp2, des2 = sift.detectAndCompute(imgr,None)
# FLANN matcher parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params,search_params)
# find matches
matches = flann.knnMatch(des1,des2,k=2)

# outlier removal using ratio test as per Lowe's paper
ptsl = []
ptsr = []
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        ptsr.append(kp2[m.trainIdx].pt)
        ptsl.append(kp1[m.queryIdx].pt)
ptsl = np.int32(ptsl)
ptsr = np.int32(ptsr)

# calculation of fundamental matrix
F, mask = cv2.findFundamentalMat(ptsl,ptsr,cv2.FM_LMEDS)
# further filtering of inlier points
ptsl = ptsl[mask.ravel()==1]
ptsr = ptsr[mask.ravel()==1]

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines2draw = 100
lines1 = cv2.computeCorrespondEpilines(ptsl.reshape(-1,1,2), 1, F)
lines1 = lines1.reshape(-1,3)
lines1 = lines1[:lines2draw]
imgl_epilines,imgl2 = drawlines(imgl,imgr,lines1,ptsl,ptsr)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(ptsr.reshape(-1,1,2), 2, F)
lines2 = lines2.reshape(-1,3)
lines2 = lines2[:lines2draw]
imgr_epilines,imgr2 = drawlines(imgl,imgr,lines2,ptsl,ptsr)
# cv2.imshow("1",imgl_epilines)
# cv2.imshow("2",imgr_epilines)

# Stereo rectification (uncalibrated variant)
# Adapted from: https://stackoverflow.com/a/62607343
h1, w1 = imgl.shape
h2, w2 = imgr.shape
_, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(ptsl), np.float32(ptsr), F, imgSize=(w1, h1))




# Undistort (rectify) the images and save them
# Adapted from: https://stackoverflow.com/a/62607343
imgl_epilines_rectified = cv2.warpPerspective(imgl_epilines, H1, (w1, h1))
imgr_epilines_rectified = cv2.warpPerspective(imgr_epilines, H2, (w2, h2))
imgl_rectified = cv2.warpPerspective(imgl, H1, (w1, h1))
imgr_rectified = cv2.warpPerspective(imgr, H2, (w2, h2))

ptslH1 = deepcopy(ptsl)
ptsrH2 = deepcopy(ptsr)
oness = np.ones((len(ptslH1), 1))
ptslH1 = np.concatenate((ptslH1, oness), axis=1)
ptsrH2 = np.concatenate((ptsrH2, oness), axis=1)

# warpiing of left and right feature points
for i in range(len(ptsl)):
    ptslH1[i] = H1.dot(ptslH1[i])
    ptslH1[i] = ptslH1[i]/ptslH1[i][-1]
    ptslH1[i] = ptslH1[i].astype(int) 
    ptsrH2[i] = H1.dot(ptsrH2[i])
    ptsrH2[i] = ptsrH2[i]/ptsrH2[i][-1]
    ptsrH2[i] = ptsrH2[i].astype(int)

# # print()
# cv2.imshow("l", imgl)
# cv2.imshow("r", imgr)
# cv2.imshow("rectified_r", imgr_rectified)
# cv2.imshow("rectified_l", imgl_rectified)
cv2.imshow("rectified_r", imgr_epilines_rectified)
cv2.imshow("rectified_l", imgl_epilines_rectified)
cv2.waitKey(1)

correlation_correspondence(imgl_rectified, imgr_rectified)

cv2.waitKey(0)
# cv2.destroyAllWindows()