import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

blue  = (255, 0, 0)
green = (0, 255, 0)
red   = (0, 0, 255)

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def normalize_image(img):
    '''
    normalize image
    '''
    h, w, _ = img.shape
    cxl, cxr = w//2, w//2
    cyl, cyr = h//2, h//2
    # sl = np.power((1/2), 1/2)
    # sr = 


def EstimateFundamentalMatrix(xl, xr):
    '''
    EstimateFundamentalMatrix
    xl = matching points of left image
    xr = matching points of right image
    '''

    if len(xl) != len(xr):
        raise RuntimeError

    A = np.array([0,0,0,0,0,0,0,0,0])
    for i in range(len(xl)):
        A = np.vstack((A, [xl[i][0]*xr[i][0], xl[i][0]*xr[i][1], xl[i][0], xl[i][1]*xr[i][0], xl[i][1]*xr[i][1], xl[i][1], xr[i][0], xr[i][1], 1]))
    A = A[1:]
    # xl1, yl1 = xl[0]
    # xl2, yl2 = xl[1]
    # xl3, yl3 = xl[2]
    # xl4, yl4 = xl[3]
    # xl5, yl5 = xl[4]
    # xl6, yl6 = xl[5]
    # xl7, yl7 = xl[6]
    # xl8, yl8 = xl[7]

    # xr1, yr1 = xr[0]
    # xr2, yr2 = xr[1]
    # xr3, yr3 = xr[2]
    # xr4, yr4 = xr[3]
    # xr5, yr5 = xr[4]
    # xr6, yr6 = xr[5]
    # xr7, yr7 = xr[6]
    # xr8, yr8 = xr[7]

    # A = np.array(
    #     [
    #         [xl1*xr1, xl1*yr1, xl1, yl1*xr1, yl1*yr1, yl1, xr1, yr1, 1],
    #         [xl2*xr2, xl2*yr2, xl2, yl2*xr2, yl2*yr2, yl2, xr2, yr2, 1],
    #         [xl3*xr3, xl3*yr3, xl3, yl3*xr3, yl3*yr3, yl3, xr3, yr3, 1],
    #         [xl4*xr4, xl4*yr4, xl4, yl4*xr4, yl4*yr4, yl4, xr4, yr4, 1],
    #         [xl5*xr5, xl5*yr5, xl5, yl5*xr5, yl5*yr5, yl5, xr5, yr5, 1],
    #         [xl6*xr6, xl6*yr6, xl6, yl6*xr6, yl6*yr6, yl6, xr6, yr6, 1],
    #         [xl7*xr7, xl7*yr7, xl7, yl7*xr7, yl7*yr7, yl7, xr7, yr7, 1],
    #         [xl8*xr8, xl8*yr8, xl8, yl8*xr8, yl8*yr8, yl8, xr8, yr8, 1]
    #     ]
    # )
    # print(A)

    # Note: 
    # The matrix v is transposed in the decomposition. the solution of Ax=0 is the smallest singular vector
    # which is the last column of v not v'. this means that taking the last row of v' which naturally comes out of the svd function is the right thing to do.
    
    # svf of A
    u, s, vt = np.linalg.svd(A)
    # extract smallest singular vector as solution of F
    F = vt[-1:]
    # normalization
    F = F/np.linalg.norm(F)
    # reshape F
    F = np.reshape(F, (3, -1))
    # imposing the rank 2 condition
    u, s, vt = np.linalg.svd(F)
    s[-1] = 0
    F_rank2 = u.dot(np.diag(s).dot(vt))
    # print(F_rank2)
    return F_rank2

def getInlierRANSAC(lmatches, rmatches):
    '''
    getInlierRANSAC
    '''
    iterations = 10
    error_threshold = 0.05
    # final set of inliers
    Si = []
    BestF = None
    # number of inliers
    n = 0
    for i in range(iterations):
        # random sample of 8 points
        random_sample_index = random.sample(range(len(lmatches)), k = 8)
        lsam = lmatches[random_sample_index]
        rsam = rmatches[random_sample_index]
        # calculate fundamental matrix with random sample points
        F = EstimateFundamentalMatrix(lsam, rsam)
        # set of inliers
        S = []
        for i in range(len(lmatches)):
        # for i in range(20):
            xl = np.hstack((lmatches[i], [1])).T
            xr = np.hstack((rmatches[i], [1]))
            error = xl.dot(F).dot(xr)
            error_abs = np.abs(error)
            if error_abs < error_threshold:
                S.append((xl, xr))
            # print(f"iter {i}: number of inliers: {len(S)}")
        if n < len(S):
            n = len(S)
            Si = S
            BestF = F
    print(f"total number of matches: {len(lmatches)}")
    print(f"best inlier set length: {len(Si)}")
    return Si, F


def EssentialMatrixFromFundamentalMatrix(F):
    # print(K)
    E = K.T.dot(F).dot(K)
    return E

def ExtractCameraPose(E):
    u, d, vt = np.linalg.svd(E)
    w = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    r1 = u.dot(w).dot(vt)
    r2 = u.dot(w).dot(vt)
    r3 = u.dot(w.T).dot(vt)
    r4 = u.dot(w.T).dot(vt)
    c1 = np.array([ u[:,2]])
    c2 = np.array([-u[:,2]])
    c3 = np.array([ u[:,2]])
    c4 = np.array([-u[:,2]])
    if np.linalg.det(r1) < 0:
        r1 = -r1 
        r2 = -r2 
        r3 = -r3 
        r4 = -r4 
        c1 = -c1 
        c2 = -c2 
        c3 = -c3 
        c4 = -c4 
    p1 = K.dot(r1).dot(np.concatenate((np.eye(3), c1.T), axis=1))
    p2 = K.dot(r2).dot(np.concatenate((np.eye(3), c2.T), axis=1))
    p3 = K.dot(r3).dot(np.concatenate((np.eye(3), c3.T), axis=1))
    p4 = K.dot(r4).dot(np.concatenate((np.eye(3), c4.T), axis=1))
    return c1, r1


#  __  __    _    ___ _   _ 
# |  \/  |  / \  |_ _| \ | |
# | |\/| | / _ \  | ||  \| |
# | |  | |/ ___ \ | || |\  |
# |_|  |_/_/   \_\___|_| \_|
#

# K = np.array([[1758.23, 0, 977.42], [0, 1758.23, 552.15], [0, 0, 1]])
K = np.array([[1742.11, 0, 804.90], [0, 1742.11, 541.22], [0, 0, 1]])
# K = np.array([[1729.05, 0, -364.24], [0, 1729.05, 552.22], [0, 0, 1]])
# img = cv2.imread("Questions and Inputs\curule\im0.png")
imgl = cv2.imread("Questions and Inputs\octagon\im0.png", 0)
imgr = cv2.imread("Questions and Inputs\octagon\im1.png", 0)
h, w = imgl.shape[:2]
imgl = cv2.resize(imgl, (w//3, h//3))
imgr = cv2.resize(imgr, (w//3, h//3))

orb = cv2.ORB_create()
kpl,desl = orb.detectAndCompute(imgl, None)
kpr,desr = orb.detectAndCompute(imgr, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(desl, desr)
matches = sorted(matches, key=lambda x:x.distance)

imgout = cv2.drawMatches(imgl, kpl, imgr, kpr, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow("display", imgout)


l_matches = []
r_matches = []
for i in range(len(matches)):
    # print(matches[i].queryIdx)
    l_matches.append(np.array(np.array(kpl)[matches[i].queryIdx].pt).astype(int))
    r_matches.append(np.array(np.array(kpr)[matches[i].trainIdx].pt).astype(int))
l_matches = np.array(l_matches)
r_matches = np.array(r_matches)

pts1 = np.int32(l_matches)
pts2 = np.int32(r_matches)
F_opencv, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

# todo
# work on the level of matches 
# not match point coordinates
# final set of inliers should be a set of matches not coordinates
# ez to draw

# Fundamental Matrix
inliers, F = getInlierRANSAC(l_matches, r_matches)
# print(F)

# # opencv F matrix
# F = np.array([[ 3.56792318e-21, -1.09606600e-18,  2.47205773e-16],
#               [ 9.77862806e-19,  1.74030700e-19, -1.00143592e-02],
#               [-2.30216393e-16,  1.00143592e-02,  5.55111512e-15]])

F = F_opencv
print(F)
# Draw epilines
lines1 = cv2.computeCorrespondEpilines(l_matches.reshape(-1,1,2), 1, F)
lines1 = lines1.reshape(-1,3)
img3,img4 = drawlines(imgl,imgr,lines1,l_matches,r_matches)
lines2 = cv2.computeCorrespondEpilines(r_matches.reshape(-1,1,2), 2, F)
lines2 = lines2.reshape(-1,3)
img5,img6 = drawlines(imgl,imgr,lines2,l_matches,r_matches)
# cv2.imshow("left", img4)
# cv2.imshow("right", img5)

# print(inliers[0])
# normalize_image(imgl)

# Essential Matrix
# E = EssentialMatrixFromFundamentalMatrix(F)

# Pose Matrix
# T, R = ExtractCameraPose(E)
# Stereo rectification (uncalibrated variant)
# Adapted from: https://stackoverflow.com/a/62607343
h1, w1 = imgl.shape
h2, w2 = imgr.shape
_, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(pts1), np.float32(pts2), F, imgSize=(w1, h1))


# Undistort (rectify) the images and save them
# Adapted from: https://stackoverflow.com/a/62607343
imgl_rectified = cv2.warpPerspective(imgl, H1, (w1, h1))
imgr_rectified = cv2.warpPerspective(imgr, H2, (w2, h2))
# cv2.imshow("rectified_l", imgl_rectified)
# cv2.imshow("rectified_r", imgr_rectified)

# cv2.imshow("display1", curuleleft)
# cv2.imshow("display2", curuleright)
cv2.waitKey(0)
cv2.destroyAllWindows()
