from scipy.spatial import distance
from vcam import vcam,meshGen
import imutils
from imutils import contours
from imutils import perspective
from scipy.interpolate import UnivariateSpline
from scipy.spatial import distance as dist
import cv2
import numpy as np



boundaries = [
    ([17, 15, 100], [50, 56, 200]),
    ([86, 31, 4], [220, 88, 50]),
    ([25, 146, 190], [62, 174, 250]),
    ([103, 86, 65], [145, 133, 128])
]

def _create_LUT_8UC1(x, y):
    """generates a look up table

    Args:
        x ([int]): an array of integers
        y ([int]): an array of integers

    Returns:
        ing: an array of integers
    """
    spl = UnivariateSpline(x, y)
    return spl(range(256))


x = [0, 64, 128, 192, 256]
yInc = [0, 70, 140, 210, 256]
yDec = [0, 30,  80, 120, 192]

incr_ch_lut = _create_LUT_8UC1(x,yInc)
decr_ch_lut = _create_LUT_8UC1(x,yDec)

def cooling(img):
    """ applies a cooling filter to an image that is passed in

    """
    c_r, c_g, c_b = cv2.split(img)
    c_r = cv2.LUT(c_r, incr_ch_lut).astype(np.uint8)
    c_b = cv2.LUT(c_b, decr_ch_lut).astype(np.uint8)
    img = cv2.merge((c_r, c_g, c_b))
    c_b = cv2.LUT(c_b, decr_ch_lut).astype(np.uint8)

    # increase color saturation
    c_h, c_s, c_v = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
    c_s = cv2.LUT(c_s, incr_ch_lut).astype(np.uint8)

    return cv2.cvtColor(cv2.merge((c_h, c_s, c_v)), cv2.COLOR_HSV2RGB)



def warming(img):

    """applies a warming filter to an image

    Returns:
        [type]: [description]
    """
    c_r, c_g, c_b = cv2.split(img)
    c_r = cv2.LUT(c_r, decr_ch_lut).astype(np.uint8)
    c_b = cv2.LUT(c_b, incr_ch_lut).astype(np.uint8)
    img = cv2.merge((c_r, c_g, c_b))

    # decrease color saturation
    c_h, c_s, c_v = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
    c_s = cv2.LUT(c_s, decr_ch_lut).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((c_h, c_s, c_v)), cv2.COLOR_HSV2RGB)


def change_brightness(img, value=-25):
    """changes the brightness of an image
    """
    num_channels = 1 if len(img.shape) < 3 else 1 if img.shape[-1] == 1 else 3
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if num_channels == 1 else img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    if value >= 0:
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
    else:
        value = int(-value)
        lim = 0 + value
        v[v < lim] = 0
        v[v >= lim] -= value

    final_hsv = cv2.merge((h, s, v))

    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if num_channels == 1 else img

    return img


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)






def frameFilter(filter,frame):
    """applies a user selected filter to an image

    Args:
        frame : an image being passed in

    Returns:
        the filtered image
    """

    output = frame
    if filter == "gray":
        output = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return output
    elif filter == "hsv":
        output = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return output
    elif filter == "canny":
            #temp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            r, g, b = cv2.split(frame)
            cr = cv2.Canny(r,100,200)
            cg = cv2.Canny(g,100,200)
            cb = cv2.Canny(b,100,200)
            output = cv2.merge((cr,cg,cb))
            return output
            #output = cv2.Canny(temp,100,200)
    elif filter == "cartoon":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

        # Cartoonization
        color = cv2.bilateralFilter(frame, 9, 250, 250)
        output = cv2.bitwise_and(color, color, mask=edges)
        return output
    elif filter == "negative":
        # Negate the original image
        output = 1 - frame
        return output
    elif filter=="laplace":
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #s = cv2.Laplacian(gray,cv2.CV_64F, ksize = 3)
        r, g, b = cv2.split(frame)
        lr = cv2.Laplacian(r,cv2.CV_64F, ksize = 3)
        lg = cv2.Laplacian(g,cv2.CV_64F, ksize = 3)
        lb = cv2.Laplacian(b,cv2.CV_64F, ksize = 3)
        #cv2.imshow("split laplace",cv2.convertScaleAbs(np.concatenate((lr,lg,lb),axis = 1)))
        #cv2.imshow("split", np.concatenate((r,g,b),axis = 1))
        output = cv2.merge((lr,lg,lb))
        output = cv2.convertScaleAbs(output)
        return output
    elif filter == "xyz":
        output = cv2.cvtColor(frame, cv2.COLOR_BGR2XYZ)
        return output
    elif filter =="hls":
        output = cv2.cvtColor(frame,cv2.COLOR_BGR2HLS)
        return output
    elif filter =="pencil":
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (21,21), 0, 0)
        output = cv2.divide(img_gray, img_blur, scale=256)
        return output
    elif filter =="warm":
        output = warming(frame)
        return output
    elif filter =="cool":
        output = cooling(frame)
        return output
    elif filter == "squares":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray,7)
        arr =np.array([[-1,-1,1],
                       [-1,9,-1],
                       [-1,-1,-1]])
        #arr = arr/sum(arr)
        filt = cv2.filter2D(blur,-1,arr)
        ret,thresh = cv2.threshold(filt,160,255,cv2.THRESH_BINARY)
        kernel = np.ones((5,5),np.uint8)
        #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
        morphed = cv2.morphologyEx(thresh,cv2.MORPH_GRADIENT, kernel)
        contours, hierarchy = cv2.findContours(morphed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        output = cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
        return output
    elif filter == "mirror1":
        H,W = frame.shape[:2]
        # Create a virtual camera object. Here H,W correspond to height and width of the input image frame.
        c1 = vcam(H=H,W=W)
        # Create surface object
        plane = meshGen(H,W)
        # Change the Z coordinate. By default Z is set to 1
        # We generate a mirror where for each 3D point, its Z coordinate is defined as Z = 10*sin(2*pi[x/w]*10)
        plane.Z = 10*np.sin((plane.X/plane.W)*2*np.pi*10)
        # Get modified 3D points of the surface
        pts3d = plane.getPlane()
        # Project the 3D points and get corresponding 2D image coordinates using our virtual camera object c1
        pts2d = c1.project(pts3d)
        # Get mapx and mapy from the 2d projected points
        map_x,map_y = c1.getMaps(pts2d)
        # Applying remap function to input image (img) to generate the funny mirror effect
        output = cv2.remap(frame,map_x,map_y,interpolation=cv2.INTER_LINEAR)
        return output
    elif filter == "mirror2":
        H,W = frame.shape[:2]

        # Creating the virtual camera object
        c1 = vcam(H=H,W=W)

        # Creating the surface object
        plane = meshGen(H,W)

        # We generate a mirror where for each 3D point, its Z coordinate is defined as Z = 20*exp^((x/w)^2 / 2*0.1*sqrt(2*pi))

        plane.Z += 20*np.exp(-0.5*((plane.X*1.0/plane.W)/0.1)**2)/(0.1*np.sqrt(2*np.pi))
        #plane.Z += 20*np.exp(-0.5*((plane.Y*1.0/plane.H)/0.1)**2)/(0.1*np.sqrt(2*np.pi))
        #plane.Z += 20*np.sin(2*np.pi*((plane.X-plane.W/4.0)/plane.W)) + 20*np.sin(2*np.pi*((plane.Y-plane.H/4.0)/plane.H))
        #plane.Z -= 100*np.sqrt((plane.X*1.0/plane.W)**2+(plane.Y*1.0/plane.H)**2)
        pts3d = plane.getPlane()

        pts2d = c1.project(pts3d)
        map_x,map_y = c1.getMaps(pts2d)

        output = cv2.remap(frame,map_x,map_y,interpolation=cv2.INTER_LINEAR)
        return output
    elif filter == "mirror3":
        H,W = frame.shape[:2]

        # Creating the virtual camera object
        c1 = vcam(H=H,W=W)

        # Creating the surface object
        plane = meshGen(H,W)

        # We generate a mirror where for each 3D point, its Z coordinate is defined as Z = 20*exp^((x/w)^2 / 2*0.1*sqrt(2*pi))

        #plane.Z += 20*np.exp(-0.5*((plane.X*1.0/plane.W)/0.1)**2)/(0.1*np.sqrt(2*np.pi))
        plane.Z += 20*np.exp(-0.5*((plane.Y*1.0/plane.H)/0.1)**2)/(0.1*np.sqrt(2*np.pi))
        #plane.Z += 20*np.sin(2*np.pi*((plane.X-plane.W/4.0)/plane.W)) + 20*np.sin(2*np.pi*((plane.Y-plane.H/4.0)/plane.H))
        #plane.Z -= 100*np.sqrt((plane.X*1.0/plane.W)**2+(plane.Y*1.0/plane.H)**2)
        pts3d = plane.getPlane()

        pts2d = c1.project(pts3d)
        map_x,map_y = c1.getMaps(pts2d)

        output = cv2.remap(frame,map_x,map_y,interpolation=cv2.INTER_LINEAR)
        return output
    elif filter == "mirror4":
        H,W = frame.shape[:2]

        # Creating the virtual camera object
        c1 = vcam(H=H,W=W)

        # Creating the surface object
        plane = meshGen(H,W)

        # We generate a mirror where for each 3D point, its Z coordinate is defined as Z = 20*exp^((x/w)^2 / 2*0.1*sqrt(2*pi))

        #plane.Z += 20*np.exp(-0.5*((plane.X*1.0/plane.W)/0.1)**2)/(0.1*np.sqrt(2*np.pi))
        #plane.Z += 20*np.exp(-0.5*((plane.Y*1.0/plane.H)/0.1)**2)/(0.1*np.sqrt(2*np.pi))
        plane.Z += 20*np.sin(2*np.pi*((plane.X-plane.W/4.0)/plane.W)) + 20*np.sin(2*np.pi*((plane.Y-plane.H/4.0)/plane.H))
        #plane.Z -= 100*np.sqrt((plane.X*1.0/plane.W)**2+(plane.Y*1.0/plane.H)**2)
        pts3d = plane.getPlane()

        pts2d = c1.project(pts3d)
        map_x,map_y = c1.getMaps(pts2d)

        output = cv2.remap(frame,map_x,map_y,interpolation=cv2.INTER_LINEAR)
        return output
    elif filter == "mirror5":
        H,W = frame.shape[:2]

        # Creating the virtual camera object
        c1 = vcam(H=H,W=W)

        # Creating the surface object
        plane = meshGen(H,W)

        # We generate a mirror where for each 3D point, its Z coordinate is defined as Z = 20*exp^((x/w)^2 / 2*0.1*sqrt(2*pi))

        #plane.Z += 20*np.exp(-0.5*((plane.X*1.0/plane.W)/0.1)**2)/(0.1*np.sqrt(2*np.pi))
        #plane.Z += 20*np.exp(-0.5*((plane.Y*1.0/plane.H)/0.1)**2)/(0.1*np.sqrt(2*np.pi))
        #plane.Z += 20*np.sin(2*np.pi*((plane.X-plane.W/4.0)/plane.W)) + 20*np.sin(2*np.pi*((plane.Y-plane.H/4.0)/plane.H))
        plane.Z -= 100*np.sqrt((plane.X*1.0/plane.W)**2+(plane.Y*1.0/plane.H)**2)
        pts3d = plane.getPlane()

        pts2d = c1.project(pts3d)
        map_x,map_y = c1.getMaps(pts2d)

        output = cv2.remap(frame,map_x,map_y,interpolation=cv2.INTER_LINEAR)
        return output

    return output
