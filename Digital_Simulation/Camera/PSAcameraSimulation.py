import sys
import locale
import platform
import os
import cv2
import math
import numpy as np

GERMAN_SETUP = True if "de_DE" in locale.getlocale()[0] else False
SEP_CHAR = '/' if not "Windows" in platform.platform() else '\\'
STORAGE_FOLDER = "POV"

ROTATE = False
CAMERA_TO_USE = "IMX708Wide"
PSA_DEGREE = 20.0

# Name: (FOV_H, FOV_V)
CAMERA_FOVS = {
    'IMX708Wide': (102, 67),
    'BoschMono': (120, 54.8),   # https://www.bosch-mobility.com/de/loesungen/kamera/multifunktionskamera/
    "AumovioMono": (110, 46)    # https://amv.re/I38AYQ
}

def parseTuple(tupleAsString: str) -> tuple:
    return tuple(int(elem) for elem in tupleAsString.split(','))

def pixelToAngle(u: int, v: int, W: int, H: int) -> tuple:
    # u,v = x,y of selected point; W,H = width, height of selected image
    lon = (u / W) * 2 * math.pi - math.pi
    lat = (0.5 - v / H) * math.pi
    return (lon, lat)

# Orthonormal camera basic
def basicCamera(lon, lat, roll=0.0):

    # Coordinates: x right, y up, z frontal

    #f = Forward direction aimint at <lon, lat>
    cameraF = np.array([
        math.sin(lon) * math.cos(lat),
        math.sin(lat),
        math.cos(lon) * math.cos(lat)
    ], dtype=np.float64)
    cameraF /= np.linalg.norm(cameraF) + 1e-12

    worldUp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    cameraR = np.cross(worldUp, cameraF)

    # Check straight up/down
    if np.linalg.norm(cameraR) < 1e-9:
        worldUp = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        cameraR = np.cross(worldUp, cameraF)
    cameraR /= np.linalg.norm(cameraR) + 1e-12

    cameraU = np.cross(cameraF, cameraR)
    cameraU /= np.linalg.norm(cameraU) +1e-12

    if abs(roll) > 1e-12:
        cosRoll, sinRoll = np.cos(roll), np.sin(roll)
        cameraR = cosRoll * cameraR + sinRoll * cameraU
        cameraU = -sinRoll * cameraR + cosRoll * cameraU

    return cameraR, cameraU, cameraF


def equirectToRectilin(sphereImage, lonCenterDegree, latCenterDegree, fovHDegree, fovVDegree, outW=None, outH=None, rollDegree=0.0):
    H, W = sphereImage.shape[:2]

    fovH = np.deg2rad(fovHDegree)
    fovV = np.deg2rad(fovVDegree)
    
    if np.deg2rad(180) < fovH  or np.deg2rad(180) < fovV:
            raise ValueError("FOV too large for curvilinear projection. Use < 180 degree.")
    
    lonCenter = np.deg2rad(lonCenterDegree)
    latCenter = np.deg2rad(latCenterDegree)
    roll = np.deg2rad(rollDegree)

    # Choose okayish output size (W/2pi pixels per radian)
    if outW is None and outH is None:
        pixPerRad = W/(2*math.pi)
        outW = int(max(32, round(pixPerRad * fovH)))
        outH = int(max(32, round(outW * (fovV/fovH))))
    if outW is None:
        outW = int(max(32, round(outH * (fovH/fovV))))
    if outH is None:
        outH = int(max(32, round(outW * (fovV/fovH))))

    # Pixel mapped to angles around optical center
    xs = (np.arange(outW) + 0.5) / outW     # 0-1
    ys = (np.arange(outH) + 0.5) / outH     # 0-1
    thetaX = (xs - 0.5) * fovH      # -fovH/2 .. +fovH/2
    thetaY = (0.5-ys) * fovV        # +fovV/2 .. -fovV/2

    # Plane coordinates (rectiliniear pinhole; z=1)
    X = np.tan(thetaX)[None, :]     # Shape (1, outW)
    Y = np.tan(thetaY)[:, None]     # Shape (outH, 1)
    Z = np.ones((outH, outW), dtype=np.float64)

    # Normalize camera rays
    temp = np.sqrt(X*X + Y*Y + 1.0)
    cx = X / temp
    cy = Y / temp
    cz = Z / temp

    # Rotate camera to world
    cameraR, cameraU, cameraF = basicCamera(lonCenter, latCenter, roll)
    # Calculate world from it
    wx = cx * cameraR[0] + cy * cameraU[0] + cz * cameraF[0]
    wy = cx * cameraR[1] + cy * cameraU[1] + cz * cameraF[1]
    wz = cx * cameraR[2] + cy * cameraU[2] + cz * cameraF[2]

    # To spherical
    lon = np.arctan2(wx, wz)    # [-pi, pi]
    wyClamped = np.clip(wy, -1.0, 1.0)
    lat = np.arcsin(wyClamped)  # [-pi/2, pi/2]

    # Map to equirect pixel coordinates
    uMap = (lon / (2 * math.pi) + 0.5) * W
    vMap = (0.5 - lat / math.pi) * H

    # Wrap hor and clamp vert
    uMap = np.mod(uMap, W).astype(np.float32)
    vMap = np.clip(vMap, 0, H - 1).astype(np.float32)

    # Bilinear remapping
    rect = cv2.remap(sphereImage, uMap, vMap, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rect


def main(imagePath: str, dtf_direction: str, centerPoint=(0,0)):
    # Prepare folder for log files
    if not os.path.isdir(STORAGE_FOLDER):
        os.mkdir(STORAGE_FOLDER)

    sphereImage = cv2.imread(imagePath, cv2.IMREAD_COLOR)
    sphereImage = cv2.rotate(sphereImage, cv2.ROTATE_180) if ROTATE else sphereImage    # Rotate 180 degree

    # Check aspect ration to make an educated guess whether this is a spherical panorama
    H, W = sphereImage.shape[:2]
    if abs(W / max(1, H) - 2.0) > 0.15:
        print(f" Aspect ratio {W}x{H} isn't ~2:1. -> No stitched equirectangular image?")
        return
    
    # Fancy GUI if needed
    if centerPoint == (0, 0):
        clicked = {}
        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                clicked['pt'] = (x, y)
                cv2.destroyAllWindows()

        disp = sphereImage.copy()
        disp_small = disp
        win_name = "Click reference point (left click). Press ESC to cancel."
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, min(W, 1600), min(H, 900))
        cv2.setMouseCallback(win_name, on_mouse)
        cv2.imshow(win_name, disp_small)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if 'pt' not in clicked:
            raise RuntimeError("No point selected.")
        u, v = clicked['pt']
        centerPoint = (u, v)
    
    print(centerPoint)

    (lon, lat) = pixelToAngle(centerPoint[0], centerPoint[1], W, H)
    centerPointDegree = (np.rad2deg(lon), np.rad2deg(lat))

    print(centerPointDegree)

    STORAGE_FILENAME = STORAGE_FOLDER + SEP_CHAR + imagePath.split(SEP_CHAR)[-1]
    FoVBenign = equirectToRectilin(sphereImage, centerPointDegree[0], centerPointDegree[1], CAMERA_FOVS[CAMERA_TO_USE][0], CAMERA_FOVS[CAMERA_TO_USE][1])
    cv2.imwrite(STORAGE_FILENAME + "_benign_.jpg", FoVBenign)

    if dtf_direction == 'up':
        centerPointDegree = (centerPointDegree[0], centerPointDegree[1] + PSA_DEGREE)
    elif dtf_direction == 'down':
        centerPointDegree = (centerPointDegree[0], centerPointDegree[1] - PSA_DEGREE)
    elif dtf_direction == 'left':
        centerPointDegree = (centerPointDegree[0] - PSA_DEGREE, centerPointDegree[1])
    elif dtf_direction == 'right':
        centerPointDegree = (centerPointDegree[0] + PSA_DEGREE, centerPointDegree[1])

    FoVDtf = equirectToRectilin(sphereImage, centerPointDegree[0], centerPointDegree[1], CAMERA_FOVS[CAMERA_TO_USE][0], CAMERA_FOVS[CAMERA_TO_USE][1])
    cv2.imwrite(STORAGE_FILENAME + "_psa_" + dtf_direction + "_.jpg", FoVDtf)
    return 0
    


if __name__ == "__main__":
    try:
        if len(sys.argv) == 3:
            main(sys.argv[1], sys.argv[2], (0, 0))
        elif len(sys.argv) == 4:
            centerPoint = parseTuple(sys.argv[3])
            main(sys.argv[1], sys.argv[2], centerPoint)
        else:
            print("Please use the following Syntax: python " + sys.argv[0] + " <spherical image> <PSA direction> [<x0,y0>]")
            print("\nPossible values for <PSA direction>:\n- up\n- down\n- left\n- right")
    except KeyboardInterrupt:
        pass