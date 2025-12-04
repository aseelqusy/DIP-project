import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 960)
cap.set(4, 540)

def cartoonize(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9,9)
    color = img.copy()
    for _ in range(2):
        color = cv2.bilateralFilter(color, 9, 75, 75)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def sobel_image(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx*sx + sy*sy)
    mag = np.uint8(255 * (mag / (np.max(mag)+1e-6)))
    _, th = cv2.threshold(mag, 50, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)

def draw_histogram(gray, width, height):
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist = cv2.normalize(hist, hist, 0, height, cv2.NORM_MINMAX).flatten()
    hist_img = np.ones((height, width, 3), dtype=np.uint8) * 30
    bin_w = width // 256 if width//256>0 else 1
    for i in range(256):
        x = i * bin_w
        cv2.rectangle(hist_img, (x, height), (x+bin_w, height - int(hist[i])), (200,200,200), -1)
    return hist_img

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (960,540))
    original = frame.copy()
    smooth = cv2.GaussianBlur(frame, (9,9), 0)
    cartoon = cartoonize(frame)
    sobel = sobel_image(frame)

    h = 260
    w = 320
    o_r = cv2.resize(original, (w,h))
    s_r = cv2.resize(smooth, (w,h))
    c_r = cv2.resize(cartoon, (w,h))

    top = np.hstack((o_r, s_r, c_r))

    gray_for_hist = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    hist = draw_histogram(gray_for_hist, top.shape[1], 200)

    canvas = np.vstack((top, hist))

    cv2.imshow('Live - 3 Panels + Histogram', canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('s'):
        cv2.imwrite('home_image.png', canvas)
        print('Saved home_image.png')

cap.release()
cv2.destroyAllWindows()