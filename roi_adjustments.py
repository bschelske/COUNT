import cv2

def callback(x):
    pass

img = cv2.imread(r'C:\Users\bensc\PycharmProjects\scikit\to_image\img\image-001.png', cv2.IMREAD_GRAYSCALE)
img_h, img_w = img.shape

cv2.namedWindow('image')  # make a window with name 'image'
cv2.createTrackbar('x', 'image', 0, img_w, callback)
cv2.createTrackbar('y', 'image', 0, img_h, callback)
cv2.createTrackbar('w', 'image', 0, img_w, callback)
cv2.createTrackbar('h', 'image', 0, img_h, callback)

while True:
    img_copy = img.copy()  # Make a copy of the original image to draw rectangles on

    x = cv2.getTrackbarPos('x', 'image')
    y = cv2.getTrackbarPos('y', 'image')
    w = cv2.getTrackbarPos('w', 'image')
    h = cv2.getTrackbarPos('h', 'image')

    roi = cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('image', img_copy)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # escape key
        break

cv2.destroyAllWindows()
