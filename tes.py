import cv2
#print(cv2.__version__)
import time
import numpy as np

cap = cv2.VideoCapture(0)  #Accessing Webcam
_, background = cap.read() #Reading frame
time.sleep(2)
_, background = cap.read()
#initialization
open_kernel = np.ones((5, 5), np.uint8)
close_kernel = np.ones((7, 7), np.uint8)
dilation_kernel = np.ones((10, 10), np.uint8)

#removing edges off the picture
def filter_mask(mask):
    close_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel) #remove black noise from white background
    open_mask = cv2.morphologyEx(close_mask, cv2.MORPH_OPEN, open_kernel) #remove white noise fromclack background
    dilation = cv2.dilate(open_mask, dilation_kernel, iterations= 1) #increase white region in image
    return dilation

while cap.isOpened():
    ret, frame = cap.read()  # Capture every frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #converting BGR to HSV colorspace
    # lower bound and upper bound for Green color
    lower_bound = np.array([50, 80, 50])
    upper_bound = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)  # find the colors within the boundaries
    mask = filter_mask(mask)
    cloak = cv2.bitwise_and(background, background, mask=mask) # Apply the mask  where our cloak is present in the current frame
    inverse_mask = cv2.bitwise_not(mask) # create inverse mask
    current_background = cv2.bitwise_and(frame, frame, mask=inverse_mask) # Apply the inverse mask in current frame where cloak is not present
    combined = cv2.add(cloak, current_background) #combine cloak and current frame background
    cv2.imshow("Final output", combined)

    if cv2.waitKey(1) == ord('q'):  #if 'q' is pressed delete all windows
        break

cap.release()
cv2.destroyAllWindows()

