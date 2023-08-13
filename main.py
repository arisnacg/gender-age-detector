import cv2

# use webcam
cap = cv2.VideoCapture(0)

while True:
    # show camera capture
    ret, frame = cap.read()
    cv2.imshow("Age-Gender-Detector", frame)

    # exit camera capture with pressing "q"
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()