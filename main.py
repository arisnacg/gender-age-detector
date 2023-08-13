import cv2

# import face detector model
faceProto = "./models/opencv_face_detector.pbtxt"
faceModel = "./models/opencv_face_detector_uint8.pb"
faceNet = cv2.dnn.readNet(faceModel, faceProto)

# func to detect and draw reactangle to the face
def drawFaceRectangle(faceNet, frame):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    boxes = []
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detection[0, 0, i, 3] * frameWidth)
            y1 = int(detection[0, 0, i, 4] * frameHeight)
            x2 = int(detection[0, 0, i, 5] * frameWidth)
            y2 = int(detection[0, 0, i, 6] * frameHeight)
            boxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
    return frame, boxes

# use webcam
cap = cv2.VideoCapture(0)


while True:
    # show camera capture
    ret, frame = cap.read()
    drawFaceRectangle(faceNet, frame)
    cv2.imshow("Age-Gender-Detector", frame)
    # exit camera capture with pressing "q"
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()