import cv2

# padding for face rectangle
PADDING = 20

# import face detector model
faceProto = "./models/opencv_face_detector.pbtxt"
faceModel = "./models/opencv_face_detector_uint8.pb"
faceNet = cv2.dnn.readNet(faceModel, faceProto)

# import gender model
genderProto = "./models/gender_deploy.prototxt"
genderModel = "./models/gender_net.caffemodel"
genderNet = cv2.dnn.readNet(genderModel, genderProto)
genders = ["Male", "Female"]

# import age model
ageProto = "./models/age_deploy.prototxt"
ageModel = "./models/age_net.caffemodel"
ageNet = cv2.dnn.readNet(ageModel, ageProto)
ages = [
    "(0-2)",
    "(4-6)",
    "(8-12)",
    "(15-20)",
    "(25-32)",
    "(38-43)",
    "(48-53)",
    "(60-100)",
]

# model means
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

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

# func to detect gender
def detectGender(genderNet,blob):
    genderNet.setInput(blob)
    genderPrediction = genderNet.forward()
    gender = genders[genderPrediction[0].argmax()]
    return gender

# func to detect age
def detectAge(ageNet,blob):
    ageNet.setInput(blob)
    agePrediction = ageNet.forward()
    age = ages[agePrediction[0].argmax()]
    return age


# use webcam
cap = cv2.VideoCapture(0)

while True:
    # show camera capture
    ret, frame = cap.read()
    frame, boxes = drawFaceRectangle(faceNet, frame)
    for box in boxes:
        detectedFace = frame[
            max(0, box[1] - PADDING) : min(box[3] + PADDING, frame.shape[0] - 1),
            max(0, box[0] - PADDING) : min(box[2] + PADDING, frame.shape[1] - 1),
        ]
        blob = cv2.dnn.blobFromImage(
            detectedFace, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False
        )
        gender = detectGender(genderNet,blob)
        age = detectAge(ageNet, blob)
        # debug
        print(age)


    cv2.imshow("Age-Gender-Detector", frame)
    # exit camera capture with pressing "q"
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()