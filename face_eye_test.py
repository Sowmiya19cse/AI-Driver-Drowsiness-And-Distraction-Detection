import cv2

# Open camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Unable to access camera")
    exit()

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# Load eye detector
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_eye.xml"
)

print("Camera started.")
print("Press Q to close.")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Unable to read camera")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # Make image larger
    display_frame = cv2.resize(
        frame,
        None,
        fx=1.5,
        fy=1.5
    )

    # Grayscale
    gray = cv2.cvtColor(
        display_frame,
        cv2.COLOR_BGR2GRAY
    )

    # Improve contrast
    gray = cv2.equalizeHist(gray)

    # Detect face
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(70, 70)
    )

    if len(faces) == 0:

        cv2.putText(
            display_frame,
            "FACE NOT DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    else:

        cv2.putText(
            display_frame,
            "FACE DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    # Process faces
    for (x, y, w, h) in faces:

        # Face rectangle
        cv2.rectangle(
            display_frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            3
        )

        # Face region
        face_gray = gray[
            y:y + h,
            x:x + w
        ]

        face_color = display_frame[
            y:y + h,
            x:x + w
        ]

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(
            face_gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(20, 20)
        )

        # Draw eyes
        for (ex, ey, ew, eh) in eyes:

            cv2.rectangle(
                face_color,
                (ex, ey),
                (ex + ew, ey + eh),
                (255, 0, 0),
                2
            )

        # Eye count
        cv2.putText(
            display_frame,
            "Eyes: " + str(len(eyes)),
            (x, y + h + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

    # Show camera
    cv2.imshow(
        "AI Driver - Face and Eye Detection",
        display_frame
    )

    # Press Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()