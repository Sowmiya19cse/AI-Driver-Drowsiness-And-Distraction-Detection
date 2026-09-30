import cv2

# --------------------------------------------------
# Load Haar Cascade Models
# --------------------------------------------------

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_eye.xml"
)

# Check models
if face_cascade.empty():
    print("ERROR: Face detection model not loaded.")
    exit()

if eye_cascade.empty():
    print("ERROR: Eye detection model not loaded.")
    exit()

# --------------------------------------------------
# Open Webcam
# --------------------------------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Cannot access camera.")
    exit()

print("Camera started.")
print("Press Q to stop.")

# --------------------------------------------------
# Live Video
# --------------------------------------------------

while True:

    ret, frame = camera.read()

    if not ret:
        print("ERROR: Cannot read camera frame.")
        break

    # Flip camera horizontally
    frame = cv2.flip(frame, 1)

    # Convert to grayscale
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------
    # Face Detection
    # --------------------------------------------------

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

    if len(faces) == 0:

        cv2.putText(
            frame,
            "FACE NOT DETECTED",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    # --------------------------------------------------
    # Process Face
    # --------------------------------------------------

    for (x, y, w, h) in faces:

        # Draw face rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "FACE DETECTED",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # --------------------------------------------------
        # Eye Region
        # --------------------------------------------------

        eye_gray = gray[
            y:y + int(h * 0.60),
            x:x + w
        ]

        eye_color = frame[
            y:y + int(h * 0.60),
            x:x + w
        ]

        # --------------------------------------------------
        # Eye Detection
        # --------------------------------------------------

        eyes = eye_cascade.detectMultiScale(
            eye_gray,
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=(20, 20)
        )

        # Limit to two eyes
        eyes = eyes[:2]

        if len(eyes) == 0:

            cv2.putText(
                frame,
                "EYES NOT DETECTED",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        else:

            cv2.putText(
                frame,
                f"EYES DETECTED: {len(eyes)}",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2
            )

        # --------------------------------------------------
        # Draw Eye Rectangles
        # --------------------------------------------------

        for (ex, ey, ew, eh) in eyes:

            cv2.rectangle(
                eye_color,
                (ex, ey),
                (ex + ew, ey + eh),
                (255, 0, 0),
                2
            )

    # --------------------------------------------------
    # Display Live Video
    # --------------------------------------------------

    cv2.imshow(
        "AI Driver Monitoring - Live Camera",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# --------------------------------------------------
# Release Camera
# --------------------------------------------------

camera.release()

cv2.destroyAllWindows()

print("Camera stopped.")