import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import winsound


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Driver Monitoring",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #eef5ff, #f8fbff);
}

.main-title {
    text-align: center;
    font-size: 38px;
    font-weight: 800;
    color: #12355b;
    margin-top: 20px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #5d6b7a;
    margin-bottom: 30px;
}

.welcome-text {
    text-align: center;
    font-size: 18px;
    color: #425466;
    margin-bottom: 30px;
}

.feature-card {
    background: white;
    padding: 28px;
    border-radius: 18px;
    box-shadow: 0px 5px 18px rgba(0,0,0,0.08);
    min-height: 270px;
    border: 1px solid #e2e8f0;
}

.feature-title {
    font-size: 24px;
    font-weight: 700;
    color: #12355b;
    margin-bottom: 15px;
}

.feature-text {
    font-size: 16px;
    line-height: 1.7;
    color: #4b5d70;
}

.alert-text {
    font-weight: 700;
    color: #c62828;
}

.login-box {
    background: white;
    padding: 40px;
    border-radius: 20px;
    box-shadow: 0px 6px 25px rgba(0,0,0,0.10);
}

.camera-title {
    text-align: center;
    color: #12355b;
    font-size: 32px;
    font-weight: 800;
}

.status-box {
    padding: 15px;
    border-radius: 12px;
    background: white;
    text-align: center;
    font-size: 18px;
    font-weight: 600;
    box-shadow: 0px 3px 12px rgba(0,0,0,0.08);
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "login"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


# =========================================================
# MEDIAPIPE
# =========================================================

mp_face_mesh = mp.solutions.face_mesh


# =========================================================
# LANDMARK POINTS
# =========================================================

# Right eye
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Left eye
LEFT_EYE = [362, 385, 387, 263, 373, 380]

# Mouth
MOUTH = [61, 81, 13, 311, 291, 308, 14, 178]


# =========================================================
# FUNCTIONS
# =========================================================

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def calculate_ear(landmarks, eye_points, frame_width, frame_height):

    points = []

    for index in eye_points:
        x = int(landmarks[index].x * frame_width)
        y = int(landmarks[index].y * frame_height)
        points.append((x, y))

    vertical_1 = distance(points[1], points[5])
    vertical_2 = distance(points[2], points[4])
    horizontal = distance(points[0], points[3])

    if horizontal == 0:
        return 0

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)

    return ear


def calculate_mar(landmarks, frame_width, frame_height):

    p61 = (
        int(landmarks[61].x * frame_width),
        int(landmarks[61].y * frame_height)
    )

    p291 = (
        int(landmarks[291].x * frame_width),
        int(landmarks[291].y * frame_height)
    )

    p81 = (
        int(landmarks[81].x * frame_width),
        int(landmarks[81].y * frame_height)
    )

    p178 = (
        int(landmarks[178].x * frame_width),
        int(landmarks[178].y * frame_height)
    )

    p13 = (
        int(landmarks[13].x * frame_width),
        int(landmarks[13].y * frame_height)
    )

    p14 = (
        int(landmarks[14].x * frame_width),
        int(landmarks[14].y * frame_height)
    )

    horizontal = distance(p61, p291)

    vertical_1 = distance(p81, p178)
    vertical_2 = distance(p13, p14)

    if horizontal == 0:
        return 0

    mar = (vertical_1 + vertical_2) / (2.0 * horizontal)

    return mar


# =========================================================
# ALARM
# =========================================================

def play_alarm():

    try:
        winsound.Beep(1000, 500)
    except:
        pass


# =========================================================
# LOGIN PAGE
# =========================================================

def login_page():

    st.markdown(
        '<div class="main-title">🚗 AI Driver Monitoring</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Intelligent Driver Safety System</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 1.2, 1])

    with col2:

        st.markdown('<div class="login-box">', unsafe_allow_html=True)

        st.markdown(
            "<h2 style='text-align:center;color:#12355b;'>Login</h2>",
            unsafe_allow_html=True
        )

        username = st.text_input(
            "Username",
            placeholder="Enter your username"
        )

        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter your password"
        )

        if st.button("LOGIN", use_container_width=True):

            if username == "admin" and password == "admin":

                st.session_state.logged_in = True
                st.session_state.page = "dashboard"
                st.rerun()

            else:

                st.error("Invalid username or password.")

        st.markdown("</div>", unsafe_allow_html=True)

        st.info("Demo Login: Username = admin | Password = admin")


# =========================================================
# DASHBOARD PAGE
# =========================================================

def dashboard_page():

    st.markdown(
        '<div class="main-title">Driver Monitoring Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="welcome-text">'
        'Welcome to the <b>AI Driver Monitoring System!</b><br><br>'
        'AI-based system for detecting driver drowsiness and distracted driving in real time.'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    # -----------------------------------------------------
    # DROWSINESS CARD
    # -----------------------------------------------------

    with col1:

        st.markdown("""
        <div class="feature-card">

        <div class="feature-title">
        😴 Drowsiness Detection
        </div>

        <div class="feature-text">

        Detects eye closure and yawning using
        <b>Eye Aspect Ratio (EAR)</b> and
        <b>Mouth Aspect Ratio (MAR)</b>.

        <br><br>

        <b>Alerts:</b>

        <br>

        • Eye Closed

        <br>

        • Yawning

        </div>

        </div>
        """, unsafe_allow_html=True)

    # -----------------------------------------------------
    # DISTRACTION CARD
    # -----------------------------------------------------

    with col2:

        st.markdown("""
        <div class="feature-card">

        <div class="feature-title">
        👀 Distraction Detection
        </div>

        <div class="feature-text">

        Monitors the driver's head direction and
        detects prolonged looking away from the road.

        <br><br>

        <b>Alerts:</b>

        <br>

        • Looking Left

        <br>

        • Looking Right

        <br>

        • Looking Up

        <br>

        • Looking Down

        </div>

        </div>
        """, unsafe_allow_html=True)

    st.write("")

    col1, col2, col3 = st.columns([1, 1.5, 1])

    with col2:

        if st.button(
            "🚗 START DRIVER MONITORING",
            use_container_width=True
        ):

            st.session_state.page = "camera"
            st.rerun()

    st.write("")

    if st.button("LOGOUT", use_container_width=True):

        st.session_state.logged_in = False
        st.session_state.page = "login"
        st.rerun()


# =========================================================
# CAMERA PAGE
# =========================================================

def camera_page():

    st.markdown(
        '<div class="camera-title">🚗 AI Driver Monitoring</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <p style="text-align:center;color:#5d6b7a;">
        Look directly at the camera. The system continuously monitors
        your eyes, mouth and head direction.
        </p>
        """,
        unsafe_allow_html=True
    )

    if st.button("⬅️ Back to Dashboard"):

        st.session_state.page = "dashboard"
        st.rerun()

    st.write("")

    start_camera = st.checkbox(
        "START CAMERA",
        value=False
    )

    if not start_camera:

        st.info("Enable START CAMERA to begin driver monitoring.")

        return

    frame_placeholder = st.empty()
    status_placeholder = st.empty()

    # -----------------------------------------------------
    # CAMERA
    # -----------------------------------------------------

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        st.error(
            "Unable to access camera. "
            "Please check camera permission and make sure "
            "another application is not using the camera."
        )

        return

    # -----------------------------------------------------
    # VARIABLES
    # -----------------------------------------------------

    eyes_closed_start = None
    yawn_start = None
    distraction_start = None

    last_alarm_time = 0

    EYE_CLOSED_THRESHOLD = 0.21
    EYE_CLOSED_TIME = 2.0

    YAWN_THRESHOLD = 0.65
    YAWN_TIME = 1.5

    DISTRACTION_TIME = 2.0

    ALARM_COOLDOWN = 2.0

    # -----------------------------------------------------
    # MEDIAPIPE FACE MESH
    # -----------------------------------------------------

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:

            ret, frame = cap.read()

            if not ret:
                st.error("Unable to read camera frame.")
                break

            frame = cv2.flip(frame, 1)

            frame_height, frame_width = frame.shape[:2]

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            results = face_mesh.process(rgb_frame)

            status = "Monitoring..."
            status_color = (0, 180, 0)

            # =================================================
            # FACE DETECTION
            # =================================================

            if results.multi_face_landmarks:

                face_landmarks = results.multi_face_landmarks[0]
                landmarks = face_landmarks.landmark

                # -------------------------------------------------
                # FACE BOX
                # -------------------------------------------------

                xs = [
                    int(point.x * frame_width)
                    for point in landmarks
                    if 0 <= point.x <= 1
                ]

                ys = [
                    int(point.y * frame_height)
                    for point in landmarks
                    if 0 <= point.y <= 1
                ]

                if xs and ys:

                    x1 = max(0, min(xs) - 20)
                    y1 = max(0, min(ys) - 20)
                    x2 = min(frame_width, max(xs) + 20)
                    y2 = min(frame_height, max(ys) + 20)

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                # =================================================
                # EAR
                # =================================================

                right_ear = calculate_ear(
                    landmarks,
                    RIGHT_EYE,
                    frame_width,
                    frame_height
                )

                left_ear = calculate_ear(
                    landmarks,
                    LEFT_EYE,
                    frame_width,
                    frame_height
                )

                ear = (right_ear + left_ear) / 2.0

                # =================================================
                # MAR
                # =================================================

                mar = calculate_mar(
                    landmarks,
                    frame_width,
                    frame_height
                )

                # =================================================
                # EYE CLOSED DETECTION
                # =================================================

                if ear < EYE_CLOSED_THRESHOLD:

                    if eyes_closed_start is None:
                        eyes_closed_start = time.time()

                    closed_time = (
                        time.time() - eyes_closed_start
                    )

                    cv2.putText(
                        frame,
                        f"Eyes Closed: {closed_time:.1f}s",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )

                    if closed_time >= EYE_CLOSED_TIME:

                        status = "⚠️ DROWSINESS ALERT - EYES CLOSED"
                        status_color = (0, 0, 255)

                        current_time = time.time()

                        if (
                            current_time - last_alarm_time
                            > ALARM_COOLDOWN
                        ):

                            play_alarm()
                            last_alarm_time = current_time

                else:

                    eyes_closed_start = None

                # =================================================
                # YAWN DETECTION
                # =================================================

                if mar > YAWN_THRESHOLD:

                    if yawn_start is None:
                        yawn_start = time.time()

                    yawn_time = (
                        time.time() - yawn_start
                    )

                    cv2.putText(
                        frame,
                        f"Yawning: {yawn_time:.1f}s",
                        (30, 85),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 165, 255),
                        2
                    )

                    if yawn_time >= YAWN_TIME:

                        status = "⚠️ DROWSINESS ALERT - YAWNING"
                        status_color = (0, 165, 255)

                        current_time = time.time()

                        if (
                            current_time - last_alarm_time
                            > ALARM_COOLDOWN
                        ):

                            play_alarm()
                            last_alarm_time = current_time

                else:

                    yawn_start = None

                # =================================================
                # HEAD DIRECTION / DISTRACTION
                # =================================================

                nose = landmarks[1]

                nose_x = nose.x
                nose_y = nose.y

                left_reference = landmarks[234]
                right_reference = landmarks[454]

                face_center_x = (
                    left_reference.x +
                    right_reference.x
                ) / 2

                face_width_normalized = abs(
                    right_reference.x -
                    left_reference.x
                )

                if face_width_normalized > 0:

                    horizontal_ratio = (
                        nose_x - face_center_x
                    ) / face_width_normalized

                else:

                    horizontal_ratio = 0

                top_reference = landmarks[10]
                bottom_reference = landmarks[152]

                face_center_y = (
                    top_reference.y +
                    bottom_reference.y
                ) / 2

                face_height_normalized = abs(
                    bottom_reference.y -
                    top_reference.y
                )

                if face_height_normalized > 0:

                    vertical_ratio = (
                        nose_y - face_center_y
                    ) / face_height_normalized

                else:

                    vertical_ratio = 0

                # -------------------------------------------------
                # HEAD DIRECTION THRESHOLDS
                # -------------------------------------------------

                direction = "CENTER"

                if horizontal_ratio < -0.13:

                    direction = "RIGHT"

                elif horizontal_ratio > 0.13:

                    direction = "LEFT"

                elif vertical_ratio < -0.12:

                    direction = "UP"

                elif vertical_ratio > 0.12:

                    direction = "DOWN"

                # -------------------------------------------------
                # DISPLAY DIRECTION
                # -------------------------------------------------

                cv2.putText(
                    frame,
                    f"Head Direction: {direction}",
                    (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2
                )

                # =================================================
                # DISTRACTION TIMER
                # =================================================

                if direction != "CENTER":

                    if distraction_start is None:
                        distraction_start = time.time()

                    distraction_time = (
                        time.time() -
                        distraction_start
                    )

                    cv2.putText(
                        frame,
                        f"Looking Away: {distraction_time:.1f}s",
                        (30, 155),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )

                    if distraction_time >= DISTRACTION_TIME:

                        status = (
                            f"⚠️ DISTRACTION ALERT - "
                            f"LOOKING {direction}"
                        )

                        status_color = (0, 0, 255)

                        current_time = time.time()

                        if (
                            current_time - last_alarm_time
                            > ALARM_COOLDOWN
                        ):

                            play_alarm()
                            last_alarm_time = current_time

                else:

                    distraction_start = None

                # =================================================
                # DISPLAY EAR AND MAR
                # =================================================

                cv2.putText(
                    frame,
                    f"EAR: {ear:.2f}",
                    (30, frame_height - 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"MAR: {mar:.2f}",
                    (30, frame_height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

            else:

                status = "FACE NOT DETECTED"
                status_color = (0, 0, 255)

                eyes_closed_start = None
                yawn_start = None
                distraction_start = None

                cv2.putText(
                    frame,
                    "FACE NOT DETECTED",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

            # =================================================
            # STATUS ON VIDEO
            # =================================================

            cv2.putText(
                frame,
                status,
                (30, 195),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                status_color,
                2
            )

            # =================================================
            # SHOW FRAME
            # =================================================

            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            frame_placeholder.image(
                frame_rgb,
                channels="RGB",
                use_container_width=True
            )

            status_placeholder.markdown(
                f"""
                <div class="status-box">
                Status: <span style="color:
                rgb({status_color[2]},
                {status_color[1]},
                {status_color[0]});">
                {status}
                </span>
                </div>
                """,
                unsafe_allow_html=True
            )

    cap.release()


# =========================================================
# PAGE ROUTING
# =========================================================

if not st.session_state.logged_in:

    login_page()

else:

    if st.session_state.page == "dashboard":

        dashboard_page()

    elif st.session_state.page == "camera":

        camera_page()

    else:

        dashboard_page()