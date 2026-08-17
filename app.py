
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import cv2
import numpy as np
import subprocess


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Helmet Safety Detection",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# STYLE
# ============================================================

st.markdown("""
<style>

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.title {
    text-align: center;
    font-size: 36px;
    font-weight: 700;
    margin-bottom: 4px;
}

.subtitle {
    text-align: center;
    color: #666;
    font-size: 16px;
    margin-bottom: 30px;
}

.section {
    font-size: 24px;
    font-weight: 650;
    margin-top: 20px;
    margin-bottom: 15px;
}

.result-title {
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 10px;
}

.card {
    border: 1px solid #d8d8d8;
    border-radius: 12px;
    padding: 18px 10px;
    text-align: center;
    background: #fafafa;
}

.card-title {
    font-size: 15px;
    font-weight: 600;
}

.card-value {
    font-size: 30px;
    font-weight: 700;
    margin-top: 5px;
}

.footer {
    text-align: center;
    color: #777;
    font-size: 13px;
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="title">Helmet Safety Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'YOLOv8-based Personal Protective Equipment Detection'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = (
    "/content/drive/MyDrive/Helmet_Safety_Detection/"
    "outputs/training_results/yolov8n_baseline/"
    "weights/best.pt"
)


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.35,
    step=0.05
)

st.sidebar.write(
    f"Current value: **{confidence:.2f}**"
)

st.sidebar.markdown("---")

st.sidebar.write("Model: YOLOv8n")
st.sidebar.write("Classes: 4")

st.sidebar.markdown("""
**Detection Classes**

- Helmet
- Vest
- Without Helmet
- Without Vest
""")


# ============================================================
# INPUT TYPE
# ============================================================

st.markdown(
    '<div class="section">Input</div>',
    unsafe_allow_html=True
)

input_type = st.radio(
    "Select input type",
    ["Image", "Video"],
    horizontal=True
)


# ============================================================
# IMAGE DETECTION
# ============================================================

if input_type == "Image":

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        if st.button(
            "Run Detection",
            use_container_width=True
        ):

            with st.spinner("Running detection..."):

                # PIL image is RGB.
                # Convert explicitly to BGR for OpenCV/YOLO.
                image_rgb = np.array(image)

                image_bgr = cv2.cvtColor(
                    image_rgb,
                    cv2.COLOR_RGB2BGR
                )

                result = model.predict(
                    source=image_bgr,
                    conf=confidence,
                    verbose=False
                )[0]

                # YOLO plotting output is BGR.
                detected_bgr = result.plot()

                # Convert back to RGB for Streamlit.
                detected_rgb = cv2.cvtColor(
                    detected_bgr,
                    cv2.COLOR_BGR2RGB
                )


            # ==================================================
            # IMAGE RESULTS
            # ==================================================

            st.markdown("---")

            st.markdown(
                '<div class="section">Detection Result</div>',
                unsafe_allow_html=True
            )

            col1, col2 = st.columns(2)

            with col1:

                st.markdown(
                    '<div class="result-title">'
                    'Original Image'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.image(
                    image_rgb,
                    use_container_width=True
                )

            with col2:

                st.markdown(
                    '<div class="result-title">'
                    'Detected Image'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.image(
                    detected_rgb,
                    use_container_width=True
                )


            # ==================================================
            # IMAGE COUNTS
            # ==================================================

            counts = {
                "helmet": 0,
                "vest": 0,
                "without_helmet": 0,
                "without_vest": 0
            }

            for box in result.boxes:

                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                if class_name in counts:
                    counts[class_name] += 1


            st.markdown("---")

            st.markdown(
                '<div class="section">Detected Objects</div>',
                unsafe_allow_html=True
            )

            cols = st.columns(4)

            classes = [
                ("Helmet", "helmet"),
                ("Vest", "vest"),
                ("Without Helmet", "without_helmet"),
                ("Without Vest", "without_vest")
            ]

            for col, item in zip(cols, classes):

                label, key = item

                with col:

                    st.markdown(
                        f"""
                        <div class="card">
                            <div class="card-title">
                                {label}
                            </div>
                            <div class="card-value">
                                {counts[key]}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


# ============================================================
# VIDEO DETECTION
# ============================================================

else:

    uploaded_file = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_file is not None:

        if st.button(
            "Run Video Detection",
            use_container_width=True
        ):

            with st.spinner("Processing video..."):

                # ------------------------------------------------
                # Save uploaded video
                # ------------------------------------------------

                input_file = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=os.path.splitext(
                        uploaded_file.name
                    )[1]
                )

                input_file.write(
                    uploaded_file.getbuffer()
                )

                input_file.close()

                # ------------------------------------------------
                # Convert original video to browser-compatible MP4
                # ------------------------------------------------

                original_mp4 = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                )

                original_mp4.close()

                original_conversion = subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        input_file.name,
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        "-movflags",
                        "+faststart",
                        original_mp4.name
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )

                if original_conversion.returncode == 0:
                    original_video_path = original_mp4.name
                else:
                    original_video_path = input_file.name


                # ------------------------------------------------
                # Open video
                # ------------------------------------------------

                cap = cv2.VideoCapture(
                    input_file.name
                )

                fps = cap.get(
                    cv2.CAP_PROP_FPS
                )

                if fps <= 0:
                    fps = 25

                width = int(
                    cap.get(
                        cv2.CAP_PROP_FRAME_WIDTH
                    )
                )

                height = int(
                    cap.get(
                        cv2.CAP_PROP_FRAME_HEIGHT
                    )
                )

                total_frames = int(
                    cap.get(
                        cv2.CAP_PROP_FRAME_COUNT
                    )
                )


                # ------------------------------------------------
                # Raw detected video
                # ------------------------------------------------

                raw_video = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                )

                raw_video.close()

                fourcc = cv2.VideoWriter_fourcc(
                    *"mp4v"
                )

                writer = cv2.VideoWriter(
                    raw_video.name,
                    fourcc,
                    fps,
                    (width, height)
                )


                # ------------------------------------------------
                # Peak simultaneous detections
                # ------------------------------------------------

                peak_counts = {
                    "helmet": 0,
                    "vest": 0,
                    "without_helmet": 0,
                    "without_vest": 0
                }


                frame_number = 0

                progress = st.progress(0)

                status = st.empty()


                # =================================================
                # PROCESS VIDEO
                # =================================================

                while True:

                    success, frame_bgr = cap.read()

                    if not success:
                        break


                    result = model.predict(
                        source=frame_bgr,
                        conf=confidence,
                        verbose=False
                    )[0]


                    # YOLO result is BGR.
                    # VideoWriter expects BGR.
                    annotated_bgr = result.plot()

                    writer.write(
                        annotated_bgr
                    )


                    # Count objects in CURRENT frame.
                    frame_counts = {
                        "helmet": 0,
                        "vest": 0,
                        "without_helmet": 0,
                        "without_vest": 0
                    }


                    for box in result.boxes:

                        class_id = int(
                            box.cls[0]
                        )

                        class_name = model.names[
                            class_id
                        ]

                        if class_name in frame_counts:

                            frame_counts[
                                class_name
                            ] += 1


                    # Keep maximum simultaneous detections.
                    for key in peak_counts:

                        peak_counts[key] = max(
                            peak_counts[key],
                            frame_counts[key]
                        )


                    frame_number += 1


                    if total_frames > 0:

                        progress_value = min(
                            frame_number /
                            total_frames,
                            1.0
                        )

                        progress.progress(
                            progress_value
                        )

                        status.text(
                            f"Processing frame "
                            f"{frame_number}/{total_frames}"
                        )


                cap.release()
                writer.release()

                progress.progress(1.0)

                status.text(
                    "Video processing completed."
                )


                # =================================================
                # H.264 DETECTED VIDEO
                # =================================================

                detected_video = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                )

                detected_video.close()


                conversion = subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        raw_video.name,
                        "-c:v",
                        "libx264",
                        "-preset",
                        "fast",
                        "-pix_fmt",
                        "yuv420p",
                        "-movflags",
                        "+faststart",
                        detected_video.name
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )


                if (
                    conversion.returncode == 0
                    and os.path.exists(
                        detected_video.name
                    )
                    and os.path.getsize(
                        detected_video.name
                    ) > 0
                ):

                    detected_video_path = (
                        detected_video.name
                    )

                else:

                    detected_video_path = (
                        raw_video.name
                    )


            # =====================================================
            # VIDEO RESULTS
            # =====================================================

            st.success(
                "Video detection completed successfully."
            )

            st.markdown("---")

            st.markdown(
                '<div class="section">Detection Result</div>',
                unsafe_allow_html=True
            )


            col1, col2 = st.columns(2)


            with col1:

                st.markdown(
                    '<div class="result-title">'
                    'Original Video'
                    '</div>',
                    unsafe_allow_html=True
                )

                with open(
                    original_video_path,
                    "rb"
                ) as video_file:

                    original_bytes = video_file.read()

                st.video(
                    original_bytes
                )


            with col2:

                st.markdown(
                    '<div class="result-title">'
                    'Detected Video'
                    '</div>',
                    unsafe_allow_html=True
                )

                with open(
                    detected_video_path,
                    "rb"
                ) as video_file:

                    detected_bytes = video_file.read()

                st.video(
                    detected_bytes
                )


            # =====================================================
            # VIDEO COUNTS
            # =====================================================

            st.markdown("---")

            st.markdown(
                '<div class="section">Detected Objects</div>',
                unsafe_allow_html=True
            )

            st.caption(
                "Maximum number of objects detected "
                "simultaneously in a single frame."
            )


            cols = st.columns(4)


            classes = [
                ("Helmet", "helmet"),
                ("Vest", "vest"),
                ("Without Helmet", "without_helmet"),
                ("Without Vest", "without_vest")
            ]


            for col, item in zip(cols, classes):

                label, key = item

                with col:

                    st.markdown(
                        f"""
                        <div class="card">
                            <div class="card-title">
                                {label}
                            </div>
                            <div class="card-value">
                                {peak_counts[key]}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


            # -----------------------------------------------------
            # Cleanup
            # -----------------------------------------------------

            for path in [
                input_file.name,
                raw_video.name,
                original_mp4.name,
                detected_video.name
            ]:

                try:
                    if os.path.exists(path):
                        os.remove(path)
                except:
                    pass


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="footer">'
    'Helmet Safety Detection | YOLOv8n | '
    'Four-Class PPE Detection'
    '</div>',
    unsafe_allow_html=True
)
