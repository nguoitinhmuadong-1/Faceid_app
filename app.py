import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# =========================
# CẤU HÌNH APP
# =========================
st.set_page_config(
    page_title="IDeFace",
    page_icon="🧬",
    layout="wide"
)

# =========================
# CLASS LABELS
# Lưu ý: thứ tự phải đúng với train_generator.class_indices lúc train
# =========================
class_labels = [
    "Đinh Hữu Khánh Anh",
    "Đoàn Hùng",
    "Đỗ An Phúc",
    "Hoàng Kỳ Anh",
    "Lê Quang Dũng",
    "Lê Tuấn Thành",
    "Lương Ngọc Thuận",
    "Ngô Quốc Trung",
    "Nguyễn Ngọc Bảo",
    "Nguyễn Đặng Vinh Phúc",
    "Nguyễn Hoàng Quế Châu",
    "Nguyễn Phạm Hoàng An",
    "Nguyễn Thị Khánh Lê",
    "Nguyễn Thị Ngọc Tuyết",
    "Nguyễn Tiến Mạnh",
    "Nguyễn Việt Đức",
    "Phạm Gia Thành Duy",
    "Phạm Hứa Nhật Minh",
    "Phạm Nguyễn Bảo Châu",
    "Phạm Phú Hoà",
    "Trần Hải Yến",
    "Vũ Quang Thái"
]

# =========================
# CSS GIAO DIỆN
# =========================
st.markdown("""
<style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(34, 211, 238, 0.22), transparent 32%),
            radial-gradient(circle at bottom right, rgba(168, 85, 247, 0.20), transparent 35%),
            linear-gradient(135deg, #020617 0%, #0f172a 45%, #111827 100%);
        color: white;
    }

    [data-testid="stSidebar"] {
        display: none;
    }

    [data-testid="collapsedControl"] {
        display: none;
    }

    header {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    #MainMenu {
        visibility: hidden;
    }

    .navbar {
        width: 100%;
        padding: 18px 28px;
        border-radius: 22px;
        background: rgba(15, 23, 42, 0.78);
        border: 1px solid rgba(255,255,255,0.12);
        backdrop-filter: blur(16px);
        box-shadow: 0 12px 35px rgba(0,0,0,0.35);
        margin-bottom: 25px;
    }

    .brand {
        font-size: 32px;
        font-weight: 950;
        letter-spacing: -1px;
        background: linear-gradient(90deg, #22d3ee, #a78bfa, #facc15);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .brand-sub {
        font-size: 14px;
        color: #cbd5e1;
        margin-top: -4px;
    }

    .hero {
        padding: 55px 45px;
        border-radius: 32px;
        background:
            linear-gradient(135deg, rgba(34,211,238,0.15), rgba(168,85,247,0.12)),
            rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.15);
        box-shadow: 0 18px 55px rgba(0,0,0,0.35);
        margin-bottom: 28px;
    }

    .hero-title {
        font-size: 72px;
        line-height: 1.05;
        font-weight: 950;
        letter-spacing: -3px;
        background: linear-gradient(90deg, #ffffff, #67e8f9, #c4b5fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 18px;
    }

    .hero-desc {
        font-size: 21px;
        line-height: 1.7;
        color: #dbeafe;
        max-width: 820px;
    }

    .tag {
        display: inline-block;
        padding: 8px 15px;
        border-radius: 999px;
        background: rgba(34, 211, 238, 0.14);
        border: 1px solid rgba(34, 211, 238, 0.35);
        color: #a5f3fc;
        font-size: 14px;
        font-weight: 700;
        margin-bottom: 18px;
    }

    .card {
        min-height: 190px;
        padding: 26px;
        border-radius: 26px;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.12);
        box-shadow: 0 14px 40px rgba(0,0,0,0.25);
        backdrop-filter: blur(14px);
        margin-bottom: 20px;
    }

    .card-title {
        font-size: 25px;
        font-weight: 900;
        margin-bottom: 12px;
        color: white;
    }

    .card-text {
        font-size: 17px;
        color: #cbd5e1;
        line-height: 1.65;
    }

    .guide-box {
        padding: 24px;
        border-radius: 24px;
        background: rgba(250, 204, 21, 0.10);
        border: 1px solid rgba(250, 204, 21, 0.35);
        color: #fef9c3;
        font-size: 17px;
        line-height: 1.65;
        margin-bottom: 20px;
    }

    .result-box {
        padding: 34px;
        border-radius: 28px;
        background: linear-gradient(135deg, #06b6d4, #8b5cf6);
        text-align: center;
        color: white;
        box-shadow: 0 18px 50px rgba(0,0,0,0.35);
        margin-top: 15px;
        margin-bottom: 20px;
    }

    .result-name {
        font-size: 38px;
        font-weight: 950;
        margin-bottom: 8px;
    }

    .result-score {
        font-size: 24px;
        font-weight: 800;
    }

    .stButton>button {
        width: 100%;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.18);
        background: linear-gradient(90deg, #06b6d4, #8b5cf6);
        color: white;
        font-weight: 850;
        padding: 12px 22px;
        box-shadow: 0 10px 26px rgba(0,0,0,0.28);
        transition: 0.25s;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        background: linear-gradient(90deg, #0891b2, #7c3aed);
        color: white;
    }

    .stFileUploader, .stCameraInput {
        background: rgba(255,255,255,0.06);
        padding: 18px;
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.10);
    }

    h1, h2, h3, p, label, span {
        color: white;
    }

    .scanner-box {
        position: relative;
        padding: 20px;
        border-radius: 30px;
        background: rgba(15, 23, 42, 0.88);
        border: 1px solid rgba(34, 211, 238, 0.35);
        box-shadow:
            0 0 35px rgba(34, 211, 238, 0.22),
            inset 0 0 28px rgba(34, 211, 238, 0.10);
        overflow: hidden;
        margin-bottom: 20px;
    }

    .scanner-title {
        text-align: center;
        font-size: 22px;
        font-weight: 900;
        color: #67e8f9;
        margin-bottom: 12px;
        letter-spacing: 2px;
    }

    .scanner-frame {
        position: relative;
        min-height: 320px;
        border-radius: 24px;
        overflow: hidden;
        border: 2px solid rgba(34, 211, 238, 0.65);
        box-shadow: 0 0 25px rgba(34, 211, 238, 0.25);
        background:
            linear-gradient(rgba(34, 211, 238, 0.06) 1px, transparent 1px),
            linear-gradient(90deg, rgba(34, 211, 238, 0.06) 1px, transparent 1px),
            rgba(2, 6, 23, 0.55);
        background-size: 28px 28px;
    }

    .scanner-line {
        position: absolute;
        left: 0;
        top: 0;
        width: 100%;
        height: 5px;
        background: linear-gradient(90deg, transparent, #22d3ee, #a78bfa, transparent);
        box-shadow: 0 0 22px #22d3ee;
        animation: scanMove 2.2s linear infinite;
        z-index: 10;
    }

    @keyframes scanMove {
        0% {
            top: 0%;
            opacity: 0.25;
        }
        50% {
            opacity: 1;
        }
        100% {
            top: 100%;
            opacity: 0.25;
        }
    }

    .corner {
        position: absolute;
        width: 42px;
        height: 42px;
        border-color: #22d3ee;
        z-index: 11;
    }

    .corner-tl {
        top: 14px;
        left: 14px;
        border-top: 4px solid #22d3ee;
        border-left: 4px solid #22d3ee;
    }

    .corner-tr {
        top: 14px;
        right: 14px;
        border-top: 4px solid #22d3ee;
        border-right: 4px solid #22d3ee;
    }

    .corner-bl {
        bottom: 14px;
        left: 14px;
        border-bottom: 4px solid #22d3ee;
        border-left: 4px solid #22d3ee;
    }

    .corner-br {
        bottom: 14px;
        right: 14px;
        border-bottom: 4px solid #22d3ee;
        border-right: 4px solid #22d3ee;
    }

    .scanner-status {
        text-align: center;
        margin-top: 12px;
        font-size: 15px;
        color: #cbd5e1;
    }

    .scanner-icon {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        color: rgba(103, 232, 249, 0.7);
        font-size: 76px;
        font-weight: 900;
        z-index: 3;
    }

    .metric-row {
        padding: 14px 16px;
        border-radius: 16px;
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.10);
        margin-bottom: 10px;
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_face_model():
    model_path = "faceid.h5"

    if not os.path.exists(model_path):
        st.error("Không tìm thấy file faceid.h5. Anh hãy đặt file faceid.h5 cùng thư mục với app.py")
        return None

    model = tf.keras.models.load_model(model_path, compile=False)
    return model

model = load_face_model()

# =========================
# XỬ LÝ ẢNH
# =========================
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((200, 200))
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# =========================
# DỰ ĐOÁN
# =========================
def predict_face(image):
    img_array = preprocess_image(image)

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    confidence = float(np.max(predictions)) * 100

    if predicted_index < len(class_labels):
        predicted_name = class_labels[predicted_index]
    else:
        predicted_name = "Không xác định"

    return predicted_name, confidence, predictions[0]

# =========================
# ĐIỀU HƯỚNG KHÔNG SIDEBAR
# =========================
if "page" not in st.session_state:
    st.session_state.page = "home"

st.markdown("""
<div class="navbar">
    <div class="brand">IDeFace</div>
    <div class="brand-sub">AI Face Recognition System</div>
</div>
""", unsafe_allow_html=True)

nav1, nav2, nav3, nav4 = st.columns([5.7, 1.2, 1.7, 1.4])

with nav2:
    if st.button("🏠 Home"):
        st.session_state.page = "home"

with nav3:
    if st.button("📷 Nhận dạng"):
        st.session_state.page = "detect"

with nav4:
    if st.button("ℹ️ About"):
        st.session_state.page = "about"

page = st.session_state.page

# =========================
# TRANG HOME
# =========================
if page == "home":
    st.markdown("""
    <div class="hero">
        <div class="tag">AI Face Recognition • CNN Model • Student Identity</div>
        <div class="hero-title">IDeFace</div>
        <div class="hero-desc">
            Ứng dụng nhận diện khuôn mặt sinh viên lớp Logtech bán phần bằng mô hình trí tuệ nhân tạo.
            Hệ thống cho phép tải ảnh hoặc chụp trực tiếp từ camera để dự đoán danh tính sinh viên.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-title">📷 Nhận diện khuôn mặt</div>
            <div class="card-text">
                Hỗ trợ nhận diện từ ảnh tải lên hoặc ảnh chụp trực tiếp bằng camera trên máy tính, điện thoại.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-title">🧠 Mô hình CNN</div>
            <div class="card-text">
                Ứng dụng sử dụng mô hình học sâu đã được huấn luyện từ dữ liệu khuôn mặt của sinh viên.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="card">
            <div class="card-title">⚡ Giao diện hiện đại</div>
            <div class="card-text">
                Thiết kế tối giản, màu sắc hiện đại, dễ sử dụng và phù hợp để trình bày trong đồ án.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="guide-box">
        <b>Gợi ý sử dụng:</b> Ảnh đầu vào nên rõ khuôn mặt, đủ sáng, không bị che mặt và nên chụp chính diện để tăng độ chính xác.
    </div>
    """, unsafe_allow_html=True)

# =========================
# TRANG NHẬN DẠNG
# =========================
elif page == "detect":
    st.markdown("""
    <div class="hero">
        <div class="tag">Recognition Mode</div>
        <div class="hero-title">Nhận dạng khuôn mặt</div>
        <div class="hero-desc">
            Chọn ảnh từ thiết bị hoặc chụp ảnh trực tiếp. Hệ thống IDeFace sẽ xử lý và dự đoán danh tính sinh viên.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.stop()

    tab1, tab2 = st.tabs(["📁 Tải ảnh lên", "📸 Chụp bằng máy quét"])

    image = None

    with tab1:
        st.markdown("""
        <div class="guide-box">
            Tải lên ảnh khuôn mặt rõ nét. Ảnh nên có ánh sáng tốt và khuôn mặt nằm gần trung tâm.
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Chọn ảnh khuôn mặt",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)

    with tab2:
        st.markdown("""
        <div class="scanner-box">
            <div class="scanner-title">FACE SCANNER READY</div>
            <div class="scanner-frame">
                <div class="scanner-line"></div>
                <div class="corner corner-tl"></div>
                <div class="corner corner-tr"></div>
                <div class="corner corner-bl"></div>
                <div class="corner corner-br"></div>
                <div class="scanner-icon">⌖</div>
            </div>
            <div class="scanner-status">
                Căn khuôn mặt vào giữa khung, giữ đủ sáng rồi bấm chụp ảnh
            </div>
        </div>
        """, unsafe_allow_html=True)

        camera_file = st.camera_input("📸 Mở camera và chụp khuôn mặt")

        if camera_file is not None:
            image = Image.open(camera_file)

    if image is not None:
        col_img, col_result = st.columns([1, 1])

        with col_img:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Ảnh đang quét")

            st.markdown("""
            <div class="scanner-box">
                <div class="scanner-title">SCANNING FACE</div>
                <div class="scanner-frame">
                    <div class="scanner-line"></div>
                    <div class="corner corner-tl"></div>
                    <div class="corner corner-tr"></div>
                    <div class="corner corner-bl"></div>
                    <div class="corner corner-br"></div>
                    <div class="scanner-icon">AI</div>
                </div>
                <div class="scanner-status">
                    Đang phân tích đặc trưng khuôn mặt bằng mô hình CNN
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.image(image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_result:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Kết quả dự đoán")

            if st.button("🔍 Bắt đầu nhận diện"):
                predicted_name, confidence, all_predictions = predict_face(image)

                st.markdown(f"""
                <div class="result-box">
                    <div class="result-name">👤 {predicted_name}</div>
                    <div class="result-score">Độ tin cậy: {confidence:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)

                if confidence < 60:
                    st.warning("Độ tin cậy đang thấp. Anh nên thử ảnh rõ mặt hơn, đủ sáng hơn hoặc chính diện hơn.")

                st.markdown("### Top 5 kết quả gần nhất")

                top_indices = np.argsort(all_predictions)[::-1][:5]

                for idx in top_indices:
                    if idx < len(class_labels):
                        st.markdown(
                            f"""
                            <div class="metric-row">
                                <b>{class_labels[idx]}</b>: {all_predictions[idx] * 100:.2f}%
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

            else:
                st.info("Bấm nút **Bắt đầu nhận diện** để hệ thống dự đoán.")

            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="guide-box">
            Anh hãy tải ảnh lên hoặc chụp ảnh trực tiếp. Sau đó bấm nút <b>Bắt đầu nhận diện</b> để xem kết quả.
        </div>
        """, unsafe_allow_html=True)

# =========================
# TRANG ABOUT
# =========================
elif page == "about":
    st.markdown("""
    <div class="hero">
        <div class="tag">About IDeFace</div>
        <div class="hero-title">Giới thiệu ứng dụng</div>
        <div class="hero-desc">
            IDeFace là ứng dụng AI nhận diện khuôn mặt được xây dựng phục vụ học tập và nghiên cứu.
            Ứng dụng minh họa quá trình đưa mô hình CNN đã huấn luyện vào một hệ thống web đơn giản,
            trực quan và dễ sử dụng.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-title">🎯 Mục tiêu</div>
            <div class="card-text">
                Hỗ trợ nhận diện sinh viên lớp Logtech bán phần thông qua ảnh khuôn mặt,
                đồng thời thể hiện khả năng ứng dụng trí tuệ nhân tạo vào bài toán nhận dạng.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-title">🧩 Công nghệ sử dụng</div>
            <div class="card-text">
                Python, TensorFlow/Keras, mô hình CNN, thư viện PIL, NumPy và Streamlit để xây dựng giao diện web.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <div class="card-title">⚠️ Lưu ý</div>
        <div class="card-text">
            Kết quả nhận diện phụ thuộc vào chất lượng ảnh, ánh sáng, góc chụp và dữ liệu đã dùng để huấn luyện.
            Ứng dụng phù hợp cho mục đích học tập, thử nghiệm và trình bày đồ án.
        </div>
    </div>
    """, unsafe_allow_html=True)