import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# =========================
# CẤU HÌNH APP
# =========================
st.set_page_config(
    page_title="ide.face",
    page_icon="🧬",
    layout="wide"
)

# =========================
# CLASS LABELS
# Anh sửa đúng theo thứ tự lúc train
# =========================
class_labels = [
    "HoangKyAnh",
    "Lê Quang Dũng",
    "Lê Tuấn Thành",
    "Lương Ngọc Thuận",
    "Ngô Quốc Trung",
    "Nguyễn Ngọc Bảo",
    "Nguyễn Hoàng Quế Châu",
    "Nguyễn Phạm Hoàng Anh",
    "Nguyễn Thành Đạt",
    "Nguyễn Trọng Nghĩa",
    "Nguyễn Văn A",
    "Nguyễn Văn B",
    "Phạm Hoàng Nam",
    "Phạm Minh Đức",
    "Trần Anh Khoa",
    "Trần Bảo Long",
    "Trần Đức Huy",
    "Trần Minh Nhật",
    "Võ Hoàng Phúc",
    "Võ Minh Quân",
    "Vũ Hoàng Long",
    "Đặng Quốc Bảo"
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
        margin-bottom: 35px;
    }

    .brand {
        font-size: 30px;
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
        max-width: 780px;
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

    .result-box {
        padding: 34px;
        border-radius: 28px;
        background: linear-gradient(135deg, #06b6d4, #8b5cf6);
        text-align: center;
        color: white;
        box-shadow: 0 18px 50px rgba(0,0,0,0.35);
        margin-top: 15px;
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

    .guide-box {
        padding: 24px;
        border-radius: 24px;
        background: rgba(250, 204, 21, 0.10);
        border: 1px solid rgba(250, 204, 21, 0.35);
        color: #fef9c3;
        font-size: 17px;
        line-height: 1.65;
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
    <div class="brand">ide.face</div>
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
        <div class="hero-title">ide.face</div>
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
            <div class="card-title">⚡ Giao diện nhanh gọn</div>
            <div class="card-text">
                Thiết kế tối giản, hiện đại, dễ sử dụng và phù hợp để trình bày trong đồ án hoặc báo cáo.
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
            Chọn ảnh từ thiết bị hoặc chụp ảnh trực tiếp để hệ thống ide.face tiến hành nhận diện.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.stop()

    tab1, tab2 = st.tabs(["📁 Tải ảnh lên", "📸 Chụp bằng camera"])

    image = None

    with tab1:
        uploaded_file = st.file_uploader(
            "Chọn ảnh khuôn mặt",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)

    with tab2:
        camera_file = st.camera_input("Chụp ảnh khuôn mặt")

        if camera_file is not None:
            image = Image.open(camera_file)

    if image is not None:
        col_img, col_result = st.columns([1, 1])

        with col_img:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Ảnh đầu vào")
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
                        st.write(f"**{class_labels[idx]}**: {all_predictions[idx] * 100:.2f}%")

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
        <div class="tag">About ide.face</div>
        <div class="hero-title">Giới thiệu ứng dụng</div>
        <div class="hero-desc">
            ide.face là ứng dụng AI nhận diện khuôn mặt được xây dựng phục vụ học tập và nghiên cứu.
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