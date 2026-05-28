import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# =========================
# CẤU HÌNH TRANG
# =========================
st.set_page_config(
    page_title="FaceAD",
    page_icon="🧠",
    layout="wide"
)

# =========================
# CSS GIAO DIỆN
# =========================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 45%, #0f766e 100%);
        color: white;
    }

    .main-title {
        font-size: 60px;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #22d3ee, #5eead4, #facc15);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }

    .sub-title {
        text-align: center;
        font-size: 22px;
        color: #e2e8f0;
        margin-bottom: 35px;
    }

    .card {
        background: rgba(255,255,255,0.12);
        padding: 28px;
        border-radius: 25px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.25);
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }

    .big-text {
        font-size: 28px;
        font-weight: 800;
        color: #ffffff;
    }

    .normal-text {
        font-size: 18px;
        color: #e2e8f0;
        line-height: 1.7;
    }

    .result-box {
        background: linear-gradient(135deg, #14b8a6, #0891b2);
        padding: 25px;
        border-radius: 22px;
        text-align: center;
        color: white;
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }

    .warning-box {
        background: rgba(250, 204, 21, 0.15);
        padding: 18px;
        border-radius: 18px;
        border: 1px solid #facc15;
        color: #fef9c3;
        font-size: 16px;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #020617, #0f172a);
    }

    .stButton>button {
        background: linear-gradient(90deg, #06b6d4, #14b8a6);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 12px 24px;
        font-weight: 700;
    }

    .stButton>button:hover {
        background: linear-gradient(90deg, #0891b2, #0d9488);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# CLASS LABELS
# Anh sửa lại đúng thứ tự class khi train
# =========================
class_labels = [
    "HoangKyAnh",
    "Lê Quang Dũng",
    "Lê Tuấn Thành",
    "Lương Ngọc Thuận",
    "Ngô Quốc Trung",
    "Nguyen Ngoc Bao",
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
# TIỀN XỬ LÝ ẢNH
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
# SIDEBAR
# =========================
st.sidebar.markdown("## 🧠 FaceAD")
st.sidebar.markdown("Ứng dụng nhận diện khuôn mặt sinh viên")

page = st.sidebar.radio(
    "Chọn trang",
    ["🏠 Trang chủ", "📷 Nhận dạng khuôn mặt"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Thông tin")
st.sidebar.info("""
FaceAD là ứng dụng AI sử dụng mô hình CNN để nhận diện khuôn mặt sinh viên lớp Logtech bán phần.
""")

# =========================
# TRANG CHỦ
# =========================
if page == "🏠 Trang chủ":
    st.markdown('<h1 class="main-title">FaceAD</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-title">Ứng dụng nhận diện khuôn mặt sinh viên lớp Logtech bán phần</p>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="card">
            <div class="big-text">📷 Nhận diện ảnh</div>
            <p class="normal-text">
            Cho phép tải ảnh khuôn mặt từ máy tính hoặc điện thoại để AI dự đoán sinh viên.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <div class="big-text">🤖 Mô hình CNN</div>
            <p class="normal-text">
            Ứng dụng sử dụng mô hình học sâu CNN đã được huấn luyện trên dữ liệu khuôn mặt sinh viên.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="card">
            <div class="big-text">⚡ Nhanh chóng</div>
            <p class="normal-text">
            Chỉ cần chọn ảnh hoặc chụp ảnh trực tiếp, hệ thống sẽ trả về tên sinh viên và độ tin cậy.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <div class="big-text">🎯 Mục tiêu ứng dụng</div>
        <p class="normal-text">
        FaceAD được xây dựng nhằm hỗ trợ nhận diện sinh viên trong lớp Logtech bán phần.
        Ứng dụng giúp minh họa cách trí tuệ nhân tạo có thể được áp dụng vào bài toán nhận diện khuôn mặt,
        phục vụ học tập, điểm danh thử nghiệm và nghiên cứu mô hình CNN.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="warning-box">
        Lưu ý: Kết quả nhận diện phụ thuộc vào chất lượng ảnh, ánh sáng, góc chụp và dữ liệu đã dùng để huấn luyện model.
    </div>
    """, unsafe_allow_html=True)

# =========================
# TRANG NHẬN DẠNG
# =========================
elif page == "📷 Nhận dạng khuôn mặt":
    st.markdown('<h1 class="main-title">Nhận dạng khuôn mặt</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-title">Tải ảnh hoặc chụp ảnh trực tiếp để hệ thống nhận diện</p>',
        unsafe_allow_html=True
    )

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
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Ảnh đầu vào")
            st.image(image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Kết quả nhận diện")

            if st.button("🔍 Bắt đầu nhận diện"):
                predicted_name, confidence, all_predictions = predict_face(image)

                st.markdown(f"""
                <div class="result-box">
                    <h2>👤 {predicted_name}</h2>
                    <h3>Độ tin cậy: {confidence:.2f}%</h3>
                </div>
                """, unsafe_allow_html=True)

                if confidence < 60:
                    st.warning("Độ tin cậy thấp. Anh nên thử ảnh rõ mặt hơn, đủ sáng hơn hoặc chính diện hơn.")

                st.markdown("### Xác suất các lớp cao nhất")

                top_indices = np.argsort(all_predictions)[::-1][:5]

                for idx in top_indices:
                    if idx < len(class_labels):
                        st.write(f"**{class_labels[idx]}**: {all_predictions[idx] * 100:.2f}%")

            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="card">
            <div class="big-text">📌 Hướng dẫn</div>
            <p class="normal-text">
            Anh hãy tải ảnh khuôn mặt lên hoặc chụp ảnh trực tiếp bằng camera.
            Ảnh nên rõ mặt, đủ sáng, không bị che khuất để model dự đoán chính xác hơn.
            </p>
        </div>
        """, unsafe_allow_html=True)