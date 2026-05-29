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
# CLASS LABELS ĐÚNG THỨ TỰ FOLDER TRAIN
# =========================
class_labels = [
    "Hoàng Kỳ Anh",
    "Lê Quang Dũng",
    "Lê Tuấn Thành",
    "Lương Ngọc Thuận",
    "Ngô Quốc Trung",
    "Nguyễn Ngọc Bảo",
    "Nguyễn Hoàng Quế Châu",
    "Nguyễn Phạm Hoàng An",
    "Nguyễn Thị Khánh Lê",
    "Nguyễn Thị Ngọc Tuyết",
    "Nguyễn Tiến Mạnh",
    "Nguyễn Việt Đức",
    "Nguyễn Đặng Vinh Phúc",
    "Phạm Gia Thành Duy",
    "Phạm Hứa Nhật Minh",
    "Phạm Nguyễn Bảo Châu",
    "Phạm Phú Hoà",
    "Trần Hải Yến",
    "Vũ Quang Thái",
    "Đinh Hữu Khánh Anh",
    "Đoàn Hùng",
    "Đỗ An Phúc"
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
        font-size: 35px;
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

    /* Fix chữ bị trùng nền trong upload file */
    [data-testid="stFileUploader"] label {
        color: white !important;
    }

    [data-testid="stFileUploader"] section {
        background-color: #f8fafc !important;
        border-radius: 14px !important;
    }

    [data-testid="stFileUploader"] section * {
        color: #0f172a !important;
    }

    [data-testid="stFileUploader"] small {
        color: #334155 !important;
    }

    [data-testid="stFileUploader"] button {
        background: white !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px !important;
    }

    h1, h2, h3, p, label, span {
        color: white;
    }

    .metric-row {
        padding: 14px 16px;
        border-radius: 16px;
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.10);
        margin-bottom: 10px;
        color: #e2e8f0;
    }

    .mini-card {
        padding: 20px;
        border-radius: 22px;
        background: rgba(255,255,255,0.075);
        border: 1px solid rgba(255,255,255,0.12);
        box-shadow: 0 10px 28px rgba(0,0,0,0.22);
        min-height: 120px;
        text-align: center;
        margin-bottom: 18px;
    }

    .mini-number {
        font-size: 34px;
        font-weight: 950;
        background: linear-gradient(90deg, #22d3ee, #c4b5fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }

    .mini-label {
        font-size: 15px;
        color: #cbd5e1;
        font-weight: 650;
    }

    .section-title {
        font-size: 34px;
        font-weight: 950;
        margin: 30px 0 18px 0;
        color: white;
    }

    .step-box {
        padding: 24px;
        border-radius: 24px;
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(34, 211, 238, 0.20);
        box-shadow: 0 12px 34px rgba(0,0,0,0.25);
        min-height: 170px;
        margin-bottom: 20px;
    }

    .step-number {
        width: 42px;
        height: 42px;
        border-radius: 50%;
        background: linear-gradient(135deg, #06b6d4, #8b5cf6);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 950;
        color: white;
        margin-bottom: 14px;
    }

    .empty-preview {
        padding: 40px;
        border-radius: 28px;
        background:
            linear-gradient(135deg, rgba(34,211,238,0.10), rgba(168,85,247,0.10)),
            rgba(255,255,255,0.06);
        border: 1px dashed rgba(255,255,255,0.25);
        text-align: center;
        min-height: 280px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        color: #cbd5e1;
    }

    .empty-icon {
        font-size: 70px;
        margin-bottom: 18px;
    }

    .footer-box {
        margin-top: 35px;
        padding: 20px;
        text-align: center;
        border-radius: 22px;
        background: rgba(255,255,255,0.055);
        border: 1px solid rgba(255,255,255,0.10);
        color: #94a3b8;
        font-size: 15px;
    }

    /* =========================
       CHATBOT DỄ THƯƠNG
    ========================= */
    .cute-bot-card {
        padding: 30px;
        border-radius: 32px;
        background:
            radial-gradient(circle at top left, rgba(34,211,238,0.18), transparent 35%),
            linear-gradient(135deg, rgba(255,255,255,0.10), rgba(255,255,255,0.05));
        border: 1px solid rgba(255,255,255,0.16);
        box-shadow: 0 18px 50px rgba(0,0,0,0.30);
        min-height: 520px;
        text-align: center;
    }

    .cute-bot-avatar {
        width: 145px;
        height: 145px;
        margin: 0 auto 18px auto;
        border-radius: 50%;
        background:
            radial-gradient(circle at 35% 30%, #ffffff, #93c5fd 35%, #8b5cf6 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 72px;
        box-shadow:
            0 0 35px rgba(34,211,238,0.35),
            0 16px 35px rgba(0,0,0,0.28);
        animation: floatBot 3s ease-in-out infinite;
    }

    @keyframes floatBot {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }

    .cute-bot-name {
        font-size: 34px;
        font-weight: 950;
        background: linear-gradient(90deg, #67e8f9, #c4b5fd, #facc15);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }

    .cute-bot-desc {
        font-size: 17px;
        line-height: 1.7;
        color: #dbeafe;
        margin-bottom: 22px;
    }

    .cute-suggestion {
        padding: 13px 16px;
        border-radius: 18px;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.12);
        color: #e2e8f0;
        margin-bottom: 10px;
        text-align: left;
        font-size: 15px;
    }

    .chat-panel {
        padding: 24px;
        border-radius: 32px;
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(255,255,255,0.14);
        box-shadow: 0 18px 50px rgba(0,0,0,0.30);
        min-height: 520px;
    }

    .chat-header {
        padding: 16px 18px;
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(6,182,212,0.25), rgba(139,92,246,0.25));
        border: 1px solid rgba(255,255,255,0.14);
        margin-bottom: 18px;
    }

    .chat-header-title {
        font-size: 23px;
        font-weight: 900;
        color: white;
    }

    .chat-header-sub {
        font-size: 14px;
        color: #cbd5e1;
        margin-top: 4px;
    }

    .chat-window {
        padding: 18px;
        border-radius: 24px;
        background: rgba(255,255,255,0.055);
        border: 1px solid rgba(255,255,255,0.10);
        min-height: 330px;
        max-height: 430px;
        overflow-y: auto;
        margin-bottom: 18px;
    }

    .user-bubble {
        background: linear-gradient(135deg, #06b6d4, #2563eb);
        color: white;
        padding: 14px 17px;
        border-radius: 20px 20px 4px 20px;
        margin: 12px 0 12px auto;
        max-width: 78%;
        font-size: 16px;
        line-height: 1.55;
        box-shadow: 0 8px 22px rgba(0,0,0,0.18);
    }

    .bot-bubble {
        background: rgba(255,255,255,0.10);
        color: #e2e8f0;
        padding: 14px 17px;
        border-radius: 20px 20px 20px 4px;
        margin: 12px auto 12px 0;
        max-width: 78%;
        font-size: 16px;
        line-height: 1.55;
        border: 1px solid rgba(255,255,255,0.12);
        box-shadow: 0 8px 22px rgba(0,0,0,0.14);
    }

    .bot-bubble strong {
        color: #67e8f9;
    }

    .cute-note {
        padding: 16px;
        border-radius: 20px;
        background: rgba(250,204,21,0.10);
        border: 1px solid rgba(250,204,21,0.25);
        color: #fef9c3;
        font-size: 15px;
        line-height: 1.6;
        text-align: left;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_face_model():
    model_path = "faceid_app.h5"

    if not os.path.exists(model_path):
        st.error("Không tìm thấy file faceid_app.h5. Anh hãy đặt file faceid_app.h5 cùng thư mục với app.py")
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
# CHATBOT RULE-BASED
# =========================
def chatbot_reply(user_message):
    msg = user_message.lower()

    if "chào" in msg or "hello" in msg or "hi" in msg:
        return "Xin chào anh! Em là IDeBot 🤖. Em có thể hướng dẫn anh cách dùng app nhận diện khuôn mặt."

    elif "cách dùng" in msg or "sử dụng" in msg or "dùng" in msg:
        return "Anh vào trang **Nhận dạng**, chọn **Tải ảnh lên** hoặc **Chụp bằng camera**, sau đó bấm **Bắt đầu nhận diện** để xem kết quả."

    elif "ảnh" in msg or "chụp" in msg or "camera" in msg:
        return "Ảnh nên rõ mặt, đủ sáng, khuôn mặt nằm ở giữa ảnh, không bị che bởi khẩu trang, tay hoặc vật khác."

    elif "model" in msg or "cnn" in msg:
        return "Ứng dụng IDeFace sử dụng mô hình **CNN** đã được huấn luyện từ ảnh khuôn mặt của 22 sinh viên."

    elif "độ tin cậy" in msg or "confidence" in msg:
        return "Độ tin cậy là xác suất cao nhất mà model dự đoán cho một sinh viên. Ví dụ 95% nghĩa là model khá chắc chắn với kết quả đó."

    elif "sai" in msg or "nhầm" in msg or "không đúng" in msg:
        return "Nếu nhận diện sai, anh nên thử ảnh chính diện hơn, đủ sáng hơn. Ngoài ra cần kiểm tra `class_labels` có đúng thứ tự với folder train không."

    elif "bao nhiêu" in msg or "sinh viên" in msg:
        return "Hiện tại app đang nhận diện **22 sinh viên** trong dataset đã huấn luyện."

    elif "tên app" in msg or "ideface" in msg:
        return "Tên app là **IDeFace** — ứng dụng nhận diện khuôn mặt sinh viên bằng trí tuệ nhân tạo."

    else:
        return "Em có thể hỗ trợ các nội dung như: cách dùng app, cách chụp ảnh tốt hơn, giải thích model CNN, độ tin cậy và lỗi nhận diện sai."

# =========================
# ĐIỀU HƯỚNG
# =========================
if "page" not in st.session_state:
    st.session_state.page = "home"

st.markdown("""
<div class="navbar">
    <div class="brand">IDeFace</div>
    <div class="brand-sub">AI Face Recognition System</div>
</div>
""", unsafe_allow_html=True)

nav1, nav2, nav3, nav4, nav5 = st.columns([4.6, 1.2, 1.7, 1.5, 1.3])

with nav2:
    if st.button("🏠 Home"):
        st.session_state.page = "home"

with nav3:
    if st.button("📷 Nhận dạng"):
        st.session_state.page = "detect"

with nav4:
    if st.button("🤖 Chatbot"):
        st.session_state.page = "chatbot"

with nav5:
    if st.button("ℹ️ About"):
        st.session_state.page = "about"

page = st.session_state.page

# =========================
# TRANG HOME
# =========================
if page == "home":
    left, right = st.columns([1.4, 1])

    with left:
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

    with right:
        st.markdown("""
        <div class="card" style="min-height: 335px;">
            <div class="card-title">🚀 Tổng quan hệ thống</div>
            <div class="card-text">
                IDeFace được xây dựng nhằm mô phỏng một hệ thống nhận dạng khuôn mặt đơn giản,
                trực quan và dễ sử dụng. Ứng dụng sử dụng mô hình CNN đã huấn luyện để phân loại
                khuôn mặt theo danh sách sinh viên có trong dataset.
                <br><br>
                Người dùng chỉ cần đưa ảnh vào hệ thống, sau đó AI sẽ xử lý ảnh và hiển thị
                sinh viên được dự đoán cùng độ tin cậy.
            </div>
        </div>
        """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown("""
        <div class="mini-card">
            <div class="mini-number">22</div>
            <div class="mini-label">Sinh viên</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown("""
        <div class="mini-card">
            <div class="mini-number">200×200</div>
            <div class="mini-label">Kích thước ảnh đầu vào</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown("""
        <div class="mini-card">
            <div class="mini-number">CNN</div>
            <div class="mini-label">Mô hình nhận diện</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown("""
        <div class="mini-card">
            <div class="mini-number">2</div>
            <div class="mini-label">Upload ảnh & Camera</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Tính năng chính</div>', unsafe_allow_html=True)

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
            <div class="card-title">🤖 Chatbot hỗ trợ</div>
            <div class="card-text">
                Tích hợp IDeBot giúp người dùng hiểu cách sử dụng app, chọn ảnh phù hợp và giải thích kết quả.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Quy trình hoạt động</div>', unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)

    with s1:
        st.markdown("""
        <div class="step-box">
            <div class="step-number">1</div>
            <div class="card-title">Chọn ảnh</div>
            <div class="card-text">
                Người dùng tải ảnh khuôn mặt hoặc chụp ảnh trực tiếp bằng camera.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with s2:
        st.markdown("""
        <div class="step-box">
            <div class="step-number">2</div>
            <div class="card-title">Xử lý ảnh</div>
            <div class="card-text">
                Ảnh được resize về 200x200, chuẩn hóa pixel và đưa vào mô hình CNN.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with s3:
        st.markdown("""
        <div class="step-box">
            <div class="step-number">3</div>
            <div class="card-title">Trả kết quả</div>
            <div class="card-text">
                Hệ thống hiển thị tên sinh viên được dự đoán cùng độ tin cậy.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="guide-box">
        <b>Gợi ý sử dụng:</b> Ảnh đầu vào nên rõ khuôn mặt, đủ sáng, không bị che mặt và nên chụp chính diện để tăng độ chính xác.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="footer-box">
        IDeFace • AI Face Recognition System • Built with Python, TensorFlow/Keras and Streamlit
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

    tab1, tab2 = st.tabs(["📁 Tải ảnh lên", "📸 Chụp bằng camera"])

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
        <div class="guide-box">
            Chụp ảnh khuôn mặt trực tiếp bằng camera. Anh nên căn mặt ở giữa khung, đủ sáng và không bị che mặt.
        </div>
        """, unsafe_allow_html=True)

        camera_file = st.camera_input("📸 Chụp ảnh khuôn mặt")

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
                    <div class="result-name">Đây là sinh viên: 👤 {predicted_name}</div>
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
        c1, c2 = st.columns([1, 1])

        with c1:
            st.markdown("""
            <div class="empty-preview">
                <div class="empty-icon">🖼️</div>
                <div class="card-title">Chưa có ảnh đầu vào</div>
                <div class="card-text">
                    Anh hãy tải ảnh khuôn mặt lên hoặc chụp ảnh trực tiếp bằng camera để bắt đầu nhận diện.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown("""
            <div class="card" style="min-height: 280px;">
                <div class="card-title">📌 Mẹo để nhận diện tốt hơn</div>
                <div class="card-text">
                    • Chụp ảnh rõ mặt, không bị mờ.<br>
                    • Khuôn mặt nên nằm ở giữa ảnh.<br>
                    • Tránh ảnh quá tối hoặc ngược sáng.<br>
                    • Không che mặt bằng khẩu trang, tay hoặc vật khác.<br>
                    • Nên dùng ảnh chính diện để độ chính xác cao hơn.
                </div>
            </div>
            """, unsafe_allow_html=True)

# =========================
# TRANG CHATBOT
# =========================
elif page == "chatbot":
    st.markdown("""
    <div class="hero">
        <div class="tag">Cute AI Assistant</div>
        <div class="hero-title">IDeBot</div>
        <div class="hero-desc">
            Trợ lý chatbot dễ thương hỗ trợ người dùng trong quá trình sử dụng ứng dụng IDeFace.
        </div>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1, 1.45])

    with left:
        st.markdown("""
        <div class="cute-bot-card">
            <div class="cute-bot-avatar">🤖</div>
            <div class="cute-bot-name">IDeBot</div>
            <div class="cute-bot-desc">
                Xin chào! Em là chatbot hỗ trợ của IDeFace.
                Em sẽ giúp anh hiểu cách dùng app, cách chọn ảnh và giải thích kết quả nhận diện.
            </div>

            <div class="cute-suggestion">💡 Cách dùng app như thế nào?</div>
            <div class="cute-suggestion">📷 Ảnh như thế nào thì nhận diện tốt?</div>
            <div class="cute-suggestion">🧠 Model CNN là gì?</div>
            <div class="cute-suggestion">📊 Độ tin cậy có ý nghĩa gì?</div>

            <div class="cute-note">
                Mẹo nhỏ: Hãy hỏi chatbot bằng các câu ngắn gọn như 
                <b>“cách dùng app”</b>, <b>“độ tin cậy là gì”</b> hoặc 
                <b>“vì sao nhận diện sai”</b>.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div class="chat-panel">
            <div class="chat-header">
                <div class="chat-header-title">💬 Trò chuyện với IDeBot</div>
                <div class="chat-header-sub">Trợ lý hỗ trợ sử dụng app nhận diện khuôn mặt</div>
            </div>
        """, unsafe_allow_html=True)

        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = [
                {
                    "role": "bot",
                    "content": "Xin chào anh! Em là **IDeBot** 🤖. Anh muốn hỏi gì về app IDeFace?"
                }
            ]

        st.markdown('<div class="chat-window">', unsafe_allow_html=True)

        for message in st.session_state.chat_messages:
            if message["role"] == "user":
                st.markdown(
                    f'<div class="user-bubble">{message["content"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="bot-bubble">🤖 {message["content"]}</div>',
                    unsafe_allow_html=True
                )

        st.markdown('</div>', unsafe_allow_html=True)

        user_input = st.chat_input("Nhập câu hỏi của anh tại đây...")

        if user_input:
            st.session_state.chat_messages.append(
                {"role": "user", "content": user_input}
            )

            bot_answer = chatbot_reply(user_input)

            st.session_state.chat_messages.append(
                {"role": "bot", "content": bot_answer}
            )

            st.rerun()

        if st.button("🧹 Xóa đoạn chat"):
            st.session_state.chat_messages = [
                {
                    "role": "bot",
                    "content": "Đoạn chat đã được làm mới. Anh cần em hỗ trợ gì tiếp theo?"
                }
            ]
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

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

    a1, a2, a3 = st.columns(3)

    with a1:
        st.markdown("""
        <div class="card">
            <div class="card-title">🎯 Mục tiêu</div>
            <div class="card-text">
                Hỗ trợ nhận diện sinh viên lớp Logtech bán phần thông qua ảnh khuôn mặt,
                đồng thời thể hiện khả năng ứng dụng trí tuệ nhân tạo vào bài toán nhận dạng.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with a2:
        st.markdown("""
        <div class="card">
            <div class="card-title">🧩 Công nghệ</div>
            <div class="card-text">
                Python, TensorFlow/Keras, mô hình CNN, thư viện PIL, NumPy và Streamlit để xây dựng giao diện web.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with a3:
        st.markdown("""
        <div class="card">
            <div class="card-title">📊 Dữ liệu</div>
            <div class="card-text">
                Dataset gồm ảnh khuôn mặt của 22 sinh viên, được chia thành tập huấn luyện và tập kiểm tra.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Thông tin hệ thống</div>', unsafe_allow_html=True)

    b1, b2, b3, b4 = st.columns(4)

    with b1:
        st.markdown("""
        <div class="mini-card">
            <div class="mini-number">22</div>
            <div class="mini-label">Lớp nhận diện</div>
        </div>
        """, unsafe_allow_html=True)

    with b2:
        st.markdown("""
        <div class="mini-card">
            <div class="mini-number">200×200</div>
            <div class="mini-label">Input model</div>
        </div>
        """, unsafe_allow_html=True)

    with b3:
        st.markdown("""
        <div class="mini-card">
            <div class="mini-number">H5</div>
            <div class="mini-label">Định dạng model</div>
        </div>
        """, unsafe_allow_html=True)

    with b4:
        st.markdown("""
        <div class="mini-card">
            <div class="mini-number">Web</div>
            <div class="mini-label">Giao diện Streamlit</div>
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

    st.markdown("""
    <div class="footer-box">
        IDeFace • Student Face Recognition App • Developed for AI learning project
    </div>
    """, unsafe_allow_html=True)
