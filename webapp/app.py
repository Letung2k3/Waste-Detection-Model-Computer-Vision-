import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# =====================
# CONFIG
# =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "training",
    "runs",
    "waste_detection_cpu",
    "weights",
    "best.pt"
)

st.set_page_config(
    page_title="Phân loại Rác Thải YOLO",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    /* Dark background toàn app */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;  /* Chữ chính trắng sáng */
    }

    /* Header cố định */
    .fixed-header {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 80px;
        background-color: #0e1117;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
        border-bottom: 1px solid #262730;
        z-index: 999;
        padding-top: 10px;
    }
    .header-title {
        font-size: 2rem;
        font-weight: bold;
        color: #4caf50;
        margin: 0;
    }
    .header-subtitle {
        font-size: 1rem;
        color: #bbbbbb;
        margin: 5px 0 0 0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1e2127;
        border-right: 1px solid #262730;
    }
    .sidebar-title {
        font-size: 1.4rem;
        font-weight: bold;
        color: #4caf50;
        text-align: center;
        margin: 2rem 0 1.5rem 0;
    }

    /* Card chung */
    .custom-card {
        background-color: #262730;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }

    /* Container ảnh */
    .image-container {
        background-color: #262730;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .image-caption {
        margin-top: 1rem;
        font-weight: 600;
        color: #ffffff;  /* Caption trắng sáng */
        font-size: 1.1rem;
    }

    /* Label của uploader, slider, text... đều trắng */
    .stFileUploader label,
    .stSlider label,
    div[data-testid="stCaption"],
    .stMarkdown,
    p, div, span {
        color: #ffffff !important;
    }

    /* Placeholder text khi chưa upload */
    .placeholder-text {
        color: #aaaaaa;
        font-size: 1.5rem;
    }

    /* Padding main content */
    .main-content {
        margin-top: 100px;
        padding: 0 2rem;
    }

    /* Ẩn header/footer mặc định Streamlit */
    #MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Fixed Header
st.markdown("""
<div class="fixed-header">
    <h1 class="header-title">🌿 Phân loại Rác Thải YOLO</h1>
    <p class="header-subtitle">YOLOv8 - Phân loại rác thải thông minh</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()

# =====================
# SIDEBAR
# =====================
with st.sidebar:
    st.markdown("<div class='sidebar-title'>Phân loại rác thải bằng YOLO</div>", unsafe_allow_html=True)

    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload ảnh rác", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is not None:
        conf = st.slider("Ngưỡng tin cậy (Confidence threshold)", 0.0, 1.0, 0.50, 0.05)
    else:
        conf = st.slider("Ngưỡng tin cậy (Confidence threshold)", 0.0, 1.0, 0.50, 0.05, disabled=True)

    st.markdown("</div>", unsafe_allow_html=True)

# =====================
# MAIN CONTENT
# =====================
st.markdown("<div class='main-content'>", unsafe_allow_html=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='image-container'>", unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown("<div class='image-caption'>Ảnh gốc</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        with st.spinner("Đang phân loại rác thải..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                image.save(tmp.name)
                results = model(tmp.name, conf=conf)[0]

        result_img = results.plot()
        result_pil = Image.fromarray(result_img[..., ::-1])

        st.markdown("<div class='image-container'>", unsafe_allow_html=True)
        st.image(result_pil, use_container_width=True)
        st.markdown("<div class='image-caption'>Kết quả phân loại</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        os.unlink(tmp.name)
else:
    st.markdown("""
    <div style='text-align: center; margin-top: 100px;'>
        <p class='placeholder-text'>← Hãy upload ảnh rác ở sidebar để bắt đầu phân loại</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)