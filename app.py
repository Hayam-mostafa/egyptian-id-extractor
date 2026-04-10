import streamlit as st
import time
from utils import full_pipeline  
from info import decode_egyptian_id

st.set_page_config(
    page_title="NID Extractor",
    layout="centered",
    page_icon="🪪"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;500;700&display=swap');

.stApp {
    background: radial-gradient(circle at top, #0b1220, #050814);
    font-family: 'Inter', sans-serif;
    color: white;
}

.title-box {
    text-align: center;
    padding: 18px;
    border-radius: 16px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 0 25px rgba(0,255,255,0.08);
    margin-bottom: 15px;
}

.title-box h1 {
    font-size: 26px;
    margin: 0;
    color: white;
    font-weight: 700;
}

.stFileUploader section {
    border-radius: 16px !important;
    border: 2px dashed rgba(0, 224, 255, 0.6) !important;
    background: rgba(255,255,255,0.03) !important;
    backdrop-filter: blur(5px);
}

.stFileUploader section:hover {
    border-color: #00e5ff !important;
    background: rgba(0, 224, 255, 0.05) !important;
}

.stFileUploader label {
    color: white !important;
    font-weight: 600 !important;
}

.stFileUploader button {
    background: rgba(255, 255, 255, 0.06) !important;
    color: white !important;
    border: 1px solid rgba(0, 224, 255, 0.4) !important;
    border-radius: 10px !important;
    margin-bottom: 10px;
}

.image-box {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(0,255,255,0.2);
    box-shadow: 0 10px 25px rgba(0,0,0,0.5);
}

.result-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    padding: 20px;
    border-radius: 16px;
    text-align: center;
    backdrop-filter: blur(12px);
    box-shadow: 0 0 25px rgba(124,77,255,0.15);
}

.id-number {
    font-size: 28px;
    font-weight: 700;
    letter-spacing: 3px;
    margin-top: 10px;
    background: linear-gradient(90deg,#00e5ff,#7c4dff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.info {
    display: flex;
    justify-content: space-between;
    padding: 10px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}

.label { color: #94a3b8; font-size: 13px; }
.value { color: white; font-weight: 600; }

.footer {
    text-align: center;
    color: #64748b;
    font-size: 15px;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="title-box">
    <h1>🪪 National ID Extractor</h1>
    <p style="color:#94a3b8;">Scan and extract ID Automatically</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Please Upload Front Side ID Card Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    img_content = uploaded_file.read()

    with st.spinner("Scanning ID..."):
        progress_bar = st.progress(0)

        try:
            progress_bar.progress(10)
            time.sleep(0.2)

            progress_bar.progress(40)
            corrected_skew, id_cropped, nid = full_pipeline(img_content)

            progress_bar.progress(80)
            info_data = decode_egyptian_id(nid)

            progress_bar.progress(100)
            progress_bar.empty()

            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<div class="image-box">', unsafe_allow_html=True)
                st.image(img_content, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="result-card">
                    <div style="color:#94a3b8; font-size:13px;">EXTRACTED ID</div>
                    <div class="id-number">{nid if nid else "Not Detected"}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                st.markdown(f"""
                <div class="result-card">
                    <div class="info"><span class="value">{info_data['Birth Date']}</span><span class="label">Birth Date</span></div>
                    <div class="info"><span class="value">{info_data['Governorate']}</span><span class="label">Governorate</span></div>
                    <div class="info"><span class="value">{info_data['Gender']}</span><span class="label">Gender</span></div>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Analysis Failed: {str(e)}")

st.markdown('<div class="footer">Developed by Hayoma🤍</div>', unsafe_allow_html=True)