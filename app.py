import streamlit as st
import cv2
from utils import full_pipeline
from info import decode_egyptian_id

st.set_page_config(
    page_title="Egyptian Extract ID",
    page_icon="🪪",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1100px;
}

.title {
    text-align: center;
    font-size: 32px;
    font-weight: 600;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 15px;
    color: #666;
    margin-bottom: 30px;
}

.section {
    background: #f9fafb;
    padding: 25px;
    border-radius: 12px;
    border: 1px solid #eee;
    margin-bottom: 25px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>Egyptian Extract National ID Number</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload an ID card image to extract the national ID number</div>", unsafe_allow_html=True)


st.markdown("<div class='section'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:
    img = uploaded_file.read()

    with st.spinner("**Processing image...**"):
        try:
            corrected_skew, id_cropped, nid = full_pipeline(img)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("<div class='section'>", unsafe_allow_html=True)
                st.markdown("**ID Card**")
                st.image(corrected_skew, use_container_width=True)

                st.markdown("**NID Number**")
                st.image(cv2.cvtColor(id_cropped, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with col2:
                st.markdown("<div class='section'>", unsafe_allow_html=True)
                st.markdown("**Extracted National ID**")

                st.code(nid if nid else "No ID detected", language=None)

                if nid and len(nid) == 14:
                    st.success("Valid National ID (14 digits)")
                    decoded_info = decode_egyptian_id(nid)
                    st.markdown(f"**Birth Date:** {decoded_info['Birth Date']}")
                    st.markdown(f"**Governorate:** {decoded_info['Governorate']}")
                    st.markdown(f"**Gender:** {decoded_info['Gender']}")
                else:
                    st.error(f"Invalid ID ({len(nid) if nid else 0} digits detected)")

                st.markdown("</div>", unsafe_allow_html=True)

        except ValueError as e:
            st.error(str(e))
            st.info("Please upload a clearer image.")


st.markdown("---")
st.markdown("<p style='text-align:center; color:#777;'>Developed by <strong>Hayoma💙</strong></p>",unsafe_allow_html=True)