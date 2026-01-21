import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="AI 藥丸計數器", layout="centered")

st.title("💊 隨手拍數藥丸")
st.info("💡 提示：請將藥丸放在素色背景上，避免重疊。")

# 參數調整區
with st.expander("⚙️ 進階設定"):
    min_area = st.slider("最小藥丸面積", 50, 500, 150)
    binary_threshold = st.slider("二值化閾值", 0, 255, 127)
    inverse_mode = st.checkbox("反轉顏色模式 (黑藥丸白背景)", value=False)

def process_image(img_buffer, min_area_val, bin_thresh_val, inverse):
    file_bytes = np.asarray(bytearray(img_buffer.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    
    thresh_type = cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
    _, thresh = cv2.threshold(blurred, bin_thresh_val, 255, thresh_type)
    
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    erode = cv2.erode(dilated, kernel, iterations=1)
    
    cnts, _ = cv2.findContours(erode.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    count = 0
    output_img = img.copy()
    
    for c in cnts:
        if cv2.contourArea(c) < min_area_val:
            continue
        count += 1
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 4)
        cv2.putText(output_img, str(count), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return count, output_img

img_file = st.camera_input("📸 點擊拍照")

if img_file is not None:
    pill_count, result_img = process_image(img_file, min_area, binary_threshold, inverse_mode)
    st.markdown(f"<h2 style='text-align: center; color: green;'>共發現 {pill_count} 顆</h2>", unsafe_allow_html=True)
    st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="偵測結果", use_container_width=True)