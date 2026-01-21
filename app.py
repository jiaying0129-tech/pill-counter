import streamlit as st
import cv2
import numpy as np
from PIL import Image

# --- 1. 頁面設定 ---
st.set_page_config(page_title="AI 藥丸計數器 (升級版)", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    h1 { color: #ff4b4b; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.title("💊 藥丸計數器 - 強力分離版")
st.info("💡 這個版本專門解決「藥丸黏在一起」的問題。")

# --- 2. 側邊欄：參數調整 ---
with st.expander("⚙️ 調整參數 (算不準請點我)"):
    st.write("### 1. 基礎設定")
    inverse_mode = st.checkbox("反轉顏色 (若背景是白紙請勾選)", value=False)
    binary_threshold = st.slider("亮度閾值 (區分背景與藥丸)", 0, 255, 127)
    
    st.write("### 2. 進階分離設定")
    st.write("如果不小心把很多顆算成一顆，請將下方數值調高")
    separation_force = st.slider("分離強度 (數值越大分得越開)", 0.0, 1.0, 0.5, 0.05)
    min_area = st.slider("最小面積 (過濾雜訊)", 10, 200, 50)

# --- 3. 核心處理邏輯 (升級版) ---
def process_image(img_buffer, bin_thresh, inverse, sep_force, min_area_val):
    # 讀取圖片
    file_bytes = np.asarray(bytearray(img_buffer.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # 轉灰階並模糊
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0) # 加大模糊半徑以減少雜訊
    
    # 根據背景反轉
    if inverse:
        thresh_type = cv2.THRESH_BINARY_INV
    else:
        thresh_type = cv2.THRESH_BINARY
        
    _, thresh = cv2.threshold(blurred, bin_thresh, 255, thresh_type)
    
    # 清理雜點 (開運算)
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # === 關鍵技術：距離變換 (Distance Transform) ===
    # 這步會算出每個白色像素「離黑色背景有多遠」。越中心越亮，邊緣越暗。
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    
    # 取出「山頂」：只保留最中心的部分，這樣黏在一起的邊緣就會斷開
    _, sure_fg = cv2.threshold(dist_transform, sep_force * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg) # 轉回整數格式
    
    # 找輪廓 (這次找的是分離後的「核心」)
    cnts, _ = cv2.findContours(sure_fg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    count = 0
    output_img = img.copy()
    
    for c in cnts:
        if cv2.contourArea(c) < min_area_val:
            continue
            
        count += 1
        
        # 找出核心位置
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            # 畫出標記 (紅點 + 數字)
            cv2.circle(output_img, (cX, cY), 10, (0, 0, 255), -1) 
            cv2.putText(output_img, str(count), (cX - 10, cY - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 畫個綠框示意 (這是核心大小，不代表實際藥丸邊緣)
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return count, output_img, thresh, sure_fg

# --- 4. 介面顯示 ---
img_file = st.camera_input("📸 請拍照 (請盡量靠近拍)")

if img_file is not None:
    # 執行處理
    count, result_img, debug_thresh, debug_fg = process_image(
        img_file, binary_threshold, inverse_mode, separation_force, min_area
    )
    
    # 顯示結果
    st.markdown(f"<h2 style='text-align: center; color: green;'>共發現 {count} 顆</h2>", unsafe_allow_html=True)
    st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="偵測結果 (紅點為藥丸中心)", use_container_width=True)
    
    # 除錯區 (給你看電腦是怎麼把藥丸切開的)
    with st.expander("👀 為什麼這樣算？(除錯影像)"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("1. 黑白二值化")
            st.image(debug_thresh, use_container_width=True)
        with col2:
            st.write("2. 分離後的核心")
            st.image(debug_fg, use_container_width=True, clamp=True)
            st.caption("如果這裡看起來還是黏在一起，請調高「分離強度」。")
