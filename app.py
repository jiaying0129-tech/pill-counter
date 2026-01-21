import streamlit as st
import cv2
import numpy as np
from PIL import Image

# --- 1. 頁面設定 ---
st.set_page_config(page_title="AI 藥丸計數器 (高對比版)", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    h1 { color: #d63384; text-align: center; }
    .stButton>button { width: 100%; border-radius: 20px; }
    </style>
""", unsafe_allow_html=True)

st.title("💊 藥丸計數器 - 高對比版")
st.info("💡 針對「粉色藥丸+深色蓋子」優化。請使用下方的「裁切」功能去除木紋背景。")

# --- 2. 側邊欄：參數調整 ---
with st.expander("⚙️ 調整參數 (算不準請點我)", expanded=True):
    st.write("### 1. 範圍設定")
    crop_size = st.slider("裁切邊緣 (去除背景)", 0, 200, 50, help="數值越大，切掉的邊緣越多")
    
    st.write("### 2. 影像增強")
    use_green_channel = st.checkbox("開啟「綠色濾鏡」 (粉紅藥丸推薦)", value=True, help="粉紅色在綠色濾鏡下對比最強")
    contrast_boost = st.slider("對比度增強 (CLAHE)", 0.0, 10.0, 3.0)
    
    st.write("### 3. 分離設定")
    block_size = st.slider("區域偵測大小 (奇數)", 3, 51, 15, step=2, help="越小越能抓到細節，但也容易抓到雜訊")
    separation_force = st.slider("分離強度", 0.0, 1.0, 0.4)
    min_area = st.slider("最小面積 (過濾雜訊)", 10, 500, 100)

# --- 3. 核心處理邏輯 ---
def process_image(img_buffer, crop_val, use_green, contrast, blk_size, sep_force, min_area_val):
    # 讀取圖片
    file_bytes = np.asarray(bytearray(img_buffer.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # 1. 裁切影像 (去除木紋背景)
    h, w = img.shape[:2]
    if crop_val > 0:
        img = img[crop_val:h-crop_val, crop_val:w-crop_val]
    
    # 2. 顏色通道選擇 (關鍵步驟)
    if use_green:
        # 粉紅色 = 高紅 + 高藍 + 中綠
        # 紅色蓋子 = 高紅 + 低綠 + 低藍
        # 取綠色通道，通常能讓粉紅藥丸(較亮)跟深紅蓋子(較暗)分開
        b, g, r = cv2.split(img)
        gray = g
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. 增強對比 (CLAHE)
    # 這能讓陰影裡的藥丸顯現出來
    clahe = cv2.createCLAHE(clipLimit=contrast, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # 4. 高斯模糊 (減少噪點)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    
    # 5. 適應性閾值 (Adaptive Threshold)
    # 自動根據區域光線決定黑白，不再用全域固定數值
    thresh = cv2.adaptiveThreshold(
        blurred, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        blk_size, 2
    )
    
    # 6. 形態學清理 (去除小白點)
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 7. 距離變換 (分離黏在一起的藥丸)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, sep_force * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # 8. 找輪廓
    cnts, _ = cv2.findContours(sure_fg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    count = 0
    output_img = img.copy()
    
    for c in cnts:
        if cv2.contourArea(c) < min_area_val:
            continue
            
        count += 1
        
        # 找中心點
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            # 繪製結果
            cv2.circle(output_img, (cX, cY), 8, (0, 0, 255), -1) 
            cv2.putText(output_img, str(count), (cX - 10, cY - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return count, output_img, gray, sure_fg

# --- 4. 介面顯示 ---
img_file = st.camera_input("📸 請盡量將藥丸放在畫面正中間")

if img_file is not None:
    count, result_img, debug_gray, debug_fg = process_image(
        img_file, crop_size, use_green_channel, contrast_boost, block_size, separation_force, min_area
    )
    
    st.markdown(f"<h2 style='text-align: center; color: #d63384;'>共發現 {count} 顆</h2>", unsafe_allow_html=True)
    st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="最終結果", use_container_width=True)
    
    # 除錯區
    with st.expander("👀 電腦看到了什麼？ (除錯影像)"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("1. 綠色濾鏡+對比增強")
            st.write("確認藥丸在這裡是否比背景亮？")
            st.image(debug_gray, use_container_width=True)
        with col2:
            st.write("2. 最終識別區域")
            st.write("確認白點是否分開？")
            st.image(debug_fg, use_container_width=True, clamp=True)
