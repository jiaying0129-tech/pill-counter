import streamlit as st
import cv2
import numpy as np

# --- 1. 頁面設定 ---
st.set_page_config(page_title="AI 藥丸計數器 (色彩鎖定版)", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    h1 { color: #4B0082; text-align: center; }
    .stButton>button { width: 100%; border-radius: 20px; }
    </style>
""", unsafe_allow_html=True)

st.title("💊 藥丸計數器 - 色彩鎖定版")
st.info("💡 這個版本專門對付「木紋背景」和「複雜雜訊」。請依照下方教學調整顏色滑桿。")

# --- 2. 側邊欄：強大的除錯參數 ---
with st.expander("🛠️ 參數調整 (第一步請先調這裡)", expanded=True):
    st.write("### 1. 範圍限制")
    mask_radius = st.slider("圓形遮罩大小 (去除角落背景)", 0.1, 1.0, 0.85, help="只保留畫面中心圓圈內的影像，周圍塗黑")
    
    st.write("### 2. 顏色過濾 (HSV)")
    st.write("調整下方滑桿，直到**只有藥丸是白色，背景全黑**")
    # 預設值針對淺粉/白色藥丸優化
    h_min = st.slider("色調下限 (H-min)", 0, 179, 0)
    h_max = st.slider("色調上限 (H-max)", 0, 179, 179)
    s_min = st.slider("飽和度下限 (S-min)", 0, 255, 0)
    s_max = st.slider("飽和度上限 (S-max)", 0, 255, 100) # 藥丸通常飽和度低(偏白)
    v_min = st.slider("亮度下限 (V-min)", 0, 255, 140) # 藥丸通常很亮
    v_max = st.slider("亮度上限 (V-max)", 0, 255, 255)

    st.write("### 3. 形狀優化")
    fill_holes = st.checkbox("填補藥丸孔洞", value=True, help="如果藥丸中間被誤判成黑色，請勾選此項")
    min_area = st.slider("最小面積 (過濾雜點)", 10, 500, 150)
    sep_force = st.slider("分離強度 (分開黏住的藥丸)", 0.0, 1.0, 0.5)

# --- 3. 核心處理邏輯 ---
def process_image(img_buffer):
    # 讀取
    file_bytes = np.asarray(bytearray(img_buffer.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    h, w = img.shape[:2]
    
    # === 步驟 1: 圓形遮罩 (強制去除角落木紋) ===
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (int(w//2), int(h//2))
    radius = int(min(h, w) / 2 * mask_radius)
    cv2.circle(mask, center, radius, 255, -1)
    
    # 套用遮罩：遮罩外變全黑
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    # === 步驟 2: HSV 顏色過濾 ===
    hsv = cv2.cvtColor(masked_img, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([h_min, s_min, v_min])
    upper_bound = np.array([h_max, s_max, v_max])
    
    # 產生二值化圖 (符合顏色的變白，其餘變黑)
    thresh = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # === 步驟 3: 形態學處理 (修補) ===
    kernel = np.ones((5,5), np.uint8)
    
    # 先閉運算 (把藥丸內部的小洞補起來)
    if fill_holes:
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # 進階填洞：尋找輪廓並把內部塗白
        contours_fill, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours_fill:
            cv2.drawContours(thresh, [c], 0, 255, -1)

    # 開運算 (去除背景小白點雜訊)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # === 步驟 4: 分水嶺演算法 (切開黏住的藥丸) ===
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, sep_force * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # === 步驟 5: 最終計數 ===
    cnts, _ = cv2.findContours(sure_fg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    count = 0
    output_img = img.copy()
    
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
            
        count += 1
        # 找中心並畫圖
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(output_img, (cX, cY), 10, (0, 0, 255), -1)
            cv2.putText(output_img, str(count), (cX-10, cY-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 畫框
            x, y, w_rect, h_rect = cv2.boundingRect(c)
            cv2.rectangle(output_img, (x, y), (x + w_rect, y + h_rect), (0, 255, 0), 2)

    return count, output_img, masked_img, thresh, sure_fg

# --- 4. 介面顯示 ---
img_file = st.camera_input("📸 請拍照")

if img_file is not None:
    count, result_img, masked_view, binary_view, core_view = process_image(img_file)
    
    st.markdown(f"<h2 style='text-align: center; color: green;'>共發現 {count} 顆</h2>", unsafe_allow_html=True)
    st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="最終結果", use_container_width=True)
    
    st.write("---")
    st.subheader("👀 調整教學 (必看！)")
    st.write("請依照下方三個影像來調整滑桿，直到**中間那張圖**變得很完美。")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**1. 範圍遮罩**")
        st.image(cv2.cvtColor(masked_view, cv2.COLOR_BGR2RGB), caption="只看中間", use_container_width=True)
        st.caption("調整 `圓形遮罩大小`，把周圍的木紋切掉。")
    
    with col2:
        st.write("**2. 顏色過濾 (最重要)**")
        st.image(binary_view, caption="黑白二值圖", use_container_width=True)
        st.caption("調整 `S-max` (飽和度) 和 `V-min` (亮度)。目標：**藥丸全白，背景全黑**。")

    with col3:
        st.write("**3. 最終核心**")
        st.image(core_view, caption="計數核心", use_container_width=True, clamp=True)
        st.caption("這是電腦最後數的點。")
