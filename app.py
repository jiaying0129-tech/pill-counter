import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="藥丸計數器 (山頂版)", layout="centered")
st.title("💊 藥丸計數器 - 山頂分離版")
st.info("💡 這個版本專門對付「黏在一起」的藥丸。它計算的是藥丸的中心點（山頂），而不是輪廓。")

# --- 參數區 ---
with st.expander("🎛️ 調整參數 (請依照下方教學)", expanded=True):
    st.write("### 1. 第一步：讓藥丸變白")
    # 綠色通道對粉紅藥丸/紅蓋子的分離效果最好
    use_green = st.checkbox("使用綠色濾鏡 (推薦粉/紅藥丸)", value=True)
    thresh_val = st.slider("亮度門檻", 0, 255, 140, help="調整直到藥丸變成白色，背景變黑")
    
    st.write("### 2. 第二步：切開它們")
    # 這是核心：距離變換的閾值
    peak_threshold = st.slider("山頂分離度 (關鍵)", 0.1, 1.0, 0.5, step=0.05, help="數值越高，只保留越中心的點（切得越開）；數值越低，保留越多邊緣")
    
    st.write("### 3. 範圍過濾")
    crop_center = st.slider("裁切周圍 (去除木紋)", 0, 200, 80)
    min_area = st.number_input("最小山頂面積", value=10)

def process_image(img_buffer):
    # 1. 讀取
    file_bytes = np.asarray(bytearray(img_buffer.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # 2. 裁切 (把討厭的木紋切掉)
    h, w = img.shape[:2]
    if crop_center > 0:
        img = img[crop_center:h-crop_center, crop_center:w-crop_center]
    
    # 3. 預處理
    if use_green:
        # 取綠色通道 (粉紅藥丸在綠色通道會很亮，紅色蓋子會變暗)
        b, g, r = cv2.split(img)
        gray = g
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    # 增強對比 (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # 高斯模糊
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    
    # 4. 二值化 (造出那坨幸運草)
    _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
    
    # 形態學清理 (把洞補起來)
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 5. 距離變換 (計算山高)
    # 算出每個白點離黑色背景有多遠。越中心越亮。
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    
    # 正規化以便顯示
    dist_display = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)
    
    # 6. 尋找山頂 (Thresholding the Distance Map)
    # 這是關鍵！我們只取距離變換圖中最亮的那 X%
    _, peaks = cv2.threshold(dist_transform, peak_threshold * dist_transform.max(), 255, 0)
    peaks = np.uint8(peaks)
    
    # 7. 計算山頂數量
    cnts, _ = cv2.findContours(peaks.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    count = 0
    output_img = img.copy()
    
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
            
        count += 1
        
        # 找中心
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            # 畫個點代表算到了
            cv2.circle(output_img, (cX, cY), 8, (0, 0, 255), -1)
            cv2.putText(output_img, str(count), (cX-5, cY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return count, output_img, binary, dist_display, peaks

# --- 介面 ---
img_file = st.camera_input("📸 請拍照")

if img_file is not None:
    count, result_img, bin_img, dist_img, peak_img = process_image(img_file)
    
    st.markdown(f"<h1 style='text-align: center; color: red;'>共 {count} 顆</h1>", unsafe_allow_html=True)
    st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="最終結果", use_container_width=True)
    
    st.write("---")
    st.subheader("🧐 為什麼這樣算？ (除錯區)")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(bin_img, caption="1. 黏在一起的藥丸", use_container_width=True)
        st.caption("調整「亮度門檻」，讓這裡變成一坨白色的塊狀。")
        
    with col2:
        st.image(dist_img, caption="2. 能量圖 (越亮越高)", use_container_width=True, clamp=True)
        st.caption("電腦計算中心點。你看得出有 4 個亮點嗎？")
        
    with col3:
        st.image(peak_img, caption="3. 只留山頂", use_container_width=True)
        st.caption("調整「山頂分離度」。調高 = 點變小(分開)；調低 = 點變大(黏住)。")
