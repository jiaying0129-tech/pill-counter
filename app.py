import streamlit as st
import cv2
import numpy as np

# --- 1. 頁面設定 ---
st.set_page_config(page_title="全自動藥丸計數器", layout="centered")
st.title("💊 全自動藥丸計數器")
st.info("⚡️ 這是「全自動版」。它會自動分析紅/藍/綠光，找出藥丸最明顯的那個顏色來運算。")

# --- 2. 核心邏輯：全自動分析 ---
def auto_process(img_buffer):
    # 讀取
    file_bytes = np.asarray(bytearray(img_buffer.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # 1. 自動裁切：只保留中心 60% (強制避開周圍木紋)
    h, w = img.shape[:2]
    crop_rate = 0.2 # 上下左右各切掉 20%
    y1, y2 = int(h * crop_rate), int(h * (1 - crop_rate))
    x1, x2 = int(w * crop_rate), int(w * (1 - crop_rate))
    cropped = img[y1:y2, x1:x2]
    
    # 2. 光譜分離 (關鍵！)
    # 我們把圖片拆成 B(藍), G(綠), R(紅) 三個通道
    # 粉紅藥丸 = 高紅 + 高藍
    # 紅色蓋子 = 高紅 + 低藍
    # 所以「藍色通道」是唯一能把粉紅藥丸跟紅蓋子分開的關鍵！
    b, g, r = cv2.split(cropped)
    
    # 計算每個通道的「標準差」(代表對比度)
    # 我們選擇對比最強的那個通道 (通常是藍色或綠色)
    channels = {'Blue': b, 'Green': g, 'Red': r}
    best_channel_name = max(channels, key=lambda k: np.std(channels[k]))
    gray = channels[best_channel_name]
    
    # 3. 增強對比 (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # 4. 雙邊濾波 (磨皮，去除殘留雜訊)
    blurred = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # 5. 自動閾值 (Otsu's Binarization)
    # 這一步完全取代手動滑桿，讓電腦自己算黑白分界線
    thresh_val, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 6. 距離變換 + 自動找山頂
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    
    # 自動尋找局部最大值 (Local Maxima) 代替手動閾值
    # 先做膨脹，如果一個點膨脹後跟原本一樣，那它就是局部最高點
    kernel_size = 15 # 這個值決定了「多近的兩顆算一顆」
    dilated = cv2.dilate(dist_transform, np.ones((kernel_size, kernel_size)))
    peaks = (dist_transform == dilated) & (dist_transform > 0.3 * dist_transform.max())
    peaks = np.uint8(peaks)
    
    # 7. 計數
    cnts, _ = cv2.findContours(peaks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    count = 0
    output_img = cropped.copy()
    
    for c in cnts:
        # 過濾太小的雜訊點
        if cv2.contourArea(c) < 2: 
            continue
        count += 1
        
        # 標記
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(output_img, (cX, cY), 10, (0, 0, 255), -1)
            cv2.circle(output_img, (cX, cY), 30, (0, 255, 0), 2)
            cv2.putText(output_img, str(count), (cX-10, cY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    return count, output_img, best_channel_name, enhanced, binary

# --- 3. 介面 ---
img_file = st.camera_input("📸 請直接拍照 (無需調整)")

if img_file is not None:
    # 直接執行，不給參數
    count, result, channel_used, debug_gray, debug_bin = auto_process(img_file)
    
    st.success(f"✅ AI 判定使用「{channel_used} 光譜」分析最佳")
    st.markdown(f"<h1 style='text-align: center; color: blue;'>共發現 {count} 顆</h1>", unsafe_allow_html=True)
    
    st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="偵測結果", use_container_width=True)
    
    with st.expander("🔍 電腦的思考過程 (除錯)"):
        col1, col2 = st.columns(2)
        with col1:
            st.image(debug_gray, caption=f"1. 自動選用的{channel_used}光", use_container_width=True)
            st.write("粉紅藥丸在這裡應該最亮")
        with col2:
            st.image(debug_bin, caption="2. 自動二值化", use_container_width=True)
            st.write("白色的區塊代表藥丸")
