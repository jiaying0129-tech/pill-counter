import streamlit as st
import cv2
import numpy as np

# --- 1. 設定 ---
st.set_page_config(page_title="抗干擾藥丸計數器", layout="centered")
st.title("💊 藥丸計數器 - 終極抗干擾版")
st.info("⚡️ 此版本使用「局部適應性演算法」，專門對付容器邊緣反光與藥丸黏連的問題。")

# --- 2. 參數 (雖然是自動，但保留微調給極端狀況) ---
with st.expander("🛠️ 如果還是切不開，請點這裡微調"):
    # 預設裁切範圍加大，強制只看中心，避開容器邊緣
    mask_size = st.slider("視野範圍 (只看中間)", 0.3, 0.9, 0.65, help="數值越小，視野越窄，越能避開容器邊緣")
    # 這是分開藥丸的關鍵
    peak_sensitivity = st.slider("藥丸分離度", 0.1, 1.0, 0.4, help="數值越大，分得越開")

# --- 3. 核心處理邏輯 ---
def process_image(img_buffer):
    # 1. 讀取圖片
    file_bytes = np.asarray(bytearray(img_buffer.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # 2. 建立圓形遮罩 (Spotlight) - 關鍵步驟！
    # 直接把照片周圍塗黑，只留最中間，這樣容器邊緣就會被蓋掉
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (int(w//2), int(h//2))
    radius = int(min(h, w) * mask_size / 2)
    cv2.circle(mask, center, radius, 255, -1)
    
    # 3. 取得綠色通道 (對粉紅/白藥丸對比最強)
    b, g, r = cv2.split(img)
    gray = g # 使用綠色通道作為基底
    
    # 4. 增強對比 (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # 5. 局部適應性閾值 (Adaptive Threshold) - 核心升級！
    # 這行程式碼會自動計算每個小區域的黑白分界，不再受整體光線影響
    # Block Size = 25 (奇數), C = 3 (常數調整)
    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, 25, 3)
    
    # 6. 套用遮罩 (把容器邊緣切掉)
    binary = cv2.bitwise_and(binary, binary, mask=mask)
    
    # 7. 形態學清理 (去除雜訊，修補藥丸內部)
    # 先腐蝕掉細小的雜訊(如木紋殘留)，再膨脹回來
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    # 閉運算把藥丸裡面的字(如刻痕)補滿
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # 8. 距離變換 (找山頂)
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    
    # 9. 尋找極大值 (藥丸中心)
    _, peaks = cv2.threshold(dist_transform, peak_sensitivity * dist_transform.max(), 255, 0)
    peaks = np.uint8(peaks)
    
    # 10. 計數
    cnts, _ = cv2.findContours(peaks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    count = 0
    output_img = img.copy()
    
    for c in cnts:
        # 過濾太小的雜訊點 (例如沒切乾淨的渣渣)
        if cv2.contourArea(c) < 5: 
            continue
            
        count += 1
        # 標記
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # 畫出中心點
            cv2.circle(output_img, (cX, cY), 10, (0, 0, 255), -1)
            # 畫出大概範圍
            cv2.circle(output_img, (cX, cY), 25, (0, 255, 0), 2)
            cv2.putText(output_img, str(count), (cX-10, cY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    return count, output_img, binary, peaks

# --- 4. 介面顯示 ---
img_file = st.camera_input("📸 請將藥丸置於畫面正中間")

if img_file is not None:
    count, result, bin_img, peak_img = process_image(img_file)
    
    st.success(f"✅ 計算完成")
    st.markdown(f"<h1 style='text-align: center; color: #E74C3C;'>共 {count} 顆</h1>", unsafe_allow_html=True)
    st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="偵測結果", use_container_width=True)
    
    with st.expander("👀 為什麼這次會準？ (除錯影像)"):
        col1, col2 = st.columns(2)
        with col1:
            st.image(bin_img, caption="1. 適應性黑白圖", use_container_width=True)
            st.write("你看，周圍的容器邊緣被強制塗黑了，藥丸也分得比較開。")
        with col2:
            st.image(peak_img, caption="2. 最終計算點", use_container_width=True)
            st.write("只計算最中心的白點。")
