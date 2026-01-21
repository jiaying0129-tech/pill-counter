import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="全能藥丸計數器", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    h1 { color: #2e86c1; text-align: center; }
    .stButton>button { width: 100%; border-radius: 20px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title("💊 全能藥丸計數器")
st.info("💡 這個版本透過「形狀特徵」來辨識，可同時計算圓形、膠囊與不同大小的藥丸。")

# --- 側邊欄：強大的過濾器 ---
with st.expander("🎛️ 參數控制台 (調整核心)", expanded=True):
    st.write("### 1. 影像優化 (去除木紋)")
    # 雙邊濾波是去除木紋的神器，能保邊去噪
    blur_strength = st.slider("磨皮強度 (去除紋路)", 1, 50, 25, help="數值越高，木紋越不明顯，但藥丸邊緣需保持清晰")
    contrast = st.slider("對比度增強", 1.0, 3.0, 1.5)
    
    st.write("### 2. 邊緣偵測")
    canny_min = st.slider("邊緣敏銳度 (Min)", 10, 200, 50)
    canny_max = st.slider("邊緣敏銳度 (Max)", 50, 300, 150)
    
    st.write("### 3. 形狀過濾器 (關鍵！)")
    col1, col2 = st.columns(2)
    with col1:
        min_area = st.number_input("最小面積", value=100)
        max_area = st.number_input("最大面積", value=5000)
    with col2:
        # 圓度：1.0 是正圓，0.5 是膠囊，0.1 是長條
        min_circularity = st.slider("形狀限制 (圓度)", 0.0, 1.0, 0.4, help="1.0=只要正圓, 0.4=包含橢圓/膠囊")

# --- 核心邏輯 ---
def process_image(img_buffer):
    # 1. 讀取圖片
    file_bytes = np.asarray(bytearray(img_buffer.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # 2. 影像增強 (對比度)
    # 讓藥丸跟背景分離更明顯
    img_float = img.astype(float) * contrast
    img_float[img_float > 255] = 255
    img = img_float.astype(np.uint8)
    
    # 3. 強力去噪 (雙邊濾波 Bilateral Filter)
    # 這是對付木紋的關鍵，它會模糊紋理但保留藥丸邊緣
    filtered = cv2.bilateralFilter(img, 9, 75, 75)
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    
    # 也可以疊加高斯模糊
    if blur_strength > 0:
        # 確保核大小是奇數
        k_size = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
        gray = cv2.GaussianBlur(gray, (k_size, k_size), 0)
    
    # 4. 邊緣偵測 (Canny)
    edged = cv2.Canny(gray, canny_min, canny_max)
    
    # 5. 形態學閉運算 (把邊緣斷掉的地方接起來)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    
    # 6. 找輪廓
    cnts, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    count = 0
    output_img = img.copy()
    valid_contours = []
    
    for c in cnts:
        area = cv2.contourArea(c)
        
        # 過濾 1: 面積
        if area < min_area or area > max_area:
            continue
            
        # 過濾 2: 圓度 (Circularity)
        # 公式: 4 * Pi * Area / (Perimeter^2)
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0: continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        if circularity < min_circularity:
            continue
            
        # 通過所有測試！
        count += 1
        valid_contours.append(c)
        
        # 畫圖
        cv2.drawContours(output_img, [c], -1, (0, 255, 0), 3)
        
        # 標記數字
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(output_img, str(count), (cX - 10, cY - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return count, output_img, gray, closed

# --- 介面 ---
img_file = st.camera_input("📸 請拍照")

if img_file is not None:
    count, result_img, debug_gray, debug_edge = process_image(img_file)
    
    st.success(f"✅ 共發現 {count} 顆")
    st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="偵測結果", use_container_width=True)
    
    st.markdown("---")
    st.write("### 🔍 除錯視窗 (如果沒抓到，請看這裡)")
    col1, col2 = st.columns(2)
    with col1:
        st.image(debug_gray, caption="1. 電腦看到的亮度 (木紋是否消失?)", use_container_width=True)
    with col2:
        st.image(debug_edge, caption="2. 電腦抓到的邊緣 (線條是否完整?)", use_container_width=True)
        
    st.info("""
    **🔧 調整攻略：**
    1. **木紋太明顯？** 👉 調高「磨皮強度」。
    2. **邊緣斷斷續續？** 👉 降低「邊緣敏銳度 (Min)」。
    3. **膠囊沒抓到？** 👉 降低「形狀限制 (圓度)」到 0.4 或更低。
    4. **抓到太多背景雜點？** 👉 調高「最小面積」。
    """)
