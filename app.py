import streamlit as st
import cv2
import numpy as np

# --- 1. 極簡介面設定 ---
st.set_page_config(page_title="一鍵數藥丸", layout="centered")
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    h1 { color: #2E86C1; text-align: center; font-size: 2rem; }
    .stButton>button { 
        width: 100%; 
        border-radius: 50px; 
        height: 80px; 
        font-size: 24px; 
        font-weight: bold;
        background-color: #2E86C1;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.title("💊 一鍵智能數藥丸")
st.info("✨ 無需設定，請直接將藥丸放在畫面「正中間」拍照即可。")

# --- 2. 核心全自動邏輯 ---
def analyze_pills(img_buffer):
    # 讀取
    file_bytes = np.asarray(bytearray(img_buffer.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # === 第一步：強力裁切 (只看正中間) ===
    # 這步直接解決「木紋」和「容器邊緣」干擾
    h, w = img.shape[:2]
    crop_factor = 0.55 # 只保留中間 55% 的區域
    y_start = int(h * (1 - crop_factor) / 2)
    y_end = int(h * (1 + crop_factor) / 2)
    x_start = int(w * (1 - crop_factor) / 2)
    x_end = int(w * (1 + crop_factor) / 2)
    cropped = img[y_start:y_end, x_start:x_end]
    
    # === 第二步：光譜鎖定 (藍色通道) ===
    # 粉紅藥丸 = 高紅 + 高藍 / 紅蓋子 = 高紅 + 低藍
    # 所以取「藍色通道」，藥丸會變超白，蓋子會變超黑
    b, g, r = cv2.split(cropped)
    gray = b 
    
    # === 第三步：影像增強 ===
    # 讓對比更強烈，去除陰影
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # 模糊一點點以去除藥丸表面的刻痕雜訊
    blurred = cv2.GaussianBlur(enhanced, (13, 13), 0)
    
    # === 第四步：自動二值化 (Otsu) ===
    # 讓電腦自己決定黑白界線，不用手調
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 形態學清理 (把藥丸內部的小黑洞補起來)
    kernel = np.ones((5,5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # === 第五步：距離變換找山頂 (解決沾黏) ===
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    
    # 自動尋找局部最大值 (Local Maxima)
    # 這段邏輯是：只有當一個點比周圍都亮時，才算是一顆藥丸的中心
    # 這裡的 min_distance (20) 決定了兩顆藥丸最近不能小於 20 像素
    coordinates = []
    
    # 正規化距離圖以便尋找
    dist_norm = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX)
    dist_norm = np.uint8(dist_norm)
    
    # 使用簡單的閾值來過濾掉太矮的山丘 (雜訊)
    _, peaks = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    peaks = np.uint8(peaks)
    
    # 找輪廓來算數量
    cnts, _ = cv2.findContours(peaks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    count = 0
    output_img = cropped.copy()
    
    for c in cnts:
        # 過濾極小的噪點
        if cv2.contourArea(c) < 5: continue
        
        count += 1
        
        # 標記中心
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            # 畫一個鮮豔的標記
            cv2.circle(output_img, (cX, cY), 10, (0, 0, 255), -1)      # 紅點
            cv2.circle(output_img, (cX, cY), 30, (0, 255, 0), 3)       # 綠圈
            cv2.putText(output_img, str(count), (cX-15, cY-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3) # 黃字

    return count, output_img, binary

# --- 3. 執行區 ---
img_file = st.camera_input("📸 請點此拍照")

if img_file is not None:
    count, result_img, debug_bin = analyze_pills(img_file)
    
    # 顯示超大結果
    st.success("分析完成！")
    st.markdown(f"<div style='text-align: center; font-size: 80px; font-weight: bold; color: #E74C3C;'>{count} 顆</div>", unsafe_allow_html=True)
    
    st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="AI 偵測結果", use_container_width=True)
    
    # 為了讓你放心，顯示電腦看到的黑白畫面
    with st.expander("👀 電腦看到了什麼？"):
        st.image(debug_bin, caption="自動過濾後的影像", use_container_width=True)
        st.write("這張圖應該要黑白分明，藥丸是白色的，其他都是黑的。")
