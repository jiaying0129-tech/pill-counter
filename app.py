import streamlit as st
import cv2
import numpy as np

# --- 1. 介面設定 ---
st.set_page_config(page_title="AI 聚光燈數藥丸", layout="centered")
st.markdown("""
    <style>
    .main { background-color: #0E1117; color: white; }
    h1 { color: #00FF00; text-align: center; }
    .stButton>button { 
        width: 100%; border-radius: 50px; height: 70px; 
        font-size: 24px; font-weight: bold;
        background-color: #00CC00; color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.title("💊 AI 聚光燈數藥丸")
st.warning("🎯 請將藥丸放在畫面**正中間**。程式會自動把周圍塗黑，無視背景。")

# --- 2. 核心邏輯：聚光燈演算法 ---
def spotlight_analysis(img_buffer):
    # 讀取
    file_bytes = np.asarray(bytearray(img_buffer.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    h, w = img.shape[:2]
    
    # === 第一步：建立強力聚光燈 (Spotlight Mask) ===
    # 這是解決你問題的關鍵！
    # 我們建立一個全黑的遮罩，只在正中間挖一個洞
    mask = np.zeros((h, w), dtype=np.uint8)
    center_x, center_y = w // 2, h // 2
    
    # 設定半徑為短邊的 35% (非常積極的過濾，強制只看中間)
    radius = int(min(h, w) * 0.35)
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    
    # 套用遮罩：遮罩外的東西全部變全黑 (R=0, G=0, B=0)
    img_spotlight = cv2.bitwise_and(img, img, mask=mask)
    
    # === 第二步：綠色通道分析 (Green Channel) ===
    # 對於粉紅藥丸與木紋背景，綠色通道通常是最乾淨的
    b, g, r = cv2.split(img_spotlight)
    gray = g
    
    # === 第三步：對比度極限增強 ===
    # 讓藥丸亮到爆，背景暗下去
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # === 第四步：閾值處理 (Threshold) ===
    # 這裡我們用一個技巧：只對「有亮光的地方」做 Otsu
    # 這樣黑色的背景就不會干擾計算
    # 先做高斯模糊去噪
    blurred = cv2.GaussianBlur(enhanced, (15, 15), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 再次強制套用圓形遮罩 (確保邊緣沒有殘留白邊)
    binary = cv2.bitwise_and(binary, binary, mask=mask)
    
    # === 第五步：分離黏在一起的藥丸 ===
    # 距離變換
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    
    # 找峰值 (Peaks)
    # 這裡設定 0.5 (50% 亮度)，這是一個很安全的數值，能分開大部分圓形藥丸
    _, peaks = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    peaks = np.uint8(peaks)
    
    # === 第六步：計數與雙重過濾 ===
    cnts, _ = cv2.findContours(peaks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    count = 0
    output_img = img.copy() # 畫在原圖上
    
    # 畫出聚光燈範圍給使用者看
    cv2.circle(output_img, (center_x, center_y), radius, (0, 255, 255), 2)
    
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 10: continue # 過濾極小噪點
        
        # 計算中心點
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            # 【終極過濾】：如果偵測點離圓心太遠，一定是誤判 (比如蓋子邊緣)
            dist_from_center = np.sqrt((cX - center_x)**2 + (cY - center_y)**2)
            if dist_from_center > radius * 0.9:
                continue
            
            count += 1
            
            # 畫標記
            cv2.circle(output_img, (cX, cY), 8, (0, 0, 255), -1) # 紅點
            cv2.circle(output_img, (cX, cY), 20, (0, 255, 0), 2) # 綠圈
            cv2.putText(output_img, str(count), (cX-10, cY-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    return count, output_img, binary, img_spotlight

# --- 3. 執行區 ---
img_file = st.camera_input("📸 請點此拍照")

if img_file is not None:
    count, result_img, debug_bin, debug_spot = spotlight_analysis(img_file)
    
    st.success("分析完成！")
    st.markdown(f"<div style='text-align: center; font-size: 80px; font-weight: bold; color: #00FF00;'>{count} 顆</div>", unsafe_allow_html=True)
    
    st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="AI 偵測結果 (黃圈內為偵測範圍)", use_container_width=True)
    
    with st.expander("👀 為什麼這次雜訊不見了？"):
        col1, col2 = st.columns(2)
        with col1:
            st.image(debug_spot, caption="1. 聚光燈效果", use_container_width=True)
            st.write("程式強制把周圍塗黑，木紋直接消失。")
        with col2:
            st.image(debug_bin, caption="2. 最終判讀", use_container_width=True)
            st.write("乾淨的黑白影像，只剩中間的藥丸。")
