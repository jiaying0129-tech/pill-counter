import streamlit as st
import cv2
import numpy as np

# --- 1. 介面設定 ---
st.set_page_config(page_title="通用智慧數藥丸", layout="centered")
st.markdown("""
    <style>
    .main { background-color: #0E1117; color: white; }
    h1 { color: #FFD700; text-align: center; }
    .stButton>button { 
        width: 100%; border-radius: 12px; height: 60px; 
        font-size: 20px; font-weight: bold;
        background-color: #FFD700; color: black; border: none;
    }
    </style>
""", unsafe_allow_html=True)

st.title("💊 通用智慧數藥丸")
st.info("🤖 此版本使用「群體分析演算法」。不限顏色形狀，會自動過濾掉不合群的雜訊（如瓶蓋反光）。")

# --- 2. 參數 (僅保留視野微調) ---
with st.expander("📐 如果抓到背景，請調整視野範圍"):
    scope_size = st.slider("視野範圍 (0.5 = 只看畫面中間 50%)", 0.3, 0.9, 0.6)

# --- 3. 核心邏輯：通用適應性演算法 ---
def smart_analysis(img_buffer, scope):
    # 1. 讀取
    file_bytes = np.asarray(bytearray(img_buffer.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    h, w = img.shape[:2]
    
    # 2. 視野裁切 (聚焦中心)
    # 我們不只是塗黑，而是直接切出來運算，減少運算量
    crop_h, crop_w = int(h*scope), int(w*scope)
    start_y, start_x = (h - crop_h)//2, (w - crop_w)//2
    cropped = img[start_y:start_y+crop_h, start_x:start_x+crop_w]
    
    # 3. 轉灰階 + 強力模糊 (去除紋路與刻痕)
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    # 使用雙邊濾波 (Bilateral Filter) 保留邊緣但模糊表面 (去除 R 字)
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    # 再加高斯模糊確保光滑
    blurred = cv2.GaussianBlur(blurred, (11, 11), 0)
    
    # 4. 適應性閾值 (Adaptive Threshold) - 通用關鍵！
    # 不管藥丸是什麼顏色，只要跟背景有亮度差，這個方法都能抓到
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 25, 3)
    
    # 5. 形態學操作 (修補與斷開)
    kernel = np.ones((3,3), np.uint8)
    # 開運算：去除小白點雜訊
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    # 閉運算：把藥丸內部的空洞填滿
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # 6. 距離變換 + 分水嶺 (分離沾黏)
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    # 這裡用較低的閾值 (0.4) 來確保不同形狀的藥丸都能找到核心
    _, peaks = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    peaks = np.uint8(peaks)
    
    cnts, _ = cv2.findContours(peaks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # === 7. 智慧過濾系統 (Smart Filter) ===
    # 這是踢掉第 5 點(瓶蓋反光)的關鍵
    
    final_candidates = []
    
    # 7a. 收集所有候選點的資訊
    candidates_data = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 10: continue # 過濾極小噪點
        
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            candidates_data.append({'cnt': c, 'area': area, 'center': (cX, cY)})
    
    # 7b. 計算群體中位數 (大家通常多大？)
    if candidates_data:
        areas = [d['area'] for d in candidates_data]
        median_area = np.median(areas)
        
        # 計算群體重心 (大家聚在哪裡？)
        centers = np.array([d['center'] for d in candidates_data])
        group_center = np.mean(centers, axis=0)
        
        for item in candidates_data:
            # 規則 1: 大小過濾
            # 如果這個點比「平均大小」小太多 (例如小於 1/5)，那就是雜訊 (瓶蓋反光通常比較小)
            if item['area'] < median_area * 0.2:
                continue
            
            # 規則 2: 距離過濾
            # 計算這個點離「大家」有多遠
            dist_from_group = np.linalg.norm(np.array(item['center']) - group_center)
            
            # 如果這個點離群體的中心太遠 (大於畫面寬度的 40%)，判定為邊緣雜訊
            if dist_from_group > crop_w * 0.4:
                continue
                
            final_candidates.append(item)
            
    # 8. 繪圖
    count = len(final_candidates)
    output_img = cropped.copy()
    
    for i, item in enumerate(final_candidates):
        cX, cY = item['center']
        cv2.circle(output_img, (cX, cY), 10, (0, 0, 255), -1) # 紅點
        cv2.circle(output_img, (cX, cY), 25, (0, 255, 0), 2) # 綠圈
        cv2.putText(output_img, str(i+1), (cX-10, cY-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    return count, output_img, binary

# --- 4. 執行區 ---
img_file = st.camera_input("📸 請拍照")

if img_file is not None:
    count, result_img, debug_bin = smart_analysis(img_file, scope_size)
    
    st.success("智慧分析完成！")
    st.markdown(f"<div style='text-align: center; font-size: 80px; font-weight: bold; color: #FFD700;'>{count} 顆</div>", unsafe_allow_html=True)
    
    st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="偵測結果 (已過濾離群雜訊)", use_container_width=True)
    
    with st.expander("🧠 AI 是如何思考的？ (除錯)"):
        st.write("1. **適應性視覺**：不分顏色，只抓結構。")
        st.image(debug_bin, caption="電腦看到的結構圖", use_container_width=True)
        st.write("2. **群體過濾**：程式計算了所有點的平均大小和位置，把角落那個長得不一樣、離大家太遠的雜訊踢掉了。")
