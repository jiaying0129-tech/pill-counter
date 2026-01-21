import streamlit as st
import cv2
import numpy as np

# --- 1. 介面設定 ---
st.set_page_config(page_title="幾何鎖定數藥丸", layout="centered")
st.markdown("""
    <style>
    .main { background-color: #262730; color: white; }
    h1 { color: #00e6e6; text-align: center; }
    .stButton>button { 
        width: 100%; border-radius: 50px; height: 72px; 
        font-size: 28px; font-weight: bold;
        background-color: #00e6e6; color: black; border: none;
    }
    </style>
""", unsafe_allow_html=True)

st.title("💊 幾何鎖定數藥丸")
st.info("🔹 這個版本使用「幾何圓形偵測」，專門過濾木紋背景與藥丸上的刻痕。")

# --- 2. 只有一個必要的滑桿 (視野控制) ---
# 為了適應你手機拿遠拿近，這是唯一保留的調整項
with st.expander("🔎 如果抓到背景，請調整這裡 (視野範圍)"):
    scope_size = st.slider("偵測範圍 (只看中間)", 0.3, 0.9, 0.55, help="數值越小，只看越中心，能避開更多背景")

# --- 3. 核心邏輯：霍夫圓形變換 + 強力聚光燈 ---
def geometry_analysis(img_buffer, scope):
    # 讀取
    file_bytes = np.asarray(bytearray(img_buffer.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    h, w = img.shape[:2]
    
    # === 第一步：建立聚光燈遮罩 (Spotlight) ===
    # 畫一個黑色的遮罩，只留中間
    mask = np.zeros((h, w), dtype=np.uint8)
    center_x, center_y = w // 2, h // 2
    radius = int(min(h, w) * scope / 2)
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    
    # 套用遮罩
    img_masked = cv2.bitwise_and(img, img, mask=mask)
    
    # === 第二步：影像前處理 (關鍵！) ===
    # 轉灰階
    gray = cv2.cvtColor(img_masked, cv2.COLOR_BGR2GRAY)
    
    # [超級關鍵] 強力高斯模糊
    # 這步會把藥丸上的 "R" 字刻痕模糊掉，讓整顆藥丸看起來像一個光滑的饅頭
    # 這樣電腦就不會把 "R" 的陰影誤判成另一顆藥丸
    blurred = cv2.GaussianBlur(gray, (15, 15), 2)
    
    # === 第三步：霍夫圓形偵測 (Hough Circles) ===
    # 這是工業界專門用來找圓形物體的演算法
    # dp=1.2: 解析度
    # minDist=40: 兩顆藥丸圓心的最小距離 (避免重複算)
    # param1=50: 邊緣偵測閾值
    # param2=30: 圓形判定閾值 (越小越容易判定是圓)
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2, 
        minDist=radius/4, # 動態調整最小距離
        param1=50, 
        param2=25, 
        minRadius=int(radius/10), # 限制藥丸最小多大
        maxRadius=int(radius/3)   # 限制藥丸最大多大
    )
    
    count = 0
    output_img = img.copy()
    
    # 畫出偵測範圍 (黃色圈) 讓你知道電腦在看哪裡
    cv2.circle(output_img, (center_x, center_y), radius, (0, 255, 255), 3)
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # 雙重確認：只有在聚光燈範圍內的圓才算
            dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            if dist_from_center > radius:
                continue
                
            count += 1
            # 畫出偵測到的藥丸 (鮮豔綠色)
            cv2.circle(output_img, (x, y), r, (0, 255, 0), 4)
            cv2.circle(output_img, (x, y), 5, (0, 0, 255), -1) # 圓心
            cv2.putText(output_img, str(count), (x-10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 3)

    return count, output_img, blurred

# --- 4. 執行區 ---
img_file = st.camera_input("📸 請拍照")

if img_file is not None:
    count, result_img, debug_blur = geometry_analysis(img_file, scope_size)
    
    # 顯示結果
    st.success("分析完成！")
    st.markdown(f"<div style='text-align: center; font-size: 80px; font-weight: bold; color: #00e6e6;'>{count} 顆</div>", unsafe_allow_html=True)
    
    st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="偵測結果 (黃圈是視野範圍)", use_container_width=True)
    
    with st.expander("👀 檢查電腦是否「眼花」？ (除錯影像)"):
        st.image(debug_blur, caption="電腦看到的模糊影像", use_container_width=True)
        st.caption("藥丸應該要看起來像模糊的光滑圓球，上面的 R 字應該要看不見。")
