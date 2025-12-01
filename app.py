import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input 

# ==========================================
# 1. TỪ ĐIỂN Y KHOA
# ==========================================
medical_info = {
    "Glioma": {
        "description": "U thần kinh đệm (Glioma) là loại u não phổ biến nhất bắt nguồn từ các tế bào thần kinh đệm. Khối u thường có tính chất xâm lấn.",
        "risk": "⚠️ Mức độ: Cần chú ý cao (Thường ác tính)",
        "recommendation": "Đề xuất: Cần chụp MRI có thuốc cản quang để xác định ranh giới u. Hội chẩn phẫu thuật hoặc xạ trị tùy vị trí."
    },
    "Meningioma": {
        "description": "U màng não (Meningioma) xuất phát từ màng nhện bao quanh não. Đa số là lành tính và phát triển chậm.",
        "risk": "ℹ️ Mức độ: Thường lành tính",
        "recommendation": "Đề xuất: Theo dõi định kỳ nếu u nhỏ. Phẫu thuật cắt bỏ nếu u gây chèn ép thần kinh."
    },
    "Pituitary": {
        "description": "U tuyến yên (Pituitary Tumor) nằm ở hố yên, có thể gây rối loạn nội tiết hoặc chèn ép giao thoa thị giác.",
        "risk": "ℹ️ Mức độ: Thường lành tính nhưng ảnh hưởng chức năng",
        "recommendation": "Đề xuất: Xét nghiệm hormone, kiểm tra thị trường mắt. Điều trị nội khoa hoặc phẫu thuật qua xoang bướm."
    },
    "No Tumor": {
        "description": "Không phát hiện khối u bất thường rõ rệt trên hình ảnh MRI này.",
        "risk": "✅ Mức độ: Bình thường",
        "recommendation": "Đề xuất: Duy trì lối sống lành mạnh. Nếu vẫn có triệu chứng đau đầu, hãy khám chuyên khoa thần kinh."
    }
}

# ==========================================
# 2. CÁC HÀM XỬ LÝ ẢNH
# ==========================================

# Hàm tự động cắt viền đen (Crop)
def crop_brain_contour(image, plot=False):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        new_image = image[y:y+h, x:x+w]
        return new_image
    return image

# Hàm tạo Heatmap Grad-CAM (Vùng đỏ)
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if isinstance(preds, list): preds = preds[0]
        preds = tf.convert_to_tensor(preds)
        
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    # LƯU Ý: Không dùng che viền để đảm bảo bắt được u sát sọ
    return heatmap.numpy()

# ==========================================
# 3. GIAO DIỆN WEB
# ==========================================
st.set_page_config(page_title="Chẩn Đoán U Não AI", layout="wide")
st.title("🧠 Hệ Thống Phân Tích MRI Não (Grad-CAM)")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model.h5')
    return model

try:
    model = load_model()
    st.toast("Đã tải mô hình thành công!", icon="✅")
except Exception as e:
    st.error(f"Lỗi tải mô hình: {e}")

uploaded_file = st.file_uploader("Chọn ảnh MRI...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Đọc ảnh
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("1. Ảnh Gốc")
        st.image(image, use_column_width=True)

    # Xử lý ảnh
    cropped_image = crop_brain_contour(image)
    IMG_SIZE = 224 
    resized_image = cv2.resize(cropped_image, (IMG_SIZE, IMG_SIZE))
    input_arr = np.array(resized_image, dtype=np.float32)
    processed_image = preprocess_input(input_arr) 
    input_data = np.expand_dims(processed_image, axis=0)

    with col2:
        st.warning(f"2. Input Model ({IMG_SIZE}x{IMG_SIZE})")
        st.image(resized_image, use_column_width=True)

    if st.button("Chạy Chẩn Đoán"):
        try:
            # Dự đoán
            prediction = model.predict(input_data)
            labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
            
            pred_index = np.argmax(prediction)
            predicted_class = labels[pred_index]
            confidence = np.max(prediction) * 100
            
            st.divider()
            
            # --- TẠO GRAD-CAM (VÙNG ĐỎ) ---
            last_conv_layer_name = "block7a_project_conv"
            heatmap = make_gradcam_heatmap(input_data, model, last_conv_layer_name)
            
            # Xử lý hiển thị màu
            heatmap_resized = cv2.resize(heatmap, (cropped_image.shape[1], cropped_image.shape[0]))
            heatmap_uint8 = np.uint8(255 * heatmap_resized)
            heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            
            # Chồng ảnh (Superimpose)
            superimposed_img = cv2.addWeighted(cropped_image, 0.6, heatmap_colored, 0.4, 0)
            
            with col3:
                st.success("3. Giải thích (Vùng nhiệt)")
                st.image(superimposed_img, use_column_width=True)
                st.caption(f"Vùng màu ĐỎ là nơi AI phát hiện đặc điểm của {predicted_class}")
            
            # Hiển thị thông tin y khoa
            info = medical_info[predicted_class]
            st.write("---")
            st.subheader(f"📋 Hồ sơ: {predicted_class}")
            
            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric(label="Rủi ro", value=predicted_class, delta=info["risk"])
            with c2:
                st.info(f"**Mô tả:** {info['description']}")
                st.warning(f"**Khuyến nghị:** {info['recommendation']}")
                
        except Exception as e:
            st.error(f"Lỗi: {e}")
