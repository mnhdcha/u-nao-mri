import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input 

# ==========================================
# 1. TỪ ĐIỂN Y KHOA (KIẾN THỨC CHO AI)
# ==========================================
medical_info = {
    "Glioma": {
        "description": "U thần kinh đệm (Glioma) là loại u não phổ biến nhất bắt nguồn từ các tế bào thần kinh đệm. Khối u thường có tính chất xâm lấn mô não xung quanh.",
        "risk": "⚠️ Mức độ: Cần chú ý cao (Thường ác tính)",
        "recommendation": "Đề xuất: Cần chụp MRI có thuốc cản quang để xác định ranh giới u. Hội chẩn phẫu thuật hoặc xạ trị tùy vị trí."
    },
    "Meningioma": {
        "description": "U màng não (Meningioma) xuất phát từ màng nhện bao quanh não. Đa số là lành tính, phát triển chậm và có ranh giới rõ ràng.",
        "risk": "ℹ️ Mức độ: Thường lành tính",
        "recommendation": "Đề xuất: Theo dõi định kỳ nếu u nhỏ. Phẫu thuật cắt bỏ nếu u gây chèn ép thần kinh."
    },
    "Pituitary": {
        "description": "U tuyến yên (Pituitary Tumor) nằm ở hố yên (đáy sọ), có thể gây rối loạn nội tiết hoặc chèn ép giao thoa thị giác (gây mờ mắt).",
        "risk": "ℹ️ Mức độ: Thường lành tính nhưng ảnh hưởng chức năng",
        "recommendation": "Đề xuất: Xét nghiệm hormone đồ, kiểm tra thị trường mắt. Điều trị nội khoa hoặc phẫu thuật qua xoang bướm."
    },
    "No Tumor": {
        "description": "Không phát hiện khối u bất thường rõ rệt trên hình ảnh MRI này.",
        "risk": "✅ Mức độ: Bình thường",
        "recommendation": "Đề xuất: Duy trì lối sống lành mạnh. Nếu vẫn có triệu chứng đau đầu dai dẳng, hãy khám chuyên khoa thần kinh để loại trừ nguyên nhân khác."
    }
}

# ==========================================
# 2. CÁC HÀM XỬ LÝ ẢNH (CORE)
# ==========================================

# Hàm 1: Tự động cắt bỏ viền đen thừa (Crop)
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

# --- HÀM GRAD-CAM (ĐÃ BỎ CHE VIỀN ĐỂ BẮT U SÁT SỌ) ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # 1. Tạo model phụ
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # 2. Tính Gradient
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if isinstance(preds, list): preds = preds[0]
        preds = tf.convert_to_tensor(preds)
        
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # 3. Tạo Heatmap
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Chuẩn hóa về 0-1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    # --- ĐÃ XÓA ĐOẠN MASKING (CHE VIỀN) Ở ĐÂY ---
    # Để đảm bảo khối u sát sọ không bị mất
    
    return heatmap.numpy()

# Hàm 3: Vẽ khung chữ nhật (Bounding Box) từ Heatmap
def draw_bbox_from_heatmap(image, heatmap, threshold=0.55):
    # Nhị phân hóa Heatmap: Chỉ lấy vùng "nóng" nhất (trên 45%)
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    
    _, thresh = cv2.threshold(heatmap_uint8, int(255 * threshold), 255, cv2.THRESH_BINARY)
    
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    output_image = image.copy()
    
    if len(cnts) > 0:
        # Tìm vùng lớn nhất
        c = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        
        # Vẽ khung màu xanh lá (Green)
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(output_image, "Tumor Region", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    return output_image

# ==========================================
# 3. GIAO DIỆN WEB (STREAMLIT APP)
# ==========================================
st.set_page_config(page_title="Chẩn Đoán U Não AI Pro", layout="wide")
st.title("🧠 Hệ Thống Phân Tích MRI Não (EfficientNet + XAI)")
st.write("Ứng dụng hỗ trợ chẩn đoán và định vị khối u não sử dụng Deep Learning.")

@st.cache_resource
def load_model():
    # Load model đã train (File phải tên là model.h5)
    model = tf.keras.models.load_model('model.h5')
    return model

try:
    model = load_model()
    st.toast("Đã tải mô hình thành công!", icon="✅")
except Exception as e:
    st.error(f"Lỗi tải mô hình: {e}. Hãy kiểm tra lại file model.h5 trên GitHub.")

# Upload ảnh
uploaded_file = st.file_uploader("Tải ảnh MRI lên để phân tích...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Đọc ảnh
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Chia cột hiển thị
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("1. Ảnh Gốc")
        st.image(image, use_column_width=True)

    # --- QUY TRÌNH XỬ LÝ ẢNH ---
    # 1. Crop bỏ viền
    cropped_image = crop_brain_contour(image)
    
    # 2. Resize chuẩn EfficientNet
    IMG_SIZE = 224 
    resized_image = cv2.resize(cropped_image, (IMG_SIZE, IMG_SIZE))
    
    # 3. Preprocess Input
    input_arr = np.array(resized_image, dtype=np.float32)
    processed_image = preprocess_input(input_arr) 
    input_data = np.expand_dims(processed_image, axis=0)

    with col2:
        st.warning(f"2. Input Model ({IMG_SIZE}x{IMG_SIZE})")
        st.image(resized_image, use_column_width=True)
        st.caption("Ảnh đã qua xử lý cắt viền và chuẩn hóa.")

    # Nút bấm dự đoán
    if st.button("Chạy Chẩn Đoán & Định Vị"):
        try:
            # --- DỰ ĐOÁN ---
            prediction = model.predict(input_data)
            labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
            
            pred_index = np.argmax(prediction)
            predicted_class = labels[pred_index]
            confidence = np.max(prediction) * 100
            
            st.divider()
            
            # --- TÍNH TOÁN GRAD-CAM & VẼ KHUNG ---
            # Lớp cuối cùng của EfficientNetB0 là 'top_activation'
            last_conv_layer_name = "top_activation"
            
            # Lấy Heatmap gốc (0-1)
            raw_heatmap = make_gradcam_heatmap(input_data, model, last_conv_layer_name)
            
            # Resize heatmap bằng kích thước ảnh crop
            heatmap_resized = cv2.resize(raw_heatmap, (cropped_image.shape[1], cropped_image.shape[0]))
            
            # Vẽ khung (Bounding Box) lên ảnh
# Vẽ khung (Bounding Box) lên ảnh
            # Tăng ngưỡng lên 0.5 để chỉ khoanh vùng thật sự đậm
            bbox_img = draw_bbox_from_heatmap(cropped_image, raw_heatmap, threshold=0.5)            
            with col3:
                st.success("3. Kết quả & Định vị")
                st.image(bbox_img, use_column_width=True)
                st.caption(f"Định vị vùng nghi ngờ ({predicted_class})")

            # --- HIỂN THỊ BÁO CÁO CHI TIẾT ---
            info = medical_info[predicted_class]
            
            st.write("---")
            st.subheader(f"📋 Kết quả chẩn đoán: {predicted_class}")
            
            res_c1, res_c2 = st.columns([1, 2])
            
            with res_c1:
                # Hiển thị độ tin cậy
                if confidence > 90:
                    st.success(f"Độ tin cậy: **{confidence:.2f}%**")
                elif confidence > 70:
                    st.warning(f"Độ tin cậy: **{confidence:.2f}%**")
                else:
                    st.error(f"Độ tin cậy: **{confidence:.2f}%**")
                
                st.metric(label="Đánh giá rủi ro", value=predicted_class, delta=info["risk"])
            
            with res_c2:
                st.info(f"**Mô tả bệnh học:** {info['description']}")
                st.warning(f"**Khuyến nghị lâm sàng:** {info['recommendation']}")
                
            st.caption("⚠️ Lưu ý: Hệ thống AI chỉ mang tính chất hỗ trợ sàng lọc. Vui lòng tham khảo ý kiến bác sĩ chuyên khoa để có chẩn đoán chính xác nhất.")
            
        except Exception as e:
            st.error(f"Đã xảy ra lỗi trong quá trình xử lý: {e}")
