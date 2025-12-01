import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input # Dùng hàm chuẩn
# --- TỪ ĐIỂN Y KHOA (Kiến thức cho AI) ---
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
        "description": "U tuyến yên (Pituitary Tumor) nằm ở hố yên, có thể gây rối loạn nội tiết hoặc chèn ép giao thoa thị giác (mờ mắt).",
        "risk": "ℹ️ Mức độ: Thường lành tính nhưng ảnh hưởng chức năng",
        "recommendation": "Đề xuất: Xét nghiệm hormone, kiểm tra thị trường mắt. Điều trị nội khoa hoặc phẫu thuật qua xoang bướm."
    },
    "No Tumor": {
        "description": "Không phát hiện khối u bất thường rõ rệt trên hình ảnh MRI này.",
        "risk": "✅ Mức độ: Bình thường",
        "recommendation": "Đề xuất: Duy trì lối sống lành mạnh. Nếu vẫn có triệu chứng đau đầu, hãy khám chuyên khoa thần kinh để loại trừ nguyên nhân khác."
    }
}
# 1. Hàm tự động cắt viền đen (Giữ nguyên vì đã tốt)
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

# --- HÀM GRAD-CAM "BỌC GIÁP" (Fix mọi lỗi version) ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # 1. Tạo model phụ
    # Sử dụng model.inputs (số nhiều) để an toàn nhất
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # 2. Tính toán Gradient
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        
        # --- ĐOẠN CODE TRỊ LỖI QUAN TRỌNG NHẤT ---
        # Kiểm tra: Nếu preds bị trả về dạng List (do khác version), lấy phần tử đầu tiên
        if isinstance(preds, list):
            preds = preds[0]
        
        # Đảm bảo nó là Tensor để tính toán được
        preds = tf.convert_to_tensor(preds)
        # -----------------------------------------

        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # 3. Xử lý Heatmap
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# 3. Giao diện Web
st.set_page_config(page_title="Chẩn Đoán U Não AI Pro", layout="wide")
st.title("🧠 Hệ Thống Phân Tích MRI Não (EfficientNetB0)")

@st.cache_resource
def load_model():
    # Load model đã train
    model = tf.keras.models.load_model('model.h5')
    return model

try:
    model = load_model()
    st.success("Đã tải mô hình (Phiên bản Fine-Tuning 224x224) thành công!")
except Exception as e:
    st.error(f"Lỗi tải mô hình: {e}")

uploaded_file = st.file_uploader("Chọn ảnh MRI...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("1. Ảnh Gốc")
        st.image(image, use_column_width=True)

    # --- XỬ LÝ ẢNH MỚI (QUAN TRỌNG) ---
    # 1. Crop
    cropped_image = crop_brain_contour(image)
    
    # 2. Resize lên 224 (Kích thước mới)
    IMG_SIZE = 224 
    resized_image = cv2.resize(cropped_image, (IMG_SIZE, IMG_SIZE))
    
    # 3. Preprocess đúng chuẩn EfficientNet (Thay vì chia 255 thủ công)
    # Vì lúc train ta dùng preprocess_input, giờ ta cũng phải dùng y hệt
    input_arr = np.array(resized_image, dtype=np.float32)
    processed_image = preprocess_input(input_arr) 
    input_data = np.expand_dims(processed_image, axis=0)

    with col2:
        st.warning(f"2. Input Model ({IMG_SIZE}x{IMG_SIZE})")
        st.image(resized_image, use_column_width=True) # Hiển thị ảnh sau crop

    if st.button("Chạy Chẩn Đoán"):
        prediction = model.predict(input_data)
        labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
        
        # Lấy kết quả cao nhất
        pred_index = np.argmax(prediction)
        predicted_class = labels[pred_index]
        confidence = np.max(prediction) * 100
        
        st.divider()
        st.subheader(f"Kết quả: {predicted_class}")
        
        # Logic hiển thị màu sắc độ tin cậy
        if confidence > 90:
            st.success(f"Độ tin cậy cao: {confidence:.2f}%")
        elif confidence > 70:
            st.warning(f"Độ tin cậy trung bình: {confidence:.2f}%")
        else:
            st.error(f"Độ tin cậy thấp ({confidence:.2f}%). Cần bác sĩ kiểm tra lại.")
        
        # --- GRAD-CAM ---
        try:
# Chỉ định trực tiếp tên lớp cuối cùng của EfficientNetB0
            # Dựa trên cấu trúc model bạn đã train, tên của nó là 'top_activation'
            last_conv_layer_name = "top_activation"
            
            heatmap = make_gradcam_heatmap(input_data, model, last_conv_layer_name)
            
            # Resize heatmap về kích thước ảnh Crop để chồng lên
            heatmap = cv2.resize(heatmap, (cropped_image.shape[1], cropped_image.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            superimposed_img = cv2.addWeighted(cropped_image, 0.6, heatmap, 0.4, 0)
            
            with col3:
                st.success("3. Giải thích (Grad-CAM)")
                st.image(superimposed_img, use_column_width=True)
                st.caption(f"AI đang nhìn vào vùng màu đỏ để kết luận là {predicted_class}")
                
        except Exception as e:
            st.error(f"Không thể tạo Grad-CAM: {e}")
# --- HIỂN THỊ HỒ SƠ BỆNH ÁN (Thêm đoạn này) ---
        info = medical_info[predicted_class]
        
        st.write("---") # Đường kẻ ngang
        st.subheader("📋 Hồ sơ chẩn đoán lâm sàng")
        
        # Chia cột để hiển thị đẹp hơn
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.metric(label="Đánh giá rủi ro", value=predicted_class, delta=info["risk"])
        
        with c2:
            st.info(f"**Mô tả:** {info['description']}")
            st.warning(f"**Khuyến nghị bác sĩ:** {info['recommendation']}")
            
        st.caption("⚠️ Lưu ý: Kết quả này chỉ mang tính chất tham khảo hỗ trợ, không thay thế chẩn đoán của bác sĩ chuyên khoa.")
