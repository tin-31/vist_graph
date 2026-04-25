import streamlit as st
import gdown
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.neighbors import kneighbors_graph
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# ==========================================
# KHU VỰC 1: ĐỊNH NGHĨA KIẾN TRÚC MÔ HÌNH THẬT TỪ COLAB
# ==========================================
class ViST_GCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super().__init__()
        self.linear1 = nn.Linear(in_feats, hidden_feats)
        self.linear2 = nn.Linear(hidden_feats, out_feats)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index):
        # Khởi tạo Ma trận Kề Thưa (Sparse Adjacency Matrix)
        num_nodes = x.size(0)
        val = torch.ones(edge_index.size(1), device=x.device)
        adj = torch.sparse_coo_tensor(edge_index, val, (num_nodes, num_nodes))

        # Tính bậc (Degree) của từng node để chuẩn hóa
        deg = torch.sparse.sum(adj, dim=1).to_dense()
        deg_inv = 1.0 / deg
        deg_inv[deg_inv == float('inf')] = 0

        # Layer 1
        x = self.dropout(x)
        x = self.linear1(x)
        x = torch.sparse.mm(adj, x) * deg_inv.unsqueeze(1)
        x = F.elu(x)

        # Layer 2
        x = self.dropout(x)
        x = self.linear2(x)
        x = torch.sparse.mm(adj, x) * deg_inv.unsqueeze(1)

        return x

# ==========================================
# KHU VỰC 2: KHỞI TẠO HỆ THỐNG VÀ TẢI TRỌNG SỐ
# ==========================================
st.set_page_config(page_title="ViST-Graph Analysis", page_icon="🔬", layout="wide")
st.title("🔬 ViST-Graph: Dự đoán Hệ phiên mã không gian từ ảnh H&E")
st.markdown("**Sản phẩm dự thi Hội thi Tin học trẻ toàn quốc (Bảng D3)**")

@st.cache_resource 
def load_ai_models():
    # 1. Tải bộ trích xuất ResNet50
    resnet = models.resnet50(pretrained=True)
    resnet = nn.Sequential(*list(resnet.children())[:-1]) # Lấy vector 2048
    resnet.eval()

    # 2. Tải mô hình ViST-Graph từ Drive
    file_id = '1bSRncH0wWJki2b8ghBWIWpO0TilG_JGY' 
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'best_vist_model.pth'
    
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
        
    # Khởi tạo với thông số ĐÚNG của bạn: hidden_feats=256
    gcn_model = ViST_GCN(in_feats=2048, hidden_feats=256, out_feats=50)
    try:
        gcn_model.load_state_dict(torch.load(output, map_location=torch.device('cpu')))
    except Exception as e:
        st.error(f"Lỗi khớp trọng số: {e}")
    gcn_model.eval()
    
    return resnet, gcn_model

resnet, gcn_model = load_ai_models()

# ==========================================
# KHU VỰC 3: HÀM TIỀN XỬ LÝ ẢNH & ĐỒ THỊ (PIPELINE THỰC TẾ)
# ==========================================
def process_image_to_graph(image_pil, patch_size=224, grid_size=6):
    """Cắt ảnh thành grid 6x6, trích xuất ResNet50 và tạo Đồ thị KNN bằng Sklearn"""
    img_resized = image_pil.resize((patch_size * grid_size, patch_size * grid_size))
    
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    features = []
    coords = []
    
    # 1. Trích xuất đặc trưng 36 patches
    for i in range(grid_size):
        for j in range(grid_size):
            left, upper = j * patch_size, i * patch_size
            patch = img_resized.crop((left, upper, left + patch_size, upper + patch_size))
            patch_tensor = preprocess(patch).unsqueeze(0)
            
            with torch.no_grad():
                feat = resnet(patch_tensor).squeeze()
            
            features.append(feat)
            coords.append([i, j])
            
    x = torch.stack(features) # Shape: [36, 2048]
    coords_np = np.array(coords)
    
    # 2. Xây dựng đồ thị KNN (Giống y hệt code Colab của bạn)
    A = kneighbors_graph(coords_np, n_neighbors=6, mode='connectivity', include_self=True)
    adj_matrix = A.toarray()
    edges = np.column_stack(np.where(adj_matrix == 1))
    edge_index = torch.tensor(edges.T, dtype=torch.long)
    
    return x, edge_index, img_resized, grid_size

# ==========================================
# KHU VỰC 4: GIAO DIỆN UPLOAD & INFERENCE
# ==========================================
st.sidebar.header("Tải lên dữ liệu")
uploaded_file = st.sidebar.file_uploader("Chọn ảnh từ máy tính", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ảnh đầu vào (H&E)")
        st.image(image, use_column_width=True)
        
    if st.button("🚀 BẮT ĐẦU PHÂN TÍCH (INFERENCE THẬT)", use_container_width=True):
        with st.spinner('Đang chạy Pipeline: ResNet50 -> KNN Graph -> ViST-GCN...'):
            try:
                # 1. Chuyển ảnh thành Đồ thị
                x, edge_index, img_resized, grid_size = process_image_to_graph(image)
                
                # 2. Suy luận bằng GCN thật
                with torch.no_grad():
                    output_metagenes = gcn_model(x, edge_index) # Shape: [36, 50]
                
                # 3. Lấy PC0 (Siêu gene quan trọng nhất) làm Heatmap
                pc0_values = output_metagenes[:, 0].numpy()
                
                # Chuẩn hóa về [0, 255]
                pc0_norm = cv2.normalize(pc0_values, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                heatmap_grid = pc0_norm.reshape((grid_size, grid_size))
                
                # Phóng to heatmap khớp ảnh gốc
                heatmap_resized = cv2.resize(heatmap_grid, (img_resized.width, img_resized.height), interpolation=cv2.INTER_CUBIC)
                heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
                
                # Trộn Heatmap vào ảnh
                img_array = np.array(img_resized)
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                overlay = cv2.addWeighted(img_bgr, 0.5, heatmap_color, 0.5, 0)
                overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                
                with col2:
                    st.subheader("Bản đồ dự đoán (Metagene 0)")
                    st.image(overlay_rgb, use_column_width=True)
                    st.success("✅ Mô hình GCN đã chạy thành công trên dữ liệu thực!")
                
                # 4. Trực quan hóa giá trị trung bình của các Metagenes (Output Thật)
                st.markdown("---")
                st.subheader("🧬 Định lượng biểu hiện Siêu gene cốt lõi")
                
                mean_expressions = output_metagenes.mean(dim=0)[:5].numpy()
                genes = ['PC0 (Tumor Boundary)', 'Metagene 1', 'Metagene 2', 'Metagene 3', 'Metagene 4']
                
                fig, ax = plt.subplots(figsize=(10, 3))
                bars = ax.bar(genes, mean_expressions, color=['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f1c40f'])
                ax.set_ylabel('Cường độ biểu hiện thực (Real Output)')
                
                for bar in bars:
                    yval = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, yval + np.abs(yval)*0.05, f"{yval:.2f}", ha='center', va='bottom', fontweight='bold')
                
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Đã xảy ra lỗi hệ thống: {e}")
