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
import joblib
from PIL import Image
import matplotlib.pyplot as plt

# ==========================================
# KHU VỰC 1: ĐỊNH NGHĨA KIẾN TRÚC MÔ HÌNH GCN (CHUẨN COLAB)
# ==========================================
class ViST_GCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super().__init__()
        self.linear1 = nn.Linear(in_feats, hidden_feats)
        self.linear2 = nn.Linear(hidden_feats, out_feats)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        val = torch.ones(edge_index.size(1), device=x.device)
        adj = torch.sparse_coo_tensor(edge_index, val, (num_nodes, num_nodes))
        deg = torch.sparse.sum(adj, dim=1).to_dense()
        deg_inv = 1.0 / deg
        deg_inv[deg_inv == float('inf')] = 0
        x = self.dropout(x)
        x = self.linear1(x)
        x = torch.sparse.mm(adj, x) * deg_inv.unsqueeze(1)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = torch.sparse.mm(adj, x) * deg_inv.unsqueeze(1)
        return x

# ==========================================
# KHU VỰC 2: TẢI HỆ THỐNG MÔ HÌNH & CHÌA KHÓA PCA
# ==========================================
st.set_page_config(page_title="ViST-Graph AI", page_icon="🔬", layout="wide")
st.title("🔬 ViST-Graph: Phân tích Hệ phiên mã từ ảnh Mô học")
st.markdown("**Sản phẩm dự thi Hội thi Tin học trẻ toàn quốc (Bảng D3)**")

@st.cache_resource 
def load_all_assets():
    # 1. Tải ResNet50 trích xuất đặc trưng
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval()

    # 2. Tải Trọng số GCN (best_vist_gcn_real.pth)
    model_id = '1bSRncH0wWJki2b8ghBWIWpO0TilG_JGY'
    model_path = 'best_vist_model.pth'
    if not os.path.exists(model_path):
        gdown.download(f'https://drive.google.com/uc?id={model_id}', model_path, quiet=False)
    
    gcn = ViST_GCN(in_feats=2048, hidden_feats=256, out_feats=50)
    gcn.load_state_dict(torch.load(model_path, map_location='cpu'))
    gcn.eval()

    # 3. Tải Chìa khóa PCA (gene_pca_model.pkl) để giải mã Top 10 Gene
    pca_id = '1wMMF7PxxVG5RkvfYhgGrbavKcC9mtNp8' 
    pca_path = 'gene_pca_model.pkl'
    pca_obj = None
    if pca_id != 'THAY_ID_FILE_PKL_CỦA_BẠN_VÀO_ĐÂY':
        if not os.path.exists(pca_path):
            gdown.download(f'https://drive.google.com/uc?id={pca_id}', pca_path, quiet=False)
        pca_obj = joblib.load(pca_path)
    
    return resnet, gcn, pca_obj

resnet, gcn_model, pca_model = load_all_assets()

# ==========================================
# KHU VỰC 3: XỬ LÝ HÌNH ẢNH & ĐỒ THỊ
# ==========================================
def run_pipeline(image_pil, grid_size=6):
    patch_size = 224
    # Ép về vuông để AI trích xuất đặc trưng công bằng
    img_square = image_pil.resize((patch_size * grid_size, patch_size * grid_size))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    features, coords = [], []
    for i in range(grid_size):
        for j in range(grid_size):
            patch = img_square.crop((j*patch_size, i*patch_size, (j+1)*patch_size, (i+1)*patch_size))
            with torch.no_grad():
                feat = resnet(transform(patch).unsqueeze(0)).squeeze()
            features.append(feat)
            coords.append([i, j])
            
    x = torch.stack(features)
    A = kneighbors_graph(np.array(coords), n_neighbors=6, mode='connectivity', include_self=True)
    edges = np.column_stack(np.where(A.toarray() == 1))
    edge_index = torch.tensor(edges.T, dtype=torch.long)
    
    return x, edge_index

# ==========================================
# KHU VỰC 4: GIAO DIỆN & CHẨN ĐOÁN
# ==========================================
uploaded_file = st.sidebar.file_uploader("Tải lên ảnh H&E (jpg/png)", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ảnh đầu vào")
        st.image(image, use_column_width=True)
        
    if st.button("🚀 THỰC THI VI TRÌNH PHÂN TÍCH", use_container_width=True):
        with st.spinner('Đang giải mã không gian đồ thị...'):
            x, edge_index = run_pipeline(image)
            with torch.no_grad():
                output = gcn_model(x, edge_index)
            
            # Vẽ Heatmap PC0 chuẩn tỷ lệ gốc
            pc0 = output[:, 0].numpy()
            pc0_norm = cv2.normalize(pc0, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            heatmap_resized = cv2.resize(pc0_norm.reshape((6, 6)), (image.size[0], image.size[1]), interpolation=cv2.INTER_CUBIC)
            heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
            
            overlay = cv2.addWeighted(np.array(image), 0.5, cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB), 0.5, 0)
            
            with col2:
                st.subheader("Bản đồ dự đoán (Lõi khối u - PC0)")
                st.image(overlay, use_column_width=True)
            
            # BIỂU ĐỒ PHÂN KỲ CHUẨN Y KHOA
            st.markdown("---")
            st.subheader("🧬 Định lượng Metagene (Kích hoạt vs Ức chế)")
            mean_vals = output.mean(dim=0)[:5].numpy()
            genes_labels = ['PC0 (Tumor)', 'MG 1', 'MG 2', 'MG 3', 'MG 4']
            
            fig, ax = plt.subplots(figsize=(10, 3.5))
            colors = ['#e74c3c' if v > 0 else '#3498db' for v in mean_vals]
            bars = ax.bar(genes_labels, mean_vals, color=colors)
            ax.axhline(0, color='black', linewidth=1)
            ax.set_ylim(min(mean_vals)-0.2, max(mean_vals)+0.2)
            for bar in bars:
                y = bar.get_height()
                ax.text(bar.get_x()+bar.get_width()/2, y+(0.02 if y>0 else -0.05), f"{y:.2f}", ha='center', va='bottom' if y>0 else 'top', fontweight='bold')
            st.pyplot(fig)

            # TÍNH NĂNG XAI: GIẢI THÍCH GENE CẤU THÀNH
            if pca_model:
                st.markdown("---")
                with st.expander("🔍 Chi tiết cấu trúc sinh học của các Metagene"):
                    loadings = pca_model.components_
                    target_mg = st.selectbox("Chọn Metagene để xem Top Gene đóng góp:", range(5))
                    weights = loadings[target_mg]
                    top_indices = np.argsort(weights)[-10:][::-1]
                    
                    st.write(f"**Top 10 Gene chủ đạo kích hoạt {genes_labels[target_mg]}:**")
                    cols = st.columns(5)
                    for idx, g_idx in enumerate(top_indices):
                        cols[idx % 5].code(f"Gene ID: {g_idx}\nWeight: {weights[g_idx]:.4f}")
            else:
                st.info("💡 Lưu ý: Hãy cập nhật PCA Model ID để xem chi tiết các gene cấu thành.")
