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
# KHU VỰC 1: ĐỊNH NGHĨA KIẾN TRÚC MÔ HÌNH (CHUẨN GCN)
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
# KHU VỰC 2: TẢI TẤT CẢ TÀI SẢN (ASSETS) TỪ DRIVE
# ==========================================
st.set_page_config(page_title="ViST-Graph Analysis", page_icon="🔬", layout="wide")
st.title("🔬 ViST-Graph: Phân tích Hệ phiên mã Không gian")
st.markdown("**Sản phẩm dự thi Hội thi Tin học trẻ toàn quốc (Bảng D3)**")

@st.cache_resource 
def load_all_assets():
    # 1. ResNet50
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval()

    # 2. GCN Model (Sửa ID nếu thay đổi)
    model_id = '1bSRncH0wWJki2b8ghBWIWpO0TilG_JGY'
    if not os.path.exists('best_vist_model.pth'):
        gdown.download(f'https://drive.google.com/uc?id={model_id}', 'best_vist_model.pth', quiet=False)
    gcn = ViST_GCN(in_feats=2048, hidden_feats=256, out_feats=50)
    gcn.load_state_dict(torch.load('best_vist_model.pth', map_location='cpu'))
    gcn.eval()

    # 3. PCA Key (Để giải mã Metagene)
    pca_id = '1wMMF7PxxVG5RkvfYhgGrbavKcC9mtNp8' 
    if pca_id != 'ID_FILE_GENE_PCA_MODEL_CỦA_BẠN' and not os.path.exists('gene_pca_model.pkl'):
        gdown.download(f'https://drive.google.com/uc?id={pca_id}', 'gene_pca_model.pkl', quiet=False)
    pca_obj = joblib.load('gene_pca_model.pkl') if os.path.exists('gene_pca_model.pkl') else None

    # 4. Gene Name Mapping (Để hiện tên BRCA1, TP53...)
    map_id = '1h0UgTQqA71UCRlvsHrAyVqNeLS1UEdAu'
    if map_id != 'ID_FILE_GENE_NAMES_MAPPING_CỦA_BẠN' and not os.path.exists('gene_names_mapping.pkl'):
        gdown.download(f'https://drive.google.com/uc?id={map_id}', 'gene_names_mapping.pkl', quiet=False)
    gene_map = joblib.load('gene_names_mapping.pkl') if os.path.exists('gene_names_mapping.pkl') else None
    
    return resnet, gcn, pca_obj, gene_map

resnet, gcn_model, pca_model, gene_mapping = load_all_assets()

# ==========================================
# KHU VỰC 3: HÀM XỬ LÝ ĐỒ THỊ
# ==========================================
def run_pipeline(image_pil, grid_size=6):
    patch_size = 224
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
    edge_index = torch.tensor(np.column_stack(np.where(A.toarray() == 1)).T, dtype=torch.long)
    return x, edge_index

# ==========================================
# KHU VỰC 4: GIAO DIỆN & CHẨN ĐOÁN CHI TIẾT
# ==========================================
uploaded_file = st.sidebar.file_uploader("Tải lên ảnh H&E", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    
    if "analysis_done" not in st.session_state:
        st.session_state.analysis_done = False

    if st.sidebar.button("🚀 BẮT ĐẦU PHÂN TÍCH TỔNG LỰC", use_container_width=True):
        with st.spinner('AI đang giải mã mạng lưới tế bào...'):
            x, edge_index = run_pipeline(image)
            with torch.no_grad():
                st.session_state.output = gcn_model(x, edge_index)
            st.session_state.analysis_done = True

    if st.session_state.analysis_done:
        st.markdown("### 🔍 Trung tâm Điều hành Phân tích Không gian")
        
        # CHỌN METAGENE ĐỂ HIỂN THỊ DYNAMIC
        target_mg = st.selectbox(
            "Chọn Siêu gene (Metagene) muốn hiển thị bản đồ nhiệt:", 
            range(50), 
            format_func=lambda x: f"Metagene {x} " + ("(Lõi khối u - PC0)" if x==0 else "")
        )

        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Ảnh đầu vào gốc")
            st.image(image, use_column_width=True)

        with col2:
            st.subheader(f"Bản đồ nhiệt: Metagene {target_mg}")
            # Xử lý heatmap động dựa trên lựa chọn
            vals = st.session_state.output[:, target_mg].numpy()
            norm = cv2.normalize(vals, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            heatmap = cv2.resize(norm.reshape((6, 6)), (image.size[0], image.size[1]), interpolation=cv2.INTER_CUBIC)
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(np.array(image), 0.5, cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB), 0.5, 0)
            st.image(overlay, use_column_width=True)

        # BIỂU ĐỒ PHÂN KỲ CHUẨN Y KHOA
        st.markdown("---")
        st.subheader("🧬 Định lượng biểu hiện Top 5 Metagene cốt lõi")
        mean_vals = st.session_state.output.mean(dim=0)[:5].numpy()
        labels = ['PC0', 'MG 1', 'MG 2', 'MG 3', 'MG 4']
        fig, ax = plt.subplots(figsize=(10, 3))
        colors = ['#e74c3c' if v > 0 else '#3498db' for v in mean_vals]
        bars = ax.bar(labels, mean_vals, color=colors)
        ax.axhline(0, color='black', linewidth=1)
        for bar in bars:
            y = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, y+(0.02 if y>0 else -0.05), f"{y:.2f}", ha='center', va='bottom' if y>0 else 'top', fontweight='bold')
        st.pyplot(fig)

        # CHI TIẾT GENE (XAI) - ÁNH XẠ TÊN GENE
        if pca_model:
            st.markdown("---")
            with st.expander(f"🔬 Xem 10 Gene chủ đạo cấu thành Metagene {target_mg}"):
                weights = pca_model.components_[target_mg]
                top_idx = np.argsort(weights)[-10:][::-1]
                
                st.write(f"Các gene này định nghĩa chức năng sinh học của Metagene {target_mg}:")
                cols = st.columns(5)
                for i, idx in enumerate(top_idx):
                    # HIỂN THỊ TÊN GENE THẬT NẾU CÓ MAPPING
                    g_name = gene_mapping[idx] if gene_mapping else f"Gene ID: {idx}"
                    cols[i % 5].success(f"**{g_name}**\n\nWeight: {weights[idx]:.4f}")
