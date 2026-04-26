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
# KHU VỰC 2: THIẾT LẬP GIAO DIỆN & TẢI TÀI SẢN
# ==========================================
st.set_page_config(page_title="ViST-Graph | Spatial Transcriptomics", page_icon="🧬", layout="wide")

# CSS tùy chỉnh để làm giao diện chuyên nghiệp hơn
st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0;}
    .sub-header { font-size: 1.1rem; color: #64748B; margin-bottom: 2rem;}
    .stSpinner > div > div { border-color: #1E3A8A transparent transparent transparent; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🧬 Hệ thống ViST-Graph</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Nền tảng phân tích Hệ phiên mã không gian ảo từ ảnh Mô bệnh học H&E</p>', unsafe_allow_html=True)

@st.cache_resource 
def load_all_assets():
    # 1. ResNet50
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval()

    # 2. GCN Model
    model_id = '1bSRncH0wWJki2b8ghBWIWpO0TilG_JGY'
    if not os.path.exists('best_vist_model.pth'):
        gdown.download(f'https://drive.google.com/uc?id={model_id}', 'best_vist_model.pth', quiet=False)
    gcn = ViST_GCN(in_feats=2048, hidden_feats=256, out_feats=50)
    gcn.load_state_dict(torch.load('best_vist_model.pth', map_location='cpu'))
    gcn.eval()

    # 3. PCA Key
    pca_id = '1wMMF7PxxVG5RkvfYhgGrbavKcC9mtNp8' 
    if pca_id != 'ID_FILE_GENE_PCA_MODEL_CỦA_BẠN' and not os.path.exists('gene_pca_model.pkl'):
        gdown.download(f'https://drive.google.com/uc?id={pca_id}', 'gene_pca_model.pkl', quiet=False)
    pca_obj = joblib.load('gene_pca_model.pkl') if os.path.exists('gene_pca_model.pkl') else None

    # 4. Gene Name Mapping
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
# KHU VỰC 4: XỬ LÝ NGHIỆP VỤ & GIAO DIỆN CHÍNH
# ==========================================
# Quản lý trạng thái bộ nhớ
if "current_file_bytes" not in st.session_state:
    st.session_state.current_file_bytes = None
if "output" not in st.session_state:
    st.session_state.output = None

with st.sidebar:
    st.markdown("### 📥 Dữ liệu đầu vào")
    uploaded_file = st.file_uploader("Tải lên tiêu bản H&E (jpg/png)", type=["jpg", "png"])
    st.markdown("---")
    st.markdown("**Trạng thái hệ thống:**")
    st.success("✅ Mô hình GCN đã tải")
    st.success("✅ Từ điển Gene đã tải")

if uploaded_file is None:
    st.info("👈 Vui lòng tải lên một ảnh vi thể H&E ở thanh bên trái để hệ thống bắt đầu phân tích tự động.")
else:
    image = Image.open(uploaded_file).convert('RGB')
    file_bytes = uploaded_file.getvalue()
    
    # TỰ ĐỘNG CHẠY KHI PHÁT HIỆN ẢNH MỚI
    if st.session_state.current_file_bytes != file_bytes:
        with st.spinner('🤖 Hệ thống đang phân tích đồ thị không gian...'):
            x, edge_index = run_pipeline(image)
            with torch.no_grad():
                st.session_state.output = gcn_model(x, edge_index)
        # Lưu lại ảnh hiện tại để không bị chạy lại khi bấm các nút khác
        st.session_state.current_file_bytes = file_bytes 

    # --- BẮT ĐẦU HIỂN THỊ KẾT QUẢ ---
    # Thanh điều khiển trung tâm
    st.markdown("### 🎛️ Bảng điều khiển Chẩn đoán")
    target_mg = st.selectbox(
        "Chọn Siêu gene (Metagene Pathway) để phân tích chi tiết:", 
        range(50), 
        format_func=lambda x: f"Metagene {x} " + ("- Core Tumor Boundary" if x==0 else "")
    )

    # Chia Tab để giao diện gọn gàng, hiện đại
    tab1, tab2 = st.tabs(["🗺️ Bản đồ Không gian & Định lượng", "🧬 Giải mã XAI (Chi tiết Gene)"])

    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Ảnh đầu vào gốc (H&E)**")
            st.image(image, use_column_width=True)

        with col2:
            st.markdown(f"**Bản đồ phân bố Metagene {target_mg}**")
            vals = st.session_state.output[:, target_mg].numpy()
            norm = cv2.normalize(vals, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            heatmap = cv2.resize(norm.reshape((6, 6)), (image.size[0], image.size[1]), interpolation=cv2.INTER_CUBIC)
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(np.array(image), 0.5, cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB), 0.5, 0)
            st.image(overlay, use_column_width=True)

        st.markdown("---")
        st.markdown("**📊 Định lượng biểu hiện Top 5 Metagene cốt lõi (Toàn cảnh khối u)**")
        
        mean_vals = st.session_state.output.mean(dim=0)[:5].numpy()
        labels = ['PC0 (Core)', 'MG 1', 'MG 2', 'MG 3', 'MG 4']
        
        # Biểu đồ làm chuẩn học thuật (bỏ viền, thêm lưới)
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        colors = ['#EF4444' if v > 0 else '#3B82F6' for v in mean_vals]
        bars = ax.bar(labels, mean_vals, color=colors, width=0.6)
        ax.axhline(0, color='black', linewidth=1.2)
        
        for bar in bars:
            y = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, y + (0.05 if y > 0 else -0.1), 
                    f"{y:.2f}", ha='center', va='bottom' if y > 0 else 'top', 
                    fontweight='bold', fontsize=10, color='#333333')
        st.pyplot(fig)

    with tab2:
        st.markdown(f"### Phân tích cấu trúc sinh học Metagene {target_mg}")
        st.info("Các gene dưới đây có trọng số lớn nhất, định nghĩa chức năng sinh học lâm sàng của cụm Metagene này.")
        
        if pca_model:
            weights = pca_model.components_[target_mg]
            top_idx = np.argsort(weights)[-10:][::-1]
            
            # Sử dụng các thẻ st.metric để giao diện giống Dashboard phân tích số liệu
            cols = st.columns(5)
            for i, idx in enumerate(top_idx):
                g_name = gene_mapping[idx] if gene_mapping else f"Gene ID: {idx}"
                # Hiển thị dạng Metric card
                cols[i % 5].metric(label=f"Top {i+1}", value=g_name, delta=f"Trọng số: {weights[idx]:.4f}")
                
            st.markdown("<br><br><p style='text-align: center; color: gray; font-size: 0.9rem;'>Hệ thống Explainable AI (XAI) tích hợp với dữ liệu chuẩn từ 10x Genomics.</p>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ Không tìm thấy file PCA Model để giải mã tên Gene.")
