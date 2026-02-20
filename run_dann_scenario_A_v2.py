"""
新版 File 4 — DANN Scenario A (Screening / No Speaker Overlap)
==============================================================
目標：與 File 5 (run_dann.py) 完全一致的 DANN 訓練方法論
唯一差異：資料路徑為 scenario_A_screening

與 run_dann.py (File 5) 的差異清單：
  → TRAIN_CSV_PATH: scenario_B_monitoring → scenario_A_screening
  → TEST_CSV_PATH:  scenario_B_monitoring → scenario_A_screening
  → t-SNE 存檔名：tsne_dann_run_{i}.png → tsne_dann_A_run_{i}.png
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Function
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torchaudio
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 🔧 1. 設定區 (Config) — 唯一差異：資料路徑
# ==========================================
TRAIN_CSV_PATH = "./experiment_sisman_scientific/scenario_A_screening/train.csv"
TEST_CSV_PATH = "./experiment_sisman_scientific/scenario_A_screening/test.csv"
AUDIO_ROOT = "" 

MODEL_NAME = "facebook/wav2vec2-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 30  
TOTAL_RUNS = 5  # 🔥 設定總共要跑幾次

print(f"🖥️ 使用裝置: {DEVICE}")

# ==========================================
# 🧠 2. 模型定義 (DANN Architecture)
# ==========================================
class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReversalLayer(nn.Module):
    def __init__(self):
        super(GradientReversalLayer, self).__init__()
    def forward(self, x, alpha=1.0):
        return GradientReversalFn.apply(x, alpha)

class DANN_Model(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128, num_classes=2, num_speakers=38):
        super(DANN_Model, self).__init__()
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.class_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_speakers)
        )
        self.grl = GradientReversalLayer()

    def forward(self, x, alpha=1.0):
        features = self.shared_encoder(x)
        class_output = self.class_classifier(features)
        reverse_features = self.grl(features, alpha)
        domain_output = self.domain_classifier(reverse_features)
        return class_output, domain_output

# ==========================================
# 📂 3. 資料處理工具 (Data Utils)
# ==========================================
def extract_speaker_id(filepath):
    filename = os.path.basename(filepath)
    speaker_id = filename.split('_')[0] 
    return speaker_id

def prepare_data(csv_path, processor, model, speaker_to_idx=None, is_train=True):
    df = pd.read_csv(csv_path)
    print(f"📂 正在處理 {csv_path} (共 {len(df)} 筆)...")
    
    features_list = []
    labels_list = []
    speaker_indices_list = []
    
    label_map = {'dep': 1, '1': 1, 1: 1, 'non': 0, '0': 0, 0: 0}

    if is_train and speaker_to_idx is None:
        all_speakers = df['path'].apply(extract_speaker_id).unique()
        all_speakers = sorted(all_speakers)
        speaker_to_idx = {spk: idx for idx, spk in enumerate(all_speakers)}
        print(f"🔍 [訓練集] Speaker Map: {list(speaker_to_idx.items())[:5]}...")
    
    model.eval()
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Features"):
            wav_path = os.path.join(AUDIO_ROOT, row['path'])
            try:
                waveform, sample_rate = torchaudio.load(wav_path)
                if sample_rate != 16000:
                    waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
                if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                raw_label = str(row['label']).strip().lower()
                if raw_label in label_map:
                    final_label = label_map[raw_label]
                else:
                    continue

                inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                embeddings = model(**inputs).last_hidden_state.mean(dim=1).cpu()
                
                features_list.append(embeddings)
                labels_list.append(final_label)
                
                spk_str = extract_speaker_id(wav_path)
                speaker_indices_list.append(speaker_to_idx.get(spk_str, 0))
                
            except Exception as e:
                print(f"⚠️ Error: {wav_path} -> {e}")
                continue

    if len(features_list) == 0:
        raise ValueError("❌ 錯誤：沒有任何資料被成功讀取！")

    X = torch.cat(features_list, dim=0)
    y = torch.tensor(labels_list, dtype=torch.long)
    s = torch.tensor(speaker_indices_list, dtype=torch.long)
    return X, y, s, speaker_to_idx

# 輔助函式：用來從 DANN 模型抽取特徵畫圖
def get_feats(model, loader):
    model.eval()
    feats = []
    spks = []
    with torch.no_grad():
        for inputs, labels, speakers in loader:
            inputs = inputs.to(DEVICE)
            f = model.shared_encoder(inputs).cpu().numpy()
            feats.append(f)
            spks.extend(speakers.cpu().numpy())
    return np.vstack(feats), np.array(spks)

# ==========================================
# 🚀 4. 主程式執行
# ==========================================
if __name__ == "__main__":
    # --- A. 準備特徵提取器 (凍結版) ---
    print("🧠 載入 Wav2Vec2 模型...")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    w2v_model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(DEVICE)
    
    # --- B. 準備資料 (只做一次！) ---
    print("\n📦 正在準備資料 (特徵提取只會執行一次)...")
    X_train, y_train, s_train, speaker_map = prepare_data(TRAIN_CSV_PATH, processor, w2v_model, is_train=True)
    X_test, y_test, s_test, _ = prepare_data(TEST_CSV_PATH, processor, w2v_model, speaker_to_idx=speaker_map, is_train=False)
    
    num_speakers = len(speaker_map)
    
    # 建立 Dataset (Tensor 不會變，Loader 在迴圈內重建即可)
    train_dataset = TensorDataset(X_train, y_train, s_train)
    test_dataset = TensorDataset(X_test, y_test, s_test)

    # --- C. 初始化結果收集容器 ---
    all_run_accs = []
    all_run_f1s = []

    # --- D. 開始 5 次實驗迴圈 ---
    for run_i in range(1, TOTAL_RUNS + 1):
        print(f"\n{'='*60}")
        print(f"🎬 開始第 {run_i} / {TOTAL_RUNS} 次實驗 (Run {run_i})")
        print(f"{'='*60}")
        
        # 1. 每次都要重新建立 Loader (Shuffle 確保隨機性)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # 2. 每次都要重新初始化模型 (確保權重重置)
        print(f"🏗️ 初始化全新的 DANN 模型...")
        dann_model = DANN_Model(num_speakers=num_speakers).to(DEVICE)
        
        optimizer = optim.Adam(dann_model.parameters(), lr=0.001)
        criterion_class = nn.CrossEntropyLoss()
        criterion_domain = nn.CrossEntropyLoss()
        
        # 3. 訓練迴圈
        print("⚔️ 開始訓練...")
        for epoch in range(EPOCHS):
            dann_model.train()
            total_loss = 0
            
            # 動態調整 alpha
            p = float(epoch) / EPOCHS
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1
            
            for inputs, labels, speakers in train_loader:
                inputs, labels, speakers = inputs.to(DEVICE), labels.to(DEVICE), speakers.to(DEVICE)
                
                optimizer.zero_grad()
                class_out, domain_out = dann_model(inputs, alpha=alpha)
                loss = criterion_class(class_out, labels) + criterion_domain(domain_out, speakers)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # 每個 Epoch 簡單評估一下
            if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
                dann_model.eval()
                correct_speakers = 0
                all_preds = []
                all_labels = []
                total_samples = 0
                with torch.no_grad():
                    for inputs, labels, speakers in test_loader:
                        inputs, labels, speakers = inputs.to(DEVICE), labels.to(DEVICE), speakers.to(DEVICE)
                        class_out, domain_out = dann_model(inputs, alpha=0)
                        
                        _, preds = torch.max(class_out, 1)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                        
                        _, spk_preds = torch.max(domain_out, 1)
                        correct_speakers += (spk_preds == speakers).sum().item()
                        total_samples += labels.size(0)
                
                acc = accuracy_score(all_labels, all_preds)
                f1 = f1_score(all_labels, all_preds, average='macro')  # ✅ 修正：補上 f1 計算
                spk_acc = correct_speakers / total_samples
                print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.2f} | Dep Acc: {acc:.4f} | Dep F1: {f1:.4f} | Spk Acc: {spk_acc:.4f}")

                # 🔥 最後一個 epoch 才收集結果
                if epoch == EPOCHS - 1:
                    all_run_accs.append(acc)
                    all_run_f1s.append(f1)

        # 4. 畫圖 (t-SNE) - 存成不同的檔名
        print(f"\n🎨 [Run {run_i}] 正在繪製 t-SNE 圖...")
        dann_feats, dann_spks = get_feats(dann_model, test_loader)
        
        tsne = TSNE(n_components=2, random_state=42 + run_i, perplexity=30)
        feats_2d = tsne.fit_transform(dann_feats)
        
        plt.figure(figsize=(10, 8))
        limit = None
        sns.scatterplot(x=feats_2d[:limit,0], y=feats_2d[:limit,1], hue=dann_spks[:limit], palette="tab10", legend=False)
        plt.title(f"DANN Feature Space (Scenario A) - Run {run_i}", fontsize=16)
        
        filename = f"tsne_dann_A_run_{run_i}.png"
        plt.savefig(filename)
        plt.close()
        print(f"✅ 圖片已儲存: {filename}")

    # --- E. 5 次實驗總結 ---
    print(f"\n{'='*60}")
    print(f"📊 {TOTAL_RUNS} 次實驗結果統計 (Scenario A)")
    print(f"{'='*60}")
    accs = np.array(all_run_accs)
    f1s = np.array(all_run_f1s)
    for i, (a, f) in enumerate(zip(accs, f1s), 1):
        print(f"  Run {i}: Acc={a:.4f}, F1={f:.4f}")
    print(f"{'─'*40}")
    print(f"  平均 Acc : {accs.mean():.4f} ± {accs.std():.4f}")
    print(f"  平均 F1  : {f1s.mean():.4f} ± {f1s.std():.4f}")
