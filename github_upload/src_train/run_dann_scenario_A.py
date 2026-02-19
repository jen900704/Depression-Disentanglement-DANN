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
# 🔧 1. 設定區 (Scenario A Config)
# ==========================================
# 🔥 改成 Scenario A 的路徑
TRAIN_CSV_PATH = "./experiment_sisman_scientific/scenario_A_screening/train.csv"
TEST_CSV_PATH = "./experiment_sisman_scientific/scenario_A_screening/test.csv"
AUDIO_ROOT = "/export/fs05/hyeh10/depression/daic_5utt_full/merged_5"  # 填入您的正確路徑

MODEL_NAME = "facebook/wav2vec2-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 30  
TOTAL_RUNS = 5 

print(f"🖥️ 使用裝置: {DEVICE}")
print(f"📂 執行 Scenario A (Strict Split) 實驗")

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
            nn.Linear(64, num_speakers) # 這裡的 num_speakers 是訓練集的總人數
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

    # 建立或使用 Speaker Map
    if is_train and speaker_to_idx is None:
        all_speakers = df['path'].apply(extract_speaker_id).unique()
        all_speakers = sorted(all_speakers)
        speaker_to_idx = {spk: idx for idx, spk in enumerate(all_speakers)}
        print(f"🔍 [訓練集] 共有 {len(all_speakers)} 位 Speaker。")
    
    model.eval()
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Features"):
            wav_path = os.path.join(AUDIO_ROOT, row['path'])
            try:
                # 簡單檢查檔案是否存在
                if not os.path.exists(wav_path):
                    continue

                waveform, sample_rate = torchaudio.load(wav_path)
                if sample_rate != 16000:
                    waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
                # 平均多聲道
                if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
                # 長度截斷 (避免 OOM)
                if waveform.shape[1] > 16000 * 8: waveform = waveform[:, :16000*8]
                
                raw_label = str(row['label']).strip().lower()
                if raw_label in label_map:
                    final_label = label_map[raw_label]
                else:
                    continue

                inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                # 取 Wav2Vec2 的最後一層特徵並做 Mean Pooling
                embeddings = model(**inputs).last_hidden_state.mean(dim=1).cpu()
                
                features_list.append(embeddings)
                labels_list.append(final_label)
                
                spk_str = extract_speaker_id(wav_path)
                # Scenario A 關鍵處理：如果是測試集且沒看過這個人，給 -1
                if spk_str in speaker_to_idx:
                    speaker_indices_list.append(speaker_to_idx[spk_str])
                else:
                    speaker_indices_list.append(-1) # Unknown speaker
                
            except Exception as e:
                # print(f"⚠️ Error: {wav_path} -> {e}")
                continue

    if len(features_list) == 0:
        raise ValueError("❌ 錯誤：沒有任何資料被成功讀取！")

    X = torch.cat(features_list, dim=0)
    y = torch.tensor(labels_list, dtype=torch.long)
    s = torch.tensor(speaker_indices_list, dtype=torch.long)
    return X, y, s, speaker_to_idx

def get_feats(model, loader):
    model.eval()
    feats = []
    spks = []
    with torch.no_grad():
        for inputs, labels, speakers in loader:
            inputs = inputs.to(DEVICE)
            # 只取 shared_encoder 出來的特徵 (這就是 DANN 清洗後的特徵)
            f = model.shared_encoder(inputs).cpu().numpy()
            feats.append(f)
            spks.extend(speakers.cpu().numpy())
    return np.vstack(feats), np.array(spks)

# ==========================================
# 🚀 4. 主程式執行 (修正版：加入存檔與 F1)
# ==========================================
if __name__ == "__main__":
    # --- A. 準備特徵提取器 (凍結版) ---
    print("🧠 載入 Wav2Vec2 模型...")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    w2v_model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(DEVICE)
    
    # --- B. 準備資料 ---
    print("\n📦 正在準備資料 (特徵提取只會執行一次)...")
    X_train, y_train, s_train, speaker_map = prepare_data(TRAIN_CSV_PATH, processor, w2v_model, is_train=True)
    X_test, y_test, s_test, _ = prepare_data(TEST_CSV_PATH, processor, w2v_model, speaker_to_idx=speaker_map, is_train=False)
    
    num_speakers = len(speaker_map)
    print(f"👥 訓練集 Speaker 數量: {num_speakers}")
    
    train_dataset = TensorDataset(X_train, y_train, s_train)
    test_dataset = TensorDataset(X_test, y_test, s_test)

    # --- C. 開始實驗迴圈 ---
    for run_i in range(1, TOTAL_RUNS + 1):
        print(f"\n{'='*60}")
        print(f"🎬 [Scenario A] 開始第 {run_i} / {TOTAL_RUNS} 次實驗")
        print(f"{'='*60}")
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        print(f"🏗️ 初始化 DANN 模型...")
        dann_model = DANN_Model(num_speakers=num_speakers).to(DEVICE)
        
        optimizer = optim.Adam(dann_model.parameters(), lr=0.001)
        criterion_class = nn.CrossEntropyLoss()
        criterion_domain = nn.CrossEntropyLoss()
        
        print("⚔️ 開始訓練...")
        for epoch in range(EPOCHS):
            dann_model.train()
            total_loss = 0
            correct_train_spk = 0
            total_train_samples = 0
            
            p = float(epoch) / EPOCHS
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1
            
            for inputs, labels, speakers in train_loader:
                inputs, labels, speakers = inputs.to(DEVICE), labels.to(DEVICE), speakers.to(DEVICE)
                
                optimizer.zero_grad()
                class_out, domain_out = dann_model(inputs, alpha=alpha)
                
                loss_class = criterion_class(class_out, labels)
                loss_domain = criterion_domain(domain_out, speakers)
                loss = loss_class + loss_domain
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, spk_preds = torch.max(domain_out, 1)
                correct_train_spk += (spk_preds == speakers).sum().item()
                total_train_samples += speakers.size(0)
            
            train_spk_acc = correct_train_spk / total_train_samples

            # --- 評估 (Validation) ---
            if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
                dann_model.eval()
                all_preds = []
                all_labels = []
                with torch.no_grad():
                    for inputs, labels, speakers in test_loader:
                        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                        class_out, _ = dann_model(inputs, alpha=0)
                        _, preds = torch.max(class_out, 1)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                
                test_acc = accuracy_score(all_labels, all_preds)
                # 🔥 這裡已經有 F1 了，很好！
                test_f1 = f1_score(all_labels, all_preds, average='macro')
                
                print(f"Epoch {epoch+1:02d} | Loss: {total_loss:.2f} | Train Spk Acc: {train_spk_acc:.4f} (↓) | Test Dep Acc: {test_acc:.4f} (↑) | Test F1: {test_f1:.4f}")

        # 4. 畫 t-SNE
        print(f"\n🎨 [Run {run_i}] 繪製 t-SNE...")
        train_feats_tsne, train_spks_tsne = get_feats(dann_model, train_loader)
        tsne = TSNE(n_components=2, random_state=42 + run_i, perplexity=30)
        feats_2d = tsne.fit_transform(train_feats_tsne)
        
        plt.figure(figsize=(10, 8))
        limit = 500
        sns.scatterplot(x=feats_2d[:limit,0], y=feats_2d[:limit,1], hue=train_spks_tsne[:limit], palette="tab10", legend=False)
        plt.title(f"DANN Feature Space (Training Set) - Run {run_i}\n(Goal: Mixed Colors = Privacy Preserved)", fontsize=16)
        filename = f"tsne_dann_scenario_A_run_{run_i}.png"
        plt.savefig(filename)
        plt.close()
        print(f"✅ 圖片已儲存: {filename}")

        # 🔥🔥🔥 5. 關鍵新增：儲存模型！ 🔥🔥🔥
        # 這樣 Probe 腳本才能讀取這個訓練好的模型
        save_path = f"dann_model_run_{run_i}.pth"
        torch.save(dann_model.state_dict(), save_path)
        print(f"💾 模型已儲存至: {save_path} (可用於後續 Probe 測試)")