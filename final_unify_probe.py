import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# ================= 🔧 統一參數設定 =================
AUDIO_ROOT = "/export/fs05/hyeh10/depression/daic_5utt_full/merged_5"
CSV_A_TEST = "experiment_sisman_scientific/scenario_A_screening/test.csv"
CSV_B_TRAIN = "experiment_sisman_scientific/scenario_B_monitoring/train.csv"
CSV_B_TEST = "experiment_sisman_scientific/scenario_B_monitoring/test.csv"

# 模型路徑
PATH_BASELINE = "best_model_frozen_weighted.pth"
PATH_DANN_A = "best_dann_model.pth"
PATH_DANN_B = "dann_model_scenario_B_final.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= 🧠 模型結構定義 =================
class DANN_Encoder(nn.Module):
    def __init__(self, is_dann=True):
        super().__init__()
        self.w2v = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.is_dann = is_dann
        if is_dann:
            self.shared_layer = nn.Sequential(
                nn.Linear(768, 128),
                nn.BatchNorm1d(128),
                nn.ReLU()
            )
    def forward(self, x):
        feat = self.w2v(x).last_hidden_state.mean(dim=1)
        if self.is_dann:
            feat = self.shared_layer(feat)
        return feat

# ================= 📂 特徵提取函數 =================
def get_features(model, csv_path, processor):
    df = pd.read_csv(csv_path)
    # 解析 Speaker ID
    df['spk'] = df['path'].apply(lambda x: str(x).split('/')[-1].split('_')[0])
    feats, spks = [], []
    model.eval()
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Reading {os.path.basename(csv_path)}", leave=False):
            wav_path = os.path.join(AUDIO_ROOT, row['path'])
            try:
                s, sr = torchaudio.load(wav_path)
                if sr != 16000: s = torchaudio.transforms.Resample(sr, 16000)(s)
                # 限制長度以免 OOM
                if s.shape[1] > 16000*8: s = s[:, :16000*8]
                inp = processor(s.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_values.to(DEVICE)
                emb = model(inp).cpu().numpy().squeeze()
                feats.append(emb)
                spks.append(row['spk'])
            except: pass
    return np.array(feats), np.array(spks)

# ================= 🚀 主程序 =================
if __name__ == "__main__":
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    results = {}

    exp_configs = [
        {"name": "Baseline_A", "path": PATH_BASELINE, "is_dann": False, "csv_tr": CSV_A_TEST, "csv_te": CSV_A_TEST, "mode": "cv"},
        {"name": "DANN_A", "path": PATH_DANN_A, "is_dann": True, "csv_tr": CSV_A_TEST, "csv_te": CSV_A_TEST, "mode": "cv"},
        {"name": "Baseline_B", "path": PATH_BASELINE, "is_dann": False, "csv_tr": CSV_B_TRAIN, "csv_te": CSV_B_TEST, "mode": "split"},
        {"name": "DANN_B", "path": PATH_DANN_B, "is_dann": True, "csv_tr": CSV_B_TRAIN, "csv_te": CSV_B_TEST, "mode": "split"},
    ]

    for config in exp_configs:
        print(f"\n🔎 正在運行: {config['name']}")
        model = DANN_Encoder(is_dann=config['is_dann']).to(DEVICE)
        
        # --- 🔥 關鍵修正：智慧權重載入邏輯 ---
        if os.path.exists(config['path']):
            raw_state_dict = torch.load(config['path'], map_location=DEVICE)
            new_state_dict = {}
            
            for k, v in raw_state_dict.items():
                # 修正 1: 處理沒有前綴的 shared_layer (如 0.weight -> shared_layer.0.weight)
                if config['is_dann'] and (k.startswith("0.") or k.startswith("1.") or k.startswith("2.")):
                    new_key = "shared_layer." + k
                    new_state_dict[new_key] = v
                # 修正 2: 保留已經正確的 shared_layer
                elif "shared_layer" in k:
                    new_state_dict[k] = v
                # 修正 3: 保留 Wav2Vec2 相關權重 (如果有微調過)
                else:
                    new_state_dict[k] = v
            
            # 載入權重 (strict=False 是為了忽略 discriminator 等多餘的層，但我們已確保 shared_layer 對齊)
            load_res = model.load_state_dict(new_state_dict, strict=False)
            
            # 簡單驗證
            if config['is_dann']:
                missing = [k for k in load_res.missing_keys if "shared_layer" in k]
                if missing:
                    print(f"⚠️ 嚴重警告: {config['name']} 的 shared_layer 仍有缺失！{missing}")
                else:
                    print(f"✅ {config['name']} 權重載入成功 (Shared Layer 已對齊)")
        else:
            print(f"❌ 找不到模型檔案: {config['path']}，將使用隨機初始權重！")

        # --- 開始特徵提取與評估 ---
        if config['mode'] == "cv":
            # Scenario A: 5-Fold Cross Validation
            X, y = get_features(model, config['csv_te'], processor)
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            accs = []
            # 為了避免某些 Fold 沒有樣本，加個簡單的錯誤處理
            try:
                for tr_idx, te_idx in skf.split(X, y):
                    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
                    clf.fit(X[tr_idx], y[tr_idx])
                    accs.append(accuracy_score(y[te_idx], clf.predict(X[te_idx])))
                results[config['name']] = np.mean(accs)
            except Exception as e:
                print(f"⚠️ CV 執行錯誤 (可能是樣本太少): {e}")
                results[config['name']] = 0.0
        else:
            # Scenario B: Train/Test Split
            X_tr, y_tr = get_features(model, config['csv_tr'], processor)
            X_te, y_te = get_features(model, config['csv_te'], processor)
            
            # 檢查是否有足夠的訓練數據
            if len(set(y_tr)) > 1:
                clf = LogisticRegression(max_iter=1000, class_weight='balanced')
                clf.fit(X_tr, y_tr)
                # 確保測試集裡的 Speaker 在訓練集裡見過 (針對 Probe 任務)
                # 但這裡是測洩漏，所以直接測也無妨，沒看過的就猜錯，符合邏輯
                results[config['name']] = accuracy_score(y_te, clf.predict(X_te))
            else:
                print("⚠️ 訓練集只有一類 Speaker，無法訓練 Probe。")
                results[config['name']] = 0.0

    # 📊 輸出最終結果
    print("\n" + "="*50)
    print("🏆 最終統一標準結果 (Speaker Accuracy)")
    print("="*50)
    for name, acc in results.items():
        print(f"{name}: {acc*100:.2f}%")