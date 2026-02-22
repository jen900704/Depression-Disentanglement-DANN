"""
Speaker Probe — Scenario B (Longitudinal) 統一隱私/身分殘留評估
====================================================
邏輯：對四種架構的 embedding，用 Logistic Regression 預測 Speaker ID。
      Speaker Accuracy 越低 → 模型越成功去除 speaker 資訊。

評估模型 (Scenario B 限定):
  1. Linear Probing → 原始凍結 Wav2Vec2 (768D)
  2. Baseline (Huang) → 微調後 Wav2Vec2 + mean pooling (768D)
  3. Static DANN → 凍結 Wav2Vec2 + 訓練好的 shared_encoder (128D)
  4. DANN-FT → 微調後 Wav2Vec2 + 微調後 shared_encoder (128D)
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2PreTrainedModel, AutoConfig
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import safetensors.torch

# ============================================================
#  ======= 路徑設定區 =======
# ============================================================
AUDIO_ROOT = "" 
CSV_B_TRAIN = "./experiment_sisman_scientific/scenario_B_monitoring/train.csv"
CSV_B_TEST  = "./experiment_sisman_scientific/scenario_B_monitoring/test.csv"

# 模型儲存路徑樣板
HUANG_B_MODEL_TEMPLATE  = "./output_scenario_B_v2/run_{run_i}/best_model"
STATIC_DANN_B_TEMPLATE  = "./dann_B_shared_encoder_run_{run_i}.pth" # 記得在 Static DANN 訓練腳本中存檔
DANN_FT_B_TEMPLATE      = "./output_dann_finetune_B_v6/run_{run_i}/checkpoint-XXX" # 請替換為你實際存下的 best checkpoint 資料夾名稱，或確保存為 best_model

MODEL_NAME = "facebook/wav2vec2-base"
TOTAL_RUNS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🖥️ 使用裝置: {DEVICE}")

# ============================================================
#  模型結構定義 (用於載入權重)
# ============================================================

class SharedEncoder(nn.Module):
    """Static DANN 的 Shared Encoder"""
    def __init__(self, input_dim=768, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.3)
        )
    def forward(self, x):
        return self.encoder(x)

class Wav2Vec2DANNFinetune(Wav2Vec2PreTrainedModel):
    """DANN-FT 的架構，用於提取 128D 特徵"""
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.shared_encoder = nn.Sequential(
            nn.Linear(config.hidden_size, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3)
        )
    def get_embedding(self, input_values):
        outputs = self.wav2vec2(input_values)
        return self.shared_encoder(torch.mean(outputs[0], dim=1))

class Wav2Vec2ClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
    def forward(self, x): return self.out_proj(self.dropout(torch.tanh(self.dense(self.dropout(x)))))

class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    """Huang Baseline 的架構"""
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)
    def get_embedding(self, input_values):
        return torch.mean(self.wav2vec2(input_values).last_hidden_state, dim=1)

# ============================================================
#  特徵提取工具
# ============================================================

def extract_speaker_id(filepath): return os.path.basename(str(filepath)).split('_')[0]

def load_weights(model, model_dir):
    """安全地載入 pytorch_model.bin 或 model.safetensors"""
    safe_path = os.path.join(model_dir, "model.safetensors")
    bin_path = os.path.join(model_dir, "pytorch_model.bin")
    if os.path.exists(safe_path):
        safetensors.torch.load_model(model, safe_path, strict=False)
    elif os.path.exists(bin_path):
        model.load_state_dict(torch.load(bin_path, map_location=DEVICE), strict=False)
    else:
        raise FileNotFoundError(f"找不到權重檔於 {model_dir}")
    return model

def extract_features(csv_path, processor, model_extractor, desc):
    """統一的特徵抽取迴圈"""
    df = pd.read_csv(csv_path)
    feats, spks = [], []
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), leave=False, desc=desc):
            wav_path = os.path.join(AUDIO_ROOT, str(row['path']))
            try:
                waveform, sr = torchaudio.load(wav_path)
                if sr != 16000: waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
                if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
                emb = model_extractor(inputs['input_values'].to(DEVICE)).cpu().numpy()
                feats.append(emb.squeeze())
                spks.append(extract_speaker_id(row['path']))
            except: continue
    return np.array(feats), np.array(spks)

def run_speaker_probe(X_train, spk_train, X_test, spk_test):
    clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    clf.fit(X_train, spk_train)
    return accuracy_score(spk_test, clf.predict(X_test))

# ============================================================
#  主程式
# ============================================================

if __name__ == "__main__":
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    w2v_frozen = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(DEVICE)
    w2v_frozen.eval()

    print("\n📦 抽取 [1. Linear Probing] 原始特徵（只需一次）...")
    def raw_extractor(x): return torch.mean(w2v_frozen(x).last_hidden_state, dim=1)
    lp_train_X, lp_train_spk = extract_features(CSV_B_TRAIN, processor, raw_extractor, "Linear Train")
    lp_test_X,  lp_test_spk  = extract_features(CSV_B_TEST,  processor, raw_extractor, "Linear Test")

    results = {"Linear_B": [], "Huang_B": [], "Static_DANN_B": [], "DANN_FT_B": []}

    for run_i in range(1, TOTAL_RUNS + 1):
        print(f"\n{'='*60}\n🎬 Run {run_i} / {TOTAL_RUNS}\n{'='*60}")

        # 1. Linear Probing
        acc = run_speaker_probe(lp_train_X, lp_train_spk, lp_test_X, lp_test_spk)
        results["Linear_B"].append(acc)
        print(f"[Linear B] Speaker Acc: {acc:.4f}")

        # 2. Huang Baseline
        huang_path = HUANG_B_MODEL_TEMPLATE.format(run_i=run_i)
        if os.path.exists(huang_path):
            config = AutoConfig.from_pretrained(huang_path)
            model = Wav2Vec2ForSpeechClassification(config).to(DEVICE)
            model = load_weights(model, huang_path).eval()
            X_tr, spk_tr = extract_features(CSV_B_TRAIN, processor, model.get_embedding, "[Huang B] Train")
            X_te, spk_te = extract_features(CSV_B_TEST,  processor, model.get_embedding, "[Huang B] Test")
            acc = run_speaker_probe(X_tr, spk_tr, X_te, spk_te)
            results["Huang_B"].append(acc)
            print(f"[Huang B] Speaker Acc: {acc:.4f}")
        else:
            print(f"⚠️ [Huang B] 找不到 {huang_path}")

        # 3. Static DANN
        static_dann_path = STATIC_DANN_B_TEMPLATE.format(run_i=run_i)
        if os.path.exists(static_dann_path):
            shared_encoder = SharedEncoder().to(DEVICE)
            shared_encoder.load_state_dict(torch.load(static_dann_path, map_location=DEVICE))
            shared_encoder.eval()
            def static_dann_extractor(x): return shared_encoder(torch.mean(w2v_frozen(x).last_hidden_state, dim=1))
            
            X_tr, spk_tr = extract_features(CSV_B_TRAIN, processor, static_dann_extractor, "[Static DANN] Train")
            X_te, spk_te = extract_features(CSV_B_TEST,  processor, static_dann_extractor, "[Static DANN] Test")
            acc = run_speaker_probe(X_tr, spk_tr, X_te, spk_te)
            results["Static_DANN_B"].append(acc)
            print(f"[Static DANN B] Speaker Acc: {acc:.4f}")
        else:
            print(f"⚠️ [Static DANN B] 找不到 {static_dann_path}")

        # 4. DANN-FT B
        # 這裡改為自動搜尋路徑邏輯
        run_dir = f"./output_dann_finetune_B_v6/run_{run_i}"
        dann_ft_path = None
        
        if os.path.exists(run_dir):
            # 尋找該 run_i 資料夾下所有的 checkpoint 資料夾
            checkpoints = [d for d in os.listdir(run_dir) if d.startswith("checkpoint-")]
            if checkpoints:
                # 排序後取最大的步數，通常 Trainer 最後會停在最佳或最後的 checkpoint
                # 或者如果你有特定的 best checkpoint 資料夾，請手動指定
                checkpoints.sort(key=lambda x: int(x.split('-')[1]))
                dann_ft_path = os.path.join(run_dir, checkpoints[-1])

        if dann_ft_path and os.path.exists(dann_ft_path):
            print(f"\n[DANN-FT B] 自動偵測路徑: {dann_ft_path}")
            config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)
            model = Wav2Vec2DANNFinetune(config).to(DEVICE)
            model = load_weights(model, dann_ft_path).eval()
            
            X_tr, spk_tr = extract_features(CSV_B_TRAIN, processor, model.get_embedding, "[DANN-FT] Train")
            X_te, spk_te = extract_features(CSV_B_TEST,  processor, model.get_embedding, "[DANN-FT] Test")
            acc = run_speaker_probe(X_tr, spk_tr, X_te, spk_te)
            results["DANN_FT_B"].append(acc)
            print(f"  → Speaker Acc: {acc:.4f}")
        else:
            print(f"⚠️ [DANN-FT B] 在 {run_dir} 找不到任何 checkpoint 資料夾")

    # 彙總輸出
    print(f"\n{'='*60}\n📊 Scenario B - Speaker Probe 彙總結果\n{'='*60}")
    summary_rows = []
    for name, accs in results.items():
        if len(accs) > 0:
            arr = np.array(accs)
            print(f"{name:<15} {len(accs)} runs | {arr.mean():.4f} ± {arr.std():.4f}")
            summary_rows.append({"model": name, "runs": len(accs), "mean": round(arr.mean(),4), "std": round(arr.std(),4)})
        else:
            print(f"{name:<15} 0 runs | N/A")
            
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv("speaker_probe_summary_B.csv", index=False)
        print(f"\n✅ 彙總已儲存至 speaker_probe_summary_B.csv")
