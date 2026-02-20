"""
Speaker Probe — 六組模型 Speaker Accuracy 統一評估
====================================================
邏輯：對每組模型抽出 embedding，用 Logistic Regression 預測 Speaker ID。
      Speaker Accuracy 越低 → 模型越成功去除 speaker 資訊。

六組設定：
  1. Huang A   → 微調後 Wav2Vec2，768 維 mean pooling
  2. Huang B   → 微調後 Wav2Vec2，768 維 mean pooling
  3. Linear A  → 原始凍結 Wav2Vec2，768 維（不需要模型檔）
  4. Linear B  → 原始凍結 Wav2Vec2，768 維（不需要模型檔）
  5. DANN A    → 凍結 Wav2Vec2 → shared_encoder，128 維
  6. DANN B    → 凍結 Wav2Vec2 → shared_encoder，128 維

每組跑 5 次（對應 5 個 run），最後輸出平均 ± 標準差。
若某個 run 的模型檔不存在則自動跳過。

【路徑說明】
  執行前請將下方 ======= 路徑設定區 ======= 內的路徑改成你的實際路徑。
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model, AutoConfig
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ============================================================
#  ======= 路徑設定區（執行前請修改） =======
# ============================================================

AUDIO_ROOT = ""   # CSV 內已是絕對路徑，保持空字串即可

# 資料 CSV
CSV_A_TRAIN = "./experiment_sisman_scientific/scenario_A_screening/train.csv"
CSV_A_TEST  = "./experiment_sisman_scientific/scenario_A_screening/test.csv"
CSV_B_TRAIN = "./experiment_sisman_scientific/scenario_B_monitoring/train.csv"
CSV_B_TEST  = "./experiment_sisman_scientific/scenario_B_monitoring/test.csv"

# Huang best_model 路徑樣板（{run_i} 會被替換成 1~5）
HUANG_A_MODEL_TEMPLATE = "./output_scenario_A_v2/run_{run_i}/best_model"
HUANG_B_MODEL_TEMPLATE = "./output_scenario_B_v2/run_{run_i}/best_model"

# DANN shared_encoder 路徑樣板（{run_i} 會被替換成 1~5）
DANN_A_ENCODER_TEMPLATE = "./dann_A_shared_encoder_run_{run_i}.pth"
DANN_B_ENCODER_TEMPLATE = "./dann_B_shared_encoder_run_{run_i}.pth"

MODEL_NAME = "facebook/wav2vec2-base"
TOTAL_RUNS = 5

# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ 使用裝置: {DEVICE}")


# ============================================================
#  模型結構定義
# ============================================================

class SharedEncoder(nn.Module):
    """對應 DANN 的 shared_encoder，輸出 128 維"""
    def __init__(self, input_dim=768, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    def forward(self, x):
        return self.encoder(x)


class Wav2Vec2ClassificationHead(nn.Module):
    """對應 build_model.py 的 head，結構需完整才能正確載入 checkpoint"""
    def __init__(self, config):
        super().__init__()
        self.dense    = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout  = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Wav2Vec2ForSpeechClassification(nn.Module):
    """對應 build_model.py 的完整模型，probe 只用 get_embedding()"""
    def __init__(self, config):
        super().__init__()
        self.wav2vec2   = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)

    def get_embedding(self, input_values):
        """回傳 mean pooling 後的 768 維特徵"""
        hidden_states = self.wav2vec2(input_values).last_hidden_state
        return torch.mean(hidden_states, dim=1)


# ============================================================
#  工具函式
# ============================================================

def extract_speaker_id(filepath):
    """從路徑取出 speaker ID（檔名底線前的部分，例如 300_xxx.wav → 300）"""
    return os.path.basename(str(filepath)).split('_')[0]


def load_raw_w2v_embeddings(csv_path, processor, w2v_model):
    """
    [Linear Probing 用]
    直接用完全凍結的 Wav2Vec2 抽 768 維特徵。
    任何時候重抽結果都相同，不需要模型檔。
    回傳：X (N, 768), speaker_ids (N,)
    """
    df = pd.read_csv(csv_path)
    feats, spks = [], []

    w2v_model.eval()
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), leave=False,
                           desc=f"  Raw W2V ← {os.path.basename(csv_path)}"):
            wav_path = os.path.join(AUDIO_ROOT, str(row['path']))
            try:
                waveform, sr = torchaudio.load(wav_path)
                if sr != 16000:
                    waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                inputs = processor(waveform.squeeze().numpy(),
                                   sampling_rate=16000,
                                   return_tensors="pt", padding=True)
                emb = w2v_model(**{k: v.to(DEVICE) for k, v in inputs.items()}
                                ).last_hidden_state.mean(dim=1).cpu().numpy()
                feats.append(emb.squeeze())
                spks.append(extract_speaker_id(row['path']))
            except:
                continue

    return np.array(feats), np.array(spks)


def load_huang_embeddings(csv_path, processor, model_dir):
    """
    [Huang 用]
    載入微調後的 best_model，抽 768 維 mean pooling 特徵。
    回傳：X (N, 768), speaker_ids (N,)
    """
    config = AutoConfig.from_pretrained(model_dir)
    model  = Wav2Vec2ForSpeechClassification(config).to(DEVICE)
    state_dict_path = os.path.join(model_dir, "pytorch_model.bin")
    model.load_state_dict(
        torch.load(state_dict_path, map_location=DEVICE), strict=False
    )
    model.eval()

    df = pd.read_csv(csv_path)
    feats, spks = [], []

    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), leave=False,
                           desc=f"  Huang ← {os.path.basename(csv_path)}"):
            wav_path = os.path.join(AUDIO_ROOT, str(row['path']))
            try:
                waveform, sr = torchaudio.load(wav_path)
                if sr != 16000:
                    waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                inputs = processor(waveform.squeeze().numpy(),
                                   sampling_rate=16000,
                                   return_tensors="pt", padding=True)
                emb = model.get_embedding(
                    inputs['input_values'].to(DEVICE)
                ).cpu().numpy()
                feats.append(emb.squeeze())
                spks.append(extract_speaker_id(row['path']))
            except:
                continue

    return np.array(feats), np.array(spks)


def load_dann_embeddings(csv_path, processor, w2v_model, encoder_path):
    """
    [DANN 用]
    凍結 Wav2Vec2 先抽 768 維，再過 shared_encoder 壓成 128 維。
    回傳：X (N, 128), speaker_ids (N,)
    """
    shared_encoder = SharedEncoder().to(DEVICE)
    shared_encoder.load_state_dict(
        torch.load(encoder_path, map_location=DEVICE)
    )
    shared_encoder.eval()

    df = pd.read_csv(csv_path)
    feats, spks = [], []

    w2v_model.eval()
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), leave=False,
                           desc=f"  DANN ← {os.path.basename(csv_path)}"):
            wav_path = os.path.join(AUDIO_ROOT, str(row['path']))
            try:
                waveform, sr = torchaudio.load(wav_path)
                if sr != 16000:
                    waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                inputs = processor(waveform.squeeze().numpy(),
                                   sampling_rate=16000,
                                   return_tensors="pt", padding=True)
                raw_emb = w2v_model(**{k: v.to(DEVICE) for k, v in inputs.items()}
                                    ).last_hidden_state.mean(dim=1)
                emb = shared_encoder(raw_emb).cpu().numpy()
                feats.append(emb.squeeze())
                spks.append(extract_speaker_id(row['path']))
            except:
                continue

    return np.array(feats), np.array(spks)


def run_speaker_probe(X_train, spk_train, X_test, spk_test):
    """
    在 X_train 上訓練 Logistic Regression 預測 speaker，
    在 X_test 上評估，回傳 speaker accuracy。
    """
    clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    clf.fit(X_train, spk_train)
    return accuracy_score(spk_test, clf.predict(X_test))


# ============================================================
#  主程式
# ============================================================

if __name__ == "__main__":

    print("\n🧠 載入凍結 Wav2Vec2（Linear Probing 和 DANN 共用）...")
    processor  = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    w2v_frozen = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(DEVICE)
    w2v_frozen.eval()

    # ----------------------------------------------------------
    # Linear Probing 特徵只需抽一次（凍結模型，每次結果相同）
    # ----------------------------------------------------------
    print("\n📦 抽取 Linear Probing 原始特徵（只需一次）...")
    lp_A_train_X, lp_A_train_spk = load_raw_w2v_embeddings(CSV_A_TRAIN, processor, w2v_frozen)
    lp_A_test_X,  lp_A_test_spk  = load_raw_w2v_embeddings(CSV_A_TEST,  processor, w2v_frozen)
    lp_B_train_X, lp_B_train_spk = load_raw_w2v_embeddings(CSV_B_TRAIN, processor, w2v_frozen)
    lp_B_test_X,  lp_B_test_spk  = load_raw_w2v_embeddings(CSV_B_TEST,  processor, w2v_frozen)

    # ----------------------------------------------------------
    # 六組結果收集容器
    # ----------------------------------------------------------
    results = {
        "Huang_A":  [], "Huang_B":  [],
        "Linear_A": [], "Linear_B": [],
        "DANN_A":   [], "DANN_B":   [],
    }

    # ----------------------------------------------------------
    # 逐 run 評估
    # ----------------------------------------------------------
    for run_i in range(1, TOTAL_RUNS + 1):
        print(f"\n{'='*60}")
        print(f"🎬 Run {run_i} / {TOTAL_RUNS}")
        print(f"{'='*60}")

        # ── 1. Huang A ──────────────────────────────────────
        huang_A_path = HUANG_A_MODEL_TEMPLATE.format(run_i=run_i)
        if os.path.exists(huang_A_path):
            print(f"\n[Huang A] 載入: {huang_A_path}")
            X_tr, spk_tr = load_huang_embeddings(CSV_A_TRAIN, processor, huang_A_path)
            X_te, spk_te = load_huang_embeddings(CSV_A_TEST,  processor, huang_A_path)
            acc = run_speaker_probe(X_tr, spk_tr, X_te, spk_te)
            results["Huang_A"].append(acc)
            print(f"  → Speaker Acc: {acc:.4f}")
        else:
            print(f"  ⚠️  Huang A Run {run_i} 不存在，跳過 ({huang_A_path})")

        # ── 2. Huang B ──────────────────────────────────────
        huang_B_path = HUANG_B_MODEL_TEMPLATE.format(run_i=run_i)
        if os.path.exists(huang_B_path):
            print(f"\n[Huang B] 載入: {huang_B_path}")
            X_tr, spk_tr = load_huang_embeddings(CSV_B_TRAIN, processor, huang_B_path)
            X_te, spk_te = load_huang_embeddings(CSV_B_TEST,  processor, huang_B_path)
            acc = run_speaker_probe(X_tr, spk_tr, X_te, spk_te)
            results["Huang_B"].append(acc)
            print(f"  → Speaker Acc: {acc:.4f}")
        else:
            print(f"  ⚠️  Huang B Run {run_i} 不存在，跳過 ({huang_B_path})")

        # ── 3. Linear Probing A（特徵已抽好，直接 probe）──
        print(f"\n[Linear A] 凍結 Wav2Vec2 特徵，直接 probe")
        acc = run_speaker_probe(lp_A_train_X, lp_A_train_spk,
                                lp_A_test_X,  lp_A_test_spk)
        results["Linear_A"].append(acc)
        print(f"  → Speaker Acc: {acc:.4f}")

        # ── 4. Linear Probing B（特徵已抽好，直接 probe）──
        print(f"\n[Linear B] 凍結 Wav2Vec2 特徵，直接 probe")
        acc = run_speaker_probe(lp_B_train_X, lp_B_train_spk,
                                lp_B_test_X,  lp_B_test_spk)
        results["Linear_B"].append(acc)
        print(f"  → Speaker Acc: {acc:.4f}")

        # ── 5. DANN A ───────────────────────────────────────
        dann_A_path = DANN_A_ENCODER_TEMPLATE.format(run_i=run_i)
        if os.path.exists(dann_A_path):
            print(f"\n[DANN A] 載入: {dann_A_path}")
            X_tr, spk_tr = load_dann_embeddings(CSV_A_TRAIN, processor, w2v_frozen, dann_A_path)
            X_te, spk_te = load_dann_embeddings(CSV_A_TEST,  processor, w2v_frozen, dann_A_path)
            acc = run_speaker_probe(X_tr, spk_tr, X_te, spk_te)
            results["DANN_A"].append(acc)
            print(f"  → Speaker Acc: {acc:.4f}")
        else:
            print(f"  ⚠️  DANN A Run {run_i} 不存在，跳過 ({dann_A_path})")

        # ── 6. DANN B ───────────────────────────────────────
        dann_B_path = DANN_B_ENCODER_TEMPLATE.format(run_i=run_i)
        if os.path.exists(dann_B_path):
            print(f"\n[DANN B] 載入: {dann_B_path}")
            X_tr, spk_tr = load_dann_embeddings(CSV_B_TRAIN, processor, w2v_frozen, dann_B_path)
            X_te, spk_te = load_dann_embeddings(CSV_B_TEST,  processor, w2v_frozen, dann_B_path)
            acc = run_speaker_probe(X_tr, spk_tr, X_te, spk_te)
            results["DANN_B"].append(acc)
            print(f"  → Speaker Acc: {acc:.4f}")
        else:
            print(f"  ⚠️  DANN B Run {run_i} 不存在，跳過 ({dann_B_path})")

    # ----------------------------------------------------------
    # 彙總輸出
    # ----------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"📊 Speaker Probe 彙總結果")
    print(f"{'='*60}")
    print(f"{'模型':<12} {'有效 runs':<10} {'平均 Spk Acc':<16} 標準差")
    print(f"{'─'*55}")

    summary_rows = []
    for name, accs in results.items():
        if len(accs) == 0:
            print(f"{name:<12} {'0':<10} {'N/A':<16} N/A")
        else:
            arr  = np.array(accs)
            mean = arr.mean()
            std  = arr.std()
            print(f"{name:<12} {len(accs):<10} {mean:.4f}{'':>10} ± {std:.4f}")
            summary_rows.append({
                "model": name, "valid_runs": len(accs),
                "spk_acc_mean": round(mean, 4),
                "spk_acc_std":  round(std,  4),
            })

    if summary_rows:
        out_path = "speaker_probe_summary.csv"
        pd.DataFrame(summary_rows).to_csv(out_path, index=False)
        print(f"\n✅ 彙總已儲存至 {out_path}")
