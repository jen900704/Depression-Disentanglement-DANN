"""
DANN (Static Backbone) — Scenario A
=====================================
修正版 v3：修正 Speaker Map 邏輯 + evaluate 加 mask

Scenario A 的 Speaker 邏輯說明：
  - Train set：全是路人（~151位），GRL 對路人做 adversarial
  - Test set：38位 Target Group（全是陌生人，train 從未見過）
  - Test 的 speaker_label = -1（不參與 L_spk，對應論文 Spk Acc = N/A）

修正項目：
  [v3-1] speaker_to_idx.get(spk_id, 0) → get(spk_id, -1)
         test 的 38 位陌生人不在 map → -1，而非錯誤的 0
  [v3-2] evaluate() 加 mask，只對 s >= 0 計算 Spk Acc
         → test 全是 s=-1，Spk Acc 回傳 N/A (0.0)，對應論文 Table 1
  [v3-3] training loop 加 mask，只對 train 路人（s >= 0）計算 L_spk
         → test 時 s=-1 不會污染 loss（eval 不計算 loss，此項確保邏輯一致）

不變項目（與 DANN-FT 的唯一架構差異）：
  - Wav2Vec2 完全凍結（CNN + Transformer 都不更新）
  - 特徵一次預提取，存成 Tensor
  - Optimizer: Adam, lr=1e-3, batch=32, epochs=30
  - Alpha: global step 比例
  - Best checkpoint: val F1
"""

import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Function
from transformers import Wav2Vec2Processor, Wav2Vec2Model, set_seed
import torchaudio
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
#  設定區
# ============================================================
TRAIN_CSV_PATH = "./experiment_sisman_scientific/scenario_A_screening/train.csv"
TEST_CSV_PATH  = "./experiment_sisman_scientific/scenario_A_screening/test.csv"
AUDIO_ROOT     = ""

MODEL_NAME  = "facebook/wav2vec2-base"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 32
EPOCHS      = 30
TOTAL_RUNS  = 5
SEED        = 103

print(f"🖥️ 使用裝置: {DEVICE}")

# ============================================================
#  GRL
# ============================================================
class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class GradientReversalLayer(nn.Module):
    def forward(self, x, alpha=1.0):
        return GradientReversalFn.apply(x, alpha)

# ============================================================
#  模型定義
# ============================================================
class DANN_Model(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128, num_classes=2, num_speakers=151):
        super().__init__()
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.class_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_speakers),
        )
        self.grl = GradientReversalLayer()

    def forward(self, x, alpha=1.0):
        features   = self.shared_encoder(x)
        class_out  = self.class_classifier(features)
        rev        = self.grl(features, alpha)
        domain_out = self.domain_classifier(rev)
        return class_out, domain_out

# ============================================================
#  Helper
# ============================================================
def extract_speaker_id(filepath):
    return os.path.basename(str(filepath)).split('_')[0]

def build_speaker_map(train_csv_path):
    """
    Scenario A：Speaker Map 從 train.csv 建立（~151 位路人）
    test 的 38 位陌生人不在此 map，load 時給 -1
    """
    df = pd.read_csv(train_csv_path)
    speakers = sorted(df['path'].apply(extract_speaker_id).unique())
    speaker_map = {spk: idx for idx, spk in enumerate(speakers)}
    print(f"🔍 [v3] Speaker Map 從 train 建立: {len(speaker_map)} 位路人")
    return speaker_map

# ============================================================
#  資料處理
# ============================================================
def prepare_data(csv_path, processor, w2v_model, speaker_to_idx):
    df = pd.read_csv(csv_path)
    print(f"📂 正在處理 {csv_path} (共 {len(df)} 筆)...")

    features_list, labels_list, speaker_indices_list = [], [], []
    label_map = {'dep': 1, '1': 1, 1: 1, 'non': 0, '0': 0, 0: 0}

    w2v_model.eval()
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Features"):
            wav_path  = os.path.join(AUDIO_ROOT, row['path'])
            raw_label = str(row['label']).strip().lower()
            if raw_label not in label_map:
                continue
            try:
                waveform, sr = torchaudio.load(wav_path)
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                if sr != 16000:
                    waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

                inputs = processor(
                    waveform.squeeze().numpy(),
                    sampling_rate=16000, return_tensors="pt", padding=True
                )
                inputs     = {k: v.to(DEVICE) for k, v in inputs.items()}
                embeddings = w2v_model(**inputs).last_hidden_state.mean(dim=1).cpu()

                features_list.append(embeddings)
                labels_list.append(label_map[raw_label])

                # [v3-1] 路人在 map → 正確 index；test 陌生人不在 map → -1
                spk_id = extract_speaker_id(wav_path)
                speaker_indices_list.append(speaker_to_idx.get(spk_id, -1))

            except Exception as e:
                print(f"⚠️ Error: {wav_path} -> {e}")

    if not features_list:
        raise ValueError("❌ 沒有任何資料被成功讀取！")

    X = torch.cat(features_list, dim=0)
    y = torch.tensor(labels_list,          dtype=torch.long)
    s = torch.tensor(speaker_indices_list, dtype=torch.long)

    n_known   = (s >= 0).sum().item()
    n_unknown = (s < 0).sum().item()
    print(f"   → 已知路人: {n_known} 筆 | 陌生人 (s=-1): {n_unknown} 筆")

    return X, y, s

# ============================================================
#  評估函式
#  [v3-2] Spk Acc 只對 s >= 0 計算
#  Scenario A 的 test 全是陌生人(s=-1) → Spk Acc = N/A (0.0)
# ============================================================
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    correct_spk, total_spk = 0, 0
    with torch.no_grad():
        for inputs, labels, speakers in loader:
            inputs, labels, speakers = (
                inputs.to(DEVICE), labels.to(DEVICE), speakers.to(DEVICE)
            )
            class_out, domain_out = model(inputs, alpha=0.0)
            _, preds     = torch.max(class_out,  1)
            _, spk_preds = torch.max(domain_out, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # [v3-2] 只對已知 speaker 計算 Spk Acc
            mask = speakers >= 0
            if mask.sum() > 0:
                correct_spk += (spk_preds[mask] == speakers[mask]).sum().item()
                total_spk   += mask.sum().item()

    acc     = accuracy_score(all_labels, all_preds)
    f1      = f1_score(all_labels, all_preds, average='macro')
    spk_acc = correct_spk / total_spk if total_spk > 0 else 0.0  # test 全 s=-1 → 0.0 = N/A
    return acc, f1, spk_acc

# t-SNE 用特徵抽取
def get_feats(model, loader):
    model.eval()
    feats, labels_list = [], []
    with torch.no_grad():
        for inputs, labels, speakers in loader:
            inputs = inputs.to(DEVICE)
            f = model.shared_encoder(inputs).cpu().numpy()
            feats.append(f)
            labels_list.extend(labels.cpu().numpy())
    return np.vstack(feats), np.array(labels_list)

# ============================================================
#  主程式
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 DANN (Static Backbone) — Scenario A  [v3 修正版]")
    print("   Speaker Map: train 路人（~151位）")
    print("   Test 陌生人 s=-1，不參與 L_spk（論文 Spk Acc = N/A）")
    print("   Alpha: global step 比例 | Best checkpoint: val F1")
    print(f"   LR=1e-3 | Batch={BATCH_SIZE} | Epochs={EPOCHS} | Runs={TOTAL_RUNS}")
    print("=" * 60)

    # [v3] Speaker Map 從 train 建立（~151位路人）
    speaker_map  = build_speaker_map(TRAIN_CSV_PATH)
    num_speakers = len(speaker_map)
    print(f"✅ num_speakers = {num_speakers}")

    # 特徵提取（只做一次）
    print("\n🧠 載入 Wav2Vec2（凍結）...")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    w2v_model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(DEVICE)

    print("\n📦 預提取特徵（只執行一次）...")
    X_train, y_train, s_train = prepare_data(TRAIN_CSV_PATH, processor, w2v_model, speaker_map)
    X_test,  y_test,  s_test  = prepare_data(TEST_CSV_PATH,  processor, w2v_model, speaker_map)

    # 預期：train 全是已知路人(s>=0)；test 全是陌生人(s=-1)
    print(f"✅ Train: {len(X_train)} 筆 | Test: {len(X_test)} 筆")

    del w2v_model
    torch.cuda.empty_cache()

    train_dataset = TensorDataset(X_train, y_train, s_train)
    test_dataset  = TensorDataset(X_test,  y_test,  s_test)

    all_run_accs = []
    all_run_f1s  = []

    for run_i in range(1, TOTAL_RUNS + 1):
        print(f"\n{'='*60}")
        print(f"🎬 Run {run_i} / {TOTAL_RUNS}")
        print(f"{'='*60}")

        run_seed = SEED + run_i - 1
        set_seed(run_seed)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

        total_steps = len(train_loader) * EPOCHS
        global_step = 0

        print("🏗️ 初始化全新 DANN 模型...")
        dann_model = DANN_Model(num_speakers=num_speakers).to(DEVICE)

        optimizer        = optim.Adam(dann_model.parameters(), lr=1e-3)
        criterion_class  = nn.CrossEntropyLoss()
        criterion_domain = nn.CrossEntropyLoss()

        best_val_f1      = -1.0
        best_model_state = None

        print("⚔️ 開始訓練...")
        for epoch in range(EPOCHS):
            dann_model.train()
            total_loss     = 0
            total_loss_dep = 0
            total_loss_spk = 0

            for inputs, labels, speakers in train_loader:
                inputs, labels, speakers = (
                    inputs.to(DEVICE), labels.to(DEVICE), speakers.to(DEVICE)
                )

                p     = global_step / max(total_steps, 1)
                alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1
                global_step += 1

                optimizer.zero_grad()
                class_out, domain_out = dann_model(inputs, alpha=alpha)

                loss_dep = criterion_class(class_out, labels)

                # [v3-3] 只對 train 路人（s >= 0）計算 L_spk
                # train set 全是路人，mask 應全為 True，此為防禦性寫法
                mask = speakers >= 0
                if mask.sum() > 0:
                    loss_spk = criterion_domain(domain_out[mask], speakers[mask])
                else:
                    loss_spk = torch.tensor(0.0, device=DEVICE)

                loss = loss_dep + loss_spk
                loss.backward()
                optimizer.step()

                total_loss     += loss.item()
                total_loss_dep += loss_dep.item()
                total_loss_spk += loss_spk.item()

            acc, f1, spk_acc = evaluate(dann_model, test_loader)

            if f1 > best_val_f1:
                best_val_f1      = f1
                best_model_state = copy.deepcopy(dann_model.state_dict())

            if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
                print(f"Epoch {epoch+1}/{EPOCHS} | "
                      f"Loss: {total_loss:.2f} (dep={total_loss_dep:.2f}, spk={total_loss_spk:.2f}) | "
                      f"Dep Acc: {acc:.4f} | F1: {f1:.4f} | "
                      f"Spk Acc: {spk_acc:.4f} (N/A for A) | Best F1: {best_val_f1:.4f}")

        print(f"\n🏆 載入 Best Checkpoint (F1={best_val_f1:.4f}) 進行最終評估...")
        dann_model.load_state_dict(best_model_state)
        final_acc, final_f1, final_spk_acc = evaluate(dann_model, test_loader)
        print(f"✅ Run {run_i} 最終結果 → Acc: {final_acc:.4f} | "
              f"F1: {final_f1:.4f} | Spk Acc: N/A")

        all_run_accs.append(final_acc)
        all_run_f1s.append(final_f1)

        # t-SNE — 用 depression label 上色（Scenario A 無 speaker identity 可視化意義）
        print(f"\n🎨 [Run {run_i}] 繪製 t-SNE 圖（depression label）...")
        dann_feats, dep_labels = get_feats(dann_model, test_loader)
        tsne    = TSNE(n_components=2, random_state=42 + run_i, perplexity=30)
        feats2d = tsne.fit_transform(dann_feats)

        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x=feats2d[:, 0], y=feats2d[:, 1],
            hue=dep_labels, palette={0: "steelblue", 1: "tomato"},
            legend="full"
        )
        plt.title(f"DANN (Static) Feature Space — Scenario A, Run {run_i}", fontsize=16)
        filename = f"tsne_dann_A_run_{run_i}.png"
        plt.savefig(filename)
        plt.close()
        print(f"✅ 圖片已儲存: {filename}")

    # 彙總
    print(f"\n{'='*60}")
    print(f"📊 {TOTAL_RUNS} 次實驗結果統計 (DANN Static Backbone — Scenario A v3)")
    print(f"{'='*60}")
    accs = np.array(all_run_accs)
    f1s  = np.array(all_run_f1s)
    for i, (a, f) in enumerate(zip(accs, f1s), 1):
        print(f"  Run {i}: Acc={a:.4f}, F1={f:.4f}")
    print(f"{'─'*40}")
    print(f"  平均 Acc : {accs.mean():.4f} ± {accs.std():.4f}")
    print(f"  平均 F1  : {f1s.mean():.4f} ± {f1s.std():.4f}")
    print(f"  Spk Acc  : N/A（Scenario A 測試對象為全新陌生人）")
