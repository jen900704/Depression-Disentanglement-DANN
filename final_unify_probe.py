"""
Speaker Probe — 八組模型 Speaker Accuracy 統一評估
====================================================
【設計邏輯】
  Speaker Probe 的目的是衡量「模型的 latent representation 裡
  殘留了多少 speaker identity 資訊」，因此 probe 的對象
  應該嚴格對應論文的兩種情境：

  ── Scenario A（Strict Screening，unseen speakers）──────────
  Test set = 38 位 target speakers 的 Current Data（714 segs）
  Train set = 151 位 control speakers（Base + Filler）
  模型完全未見過 target speakers。
  → Probe classifier 在 train（151 人）的 embedding 上訓練，
    在 test（38 位 unseen）上評估。
  → Spk Acc 理論上 ≈ 0%，語義為「probe 無法辨識 unseen speakers」。
    注意：這不代表模型沒有 encode speaker info，
    而是 probe classifier 根本沒見過這些 class。

  ── Scenario B（Longitudinal Monitoring，seen speakers）─────
  Train set = Base（151 人）+ Historical（38 位 target 的歷史錄音，714 segs）
  Test set  = 同 38 位 target 的 Current Data（714 segs）
  → 正確 probe 應只用 Historical 那 714 筆（target 38 人）
    來訓練 probe classifier，然後在 test（同 38 人）評估。
  → 若用全 5117 筆 train，容量被 151 位 control 稀釋，Spk Acc 被低估。
  → 程式上：從 train_B CSV 中篩出屬於 target speakers 的樣本。

八組設定：
  1. Huang A      → 微調後 Wav2Vec2，768 維 mean pooling
  2. Huang B      → 微調後 Wav2Vec2，768 維 mean pooling
  3. Linear A     → 原始凍結 Wav2Vec2，768 維（不需要模型檔）
  4. Linear B     → 原始凍結 Wav2Vec2，768 維（不需要模型檔）
  5. DANN A       → 凍結 Wav2Vec2 → shared_encoder，128 維（外部 .pth）
  6. DANN B       → 凍結 Wav2Vec2 → shared_encoder，128 維（外部 .pth）
  7. DANN-FT A    → 完整微調 Wav2Vec2DANNFinetune (Scenario A)，128 維
  8. DANN-FT B    → 完整微調 Wav2Vec2DANNFinetune (Scenario B)，128 維

每組跑 5 次（對應 5 個 run），最後輸出平均 ± 標準差。
若某個 run 的模型檔不存在則自動跳過（例如 DANN-FT 還在訓練中）。
支援 pytorch_model.bin 和 model.safetensors 兩種權重格式。
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
from dataclasses import dataclass
from typing import Optional, Set
from tqdm import tqdm
from torch.autograd import Function
from transformers import (
    Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2Config,
    Wav2Vec2PreTrainedModel, AutoConfig
)
from transformers.file_utils import ModelOutput
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ============================================================
#  ======= 路徑設定區（執行前請修改） =======
# ============================================================

AUDIO_ROOT = ""   # CSV 內已是絕對路徑，保持空字串即可

CSV_A_TRAIN = "./experiment_sisman_scientific/scenario_A_screening/train.csv"
CSV_A_TEST  = "./experiment_sisman_scientific/scenario_A_screening/test.csv"
CSV_B_TRAIN = "./experiment_sisman_scientific/scenario_B_monitoring/train.csv"
CSV_B_TEST  = "./experiment_sisman_scientific/scenario_B_monitoring/test.csv"

HUANG_A_MODEL_TEMPLATE  = "./output_scenario_A_v2/run_{run_i}/best_model"
HUANG_B_MODEL_TEMPLATE  = "./output_scenario_B_v2/run_{run_i}/best_model"

DANN_A_ENCODER_TEMPLATE = "./dann_A_shared_encoder_run_{run_i}.pth"
DANN_B_ENCODER_TEMPLATE = "./dann_B_shared_encoder_run_{run_i}.pth"

DANN_FT_A_DIR_TEMPLATE  = "./output_dann_finetune_A_v6/run_{run_i}"
DANN_FT_B_DIR_TEMPLATE  = "./output_dann_finetune_B_v6/run_{run_i}"

MODEL_NAME = "facebook/wav2vec2-base"
TOTAL_RUNS = 5

# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ 使用裝置: {DEVICE}")


# ============================================================
#  模型結構定義
# ============================================================

class SharedEncoder(nn.Module):
    """對應舊版 DANN 的 shared_encoder（外部 .pth），輸出 128 維"""
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
    """Huang baseline 完整模型，probe 只用 get_embedding()"""
    def __init__(self, config):
        super().__init__()
        self.wav2vec2   = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)

    def get_embedding(self, input_values):
        hidden_states = self.wav2vec2(input_values).last_hidden_state
        return torch.mean(hidden_states, dim=1)


# ── DANN-FT 相關結構 ─────────────────────────────────────────

class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha): ctx.alpha = alpha; return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output): return grad_output.neg() * ctx.alpha, None

class GradientReversalLayer(nn.Module):
    def forward(self, x, alpha=1.0): return GradientReversalFn.apply(x, alpha)

@dataclass
class DANNOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    spk_logits: Optional[torch.FloatTensor] = None

class Wav2Vec2DANNFinetune(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.shared_encoder = nn.Sequential(
            nn.Linear(config.hidden_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.dep_classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, config.num_labels)
        )
        self.spk_classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, getattr(config, "num_speakers", 151))
        )
        self.grl = GradientReversalLayer()
        self.init_weights()

    def get_embedding(self, input_values, attention_mask=None):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        pooled  = torch.mean(outputs[0], dim=1)
        return self.shared_encoder(pooled)

    def forward(self, input_values, attention_mask=None, return_dict=None,
                labels=None, speaker_labels=None, alpha=1.0):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        shared  = self.shared_encoder(torch.mean(outputs[0], dim=1))
        dep_logits = self.dep_classifier(shared)
        spk_logits = self.spk_classifier(self.grl(shared, alpha))
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(dep_logits.view(-1, self.config.num_labels), labels.view(-1))
        if speaker_labels is not None:
            mask = speaker_labels >= 0
            if mask.sum() > 0:
                loss_spk = nn.CrossEntropyLoss()(
                    spk_logits[mask].view(-1, spk_logits.size(-1)),
                    speaker_labels[mask].view(-1)
                )
                loss = loss + loss_spk if loss is not None else loss_spk
        use_return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if not use_return_dict:
            return ((loss, dep_logits, spk_logits) if loss is not None else (dep_logits, spk_logits))
        return DANNOutput(loss=loss, logits=dep_logits, spk_logits=spk_logits)


# ============================================================
#  工具函式
# ============================================================

def extract_speaker_id(filepath: str) -> str:
    return os.path.basename(str(filepath)).split('_')[0]


def get_target_speaker_set(test_csv: str) -> Set[str]:
    """從 test_B CSV 取出 38 位 target speaker ID。"""
    df = pd.read_csv(test_csv)
    return set(df['path'].apply(extract_speaker_id).unique())


def _find_best_checkpoint(run_dir: str) -> Optional[str]:
    """找最新的 checkpoint-XXXX，找不到則 fallback 到 run_dir 本身。"""
    if not os.path.isdir(run_dir):
        return None
    checkpoints = sorted(
        [d for d in os.listdir(run_dir) if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[-1])
    )
    if checkpoints:
        return os.path.join(run_dir, checkpoints[-1])
    if os.path.exists(os.path.join(run_dir, "config.json")):
        return run_dir
    return None


def _has_model_weights(dirpath: str) -> bool:
    """支援 pytorch_model.bin 和 model.safetensors。"""
    return (
        os.path.exists(os.path.join(dirpath, "pytorch_model.bin")) or
        os.path.exists(os.path.join(dirpath, "model.safetensors"))
    )


def _load_waveform(wav_path: str, processor):
    """載入音檔並前處理，失敗回傳 None。"""
    try:
        waveform, sr = torchaudio.load(wav_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        inputs = processor(waveform.squeeze().numpy(),
                           sampling_rate=16000,
                           return_tensors="pt", padding=True)
        return inputs
    except Exception:
        return None


# ── Embedding 抽取函式 ───────────────────────────────────────

def load_raw_w2v_embeddings(csv_path, processor, w2v_model,
                             filter_speakers: Optional[Set[str]] = None):
    """凍結 Wav2Vec2，抽 768 維 mean pooling。"""
    df = pd.read_csv(csv_path)
    feats, spks = [], []
    w2v_model.eval()
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), leave=False,
                           desc=f"  Raw W2V ← {os.path.basename(csv_path)}"):
            spk_id = extract_speaker_id(row['path'])
            if filter_speakers is not None and spk_id not in filter_speakers:
                continue
            inputs = _load_waveform(os.path.join(AUDIO_ROOT, str(row['path'])), processor)
            if inputs is None:
                continue
            emb = w2v_model(**{k: v.to(DEVICE) for k, v in inputs.items()}
                            ).last_hidden_state.mean(dim=1).cpu().numpy()
            feats.append(emb.squeeze())
            spks.append(spk_id)
    return np.array(feats), np.array(spks)


def load_huang_embeddings(csv_path, processor, model_dir,
                          filter_speakers: Optional[Set[str]] = None):
    """
    Huang Baseline：載入 best_model，抽 768 維。
    支援 pytorch_model.bin 和 model.safetensors。
    """
    config = AutoConfig.from_pretrained(model_dir)
    model  = Wav2Vec2ForSpeechClassification(config).to(DEVICE)

    bin_path  = os.path.join(model_dir, "pytorch_model.bin")
    safe_path = os.path.join(model_dir, "model.safetensors")

    if os.path.exists(bin_path):
        model.load_state_dict(torch.load(bin_path, map_location=DEVICE), strict=False)
    elif os.path.exists(safe_path):
        from safetensors.torch import load_file
        model.load_state_dict(load_file(safe_path, device=DEVICE), strict=False)
    else:
        raise FileNotFoundError(f"找不到模型權重檔：{model_dir}")

    model.eval()
    df = pd.read_csv(csv_path)
    feats, spks = [], []
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), leave=False,
                           desc=f"  Huang ← {os.path.basename(csv_path)}"):
            spk_id = extract_speaker_id(row['path'])
            if filter_speakers is not None and spk_id not in filter_speakers:
                continue
            inputs = _load_waveform(os.path.join(AUDIO_ROOT, str(row['path'])), processor)
            if inputs is None:
                continue
            emb = model.get_embedding(inputs['input_values'].to(DEVICE)).cpu().numpy()
            feats.append(emb.squeeze())
            spks.append(spk_id)
    return np.array(feats), np.array(spks)


def load_dann_embeddings(csv_path, processor, w2v_model, encoder_path,
                         filter_speakers: Optional[Set[str]] = None):
    """
    舊版 DANN：凍結 W2V（768 維）→ 外部 shared_encoder（128 維）。

    舊版儲存的 .pth key 格式為 "0.weight", "1.weight" ...
    但 SharedEncoder 包了一層 self.encoder，key 為 "encoder.0.weight" ...
    載入前自動做 key 映射，兩種格式都相容。
    """
    shared_encoder = SharedEncoder().to(DEVICE)
    raw_sd = torch.load(encoder_path, map_location=DEVICE)
    # 若 key 沒有 "encoder." 前綴，自動補上
    if any(not k.startswith("encoder.") for k in raw_sd.keys()):
        raw_sd = {f"encoder.{k}": v for k, v in raw_sd.items()}
    shared_encoder.load_state_dict(raw_sd)
    shared_encoder.eval()
    df = pd.read_csv(csv_path)
    feats, spks = [], []
    w2v_model.eval()
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), leave=False,
                           desc=f"  DANN ← {os.path.basename(csv_path)}"):
            spk_id = extract_speaker_id(row['path'])
            if filter_speakers is not None and spk_id not in filter_speakers:
                continue
            inputs = _load_waveform(os.path.join(AUDIO_ROOT, str(row['path'])), processor)
            if inputs is None:
                continue
            raw_emb = w2v_model(**{k: v.to(DEVICE) for k, v in inputs.items()}
                                ).last_hidden_state.mean(dim=1)
            emb = shared_encoder(raw_emb).cpu().numpy()
            feats.append(emb.squeeze())
            spks.append(spk_id)
    return np.array(feats), np.array(spks)


def load_dann_ft_embeddings(csv_path, processor, checkpoint_dir,
                            num_speakers=None,
                            filter_speakers: Optional[Set[str]] = None):
    """
    DANN-FT：從 Trainer checkpoint 載入完整模型，抽 shared_encoder 128 維。
    num_speakers 直接從 checkpoint 的 config.json 讀取，避免手動傳值出錯。
    """
    config = Wav2Vec2Config.from_pretrained(checkpoint_dir)
    if not hasattr(config, 'num_speakers') or config.num_speakers is None:
        config.num_speakers = num_speakers if num_speakers is not None else 38
    config.num_labels = 2
    model = Wav2Vec2DANNFinetune.from_pretrained(
        checkpoint_dir, config=config, ignore_mismatched_sizes=True
    
    ).to(DEVICE)
    model.eval()
    df = pd.read_csv(csv_path)
    feats, spks = [], []
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), leave=False,
                           desc=f"  DANN-FT ← {os.path.basename(csv_path)}"):
            spk_id = extract_speaker_id(row['path'])
            if filter_speakers is not None and spk_id not in filter_speakers:
                continue
            inputs = _load_waveform(os.path.join(AUDIO_ROOT, str(row['path'])), processor)
            if inputs is None:
                continue
            emb = model.get_embedding(inputs['input_values'].to(DEVICE)).cpu().numpy()
            feats.append(emb.squeeze())
            spks.append(spk_id)
    return np.array(feats), np.array(spks)


# ── Probe 評估（分兩種情境）─────────────────────────────────

def run_probe_scenario_A(X_train, spk_train, X_test, spk_test) -> float:
    """
    Scenario A：probe 在 151 位 control 上訓練，test 為 38 位 unseen。
    預期 Spk Acc ≈ 0%（probe classifier 沒見過這些 class）。
    """
    clf = LogisticRegression(max_iter=5000, class_weight='balanced', random_state=42)
    clf.fit(X_train, spk_train)
    return accuracy_score(spk_test, clf.predict(X_test))


def run_probe_scenario_B(X_hist, spk_hist, X_test, spk_test) -> float:
    """
    Scenario B：probe 只在 38 位 target 的 Historical（714 segs）上訓練，
    在同 38 人的 current data 上評估。
    精準量測 identity leakage，Spk Acc 越高 → leakage 越嚴重。
    """
    clf = LogisticRegression(max_iter=5000, class_weight='balanced', random_state=42)
    clf.fit(X_hist, spk_hist)
    return accuracy_score(spk_test, clf.predict(X_test))


# ============================================================
#  主程式
# ============================================================

if __name__ == "__main__":

    print("\n🧠 載入凍結 Wav2Vec2...")
    processor  = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    w2v_frozen = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(DEVICE)
    w2v_frozen.eval()

    # Scenario B 的 target speaker 集合（38 位）
    target_speakers_B = get_target_speaker_set(CSV_B_TEST)
    print(f"\n🎯 Scenario B target speakers: {len(target_speakers_B)} 位")

    # DANN-FT 模型初始化需要的 num_speakers
    num_speakers_A = len(pd.read_csv(CSV_A_TRAIN)['path'].apply(extract_speaker_id).unique())
    # Scenario B: run_dann_finetune_B.py builds speaker_map from TEST_CSV (38 target speakers)
    num_speakers_B = len(pd.read_csv(CSV_B_TEST)['path'].apply(extract_speaker_id).unique())
    print(f"  DANN-FT A num_speakers: {num_speakers_A}")
    print(f"  DANN-FT B num_speakers: {num_speakers_B}  (from TEST_CSV, matches training speaker_map)")

    # ----------------------------------------------------------
    # Linear Probing 特徵預先抽取（凍結模型，一次即可）
    # ----------------------------------------------------------
    print("\n📦 預先抽取 Linear Probing 特徵...")

    print("  [A] train（151 位 control）...")
    lp_A_train_X, lp_A_train_spk = load_raw_w2v_embeddings(
        CSV_A_TRAIN, processor, w2v_frozen, filter_speakers=None)

    print("  [A] test（38 位 unseen target）...")
    lp_A_test_X, lp_A_test_spk = load_raw_w2v_embeddings(
        CSV_A_TEST, processor, w2v_frozen, filter_speakers=None)

    print("  [B] probe train：train_B 中 target 38 人的 Historical 特徵...")
    lp_B_hist_X, lp_B_hist_spk = load_raw_w2v_embeddings(
        CSV_B_TRAIN, processor, w2v_frozen, filter_speakers=target_speakers_B)

    print("  [B] test（38 位 target current data）...")
    lp_B_test_X, lp_B_test_spk = load_raw_w2v_embeddings(
        CSV_B_TEST, processor, w2v_frozen, filter_speakers=None)

    print(f"\n  確認：A train {len(lp_A_train_X)} segs / {len(set(lp_A_train_spk))} spks，"
          f"A test {len(lp_A_test_X)} segs / {len(set(lp_A_test_spk))} spks")
    print(f"  確認：B hist  {len(lp_B_hist_X)} segs / {len(set(lp_B_hist_spk))} spks，"
          f"B test {len(lp_B_test_X)} segs / {len(set(lp_B_test_spk))} spks")

    # ----------------------------------------------------------
    # 結果容器
    # ----------------------------------------------------------
    results = {
        "Huang_A":   [], "Huang_B":   [],
        "Linear_A":  [], "Linear_B":  [],
        "DANN_A":    [], "DANN_B":    [],
        "DANN_FT_A": [], "DANN_FT_B": [],
    }

    # ----------------------------------------------------------
    # 逐 run 評估
    # ----------------------------------------------------------
    for run_i in range(1, TOTAL_RUNS + 1):
        print(f"\n{'='*65}")
        print(f"🎬 Run {run_i} / {TOTAL_RUNS}")
        print(f"{'='*65}")

        # ── 1. Huang A ──────────────────────────────────────────
        huang_A_path = HUANG_A_MODEL_TEMPLATE.format(run_i=run_i)
        if os.path.isdir(huang_A_path) and _has_model_weights(huang_A_path):
            print(f"\n[Huang A] 載入: {huang_A_path}")
            X_tr, spk_tr = load_huang_embeddings(CSV_A_TRAIN, processor, huang_A_path)
            X_te, spk_te = load_huang_embeddings(CSV_A_TEST,  processor, huang_A_path)
            acc = run_probe_scenario_A(X_tr, spk_tr, X_te, spk_te)
            results["Huang_A"].append(acc)
            print(f"  → Spk Acc: {acc:.4f}  [A, unseen, 預期 ≈ 0%]")
        else:
            print(f"  ⚠️  Huang A Run {run_i} 不存在或缺少權重，跳過")

        # ── 2. Huang B ──────────────────────────────────────────
        huang_B_path = HUANG_B_MODEL_TEMPLATE.format(run_i=run_i)
        if os.path.isdir(huang_B_path) and _has_model_weights(huang_B_path):
            print(f"\n[Huang B] 載入: {huang_B_path}")
            X_hist, spk_hist = load_huang_embeddings(
                CSV_B_TRAIN, processor, huang_B_path, filter_speakers=target_speakers_B)
            X_te, spk_te = load_huang_embeddings(CSV_B_TEST, processor, huang_B_path)
            acc = run_probe_scenario_B(X_hist, spk_hist, X_te, spk_te)
            results["Huang_B"].append(acc)
            print(f"  → Spk Acc: {acc:.4f}  [B, target-only probe]")
        else:
            print(f"  ⚠️  Huang B Run {run_i} 不存在或缺少權重，跳過")

        # ── 3. Linear A（已預先抽好）────────────────────────────
        print(f"\n[Linear A] 凍結 Wav2Vec2，特徵已預先抽好")
        acc = run_probe_scenario_A(lp_A_train_X, lp_A_train_spk,
                                   lp_A_test_X,  lp_A_test_spk)
        results["Linear_A"].append(acc)
        print(f"  → Spk Acc: {acc:.4f}  [A, unseen, 預期 ≈ 0%]")

        # ── 4. Linear B（已預先抽好）────────────────────────────
        print(f"\n[Linear B] 凍結 Wav2Vec2，特徵已預先抽好")
        acc = run_probe_scenario_B(lp_B_hist_X, lp_B_hist_spk,
                                   lp_B_test_X,  lp_B_test_spk)
        results["Linear_B"].append(acc)
        print(f"  → Spk Acc: {acc:.4f}  [B, target-only probe]")

        # ── 5. DANN A ────────────────────────────────────────────
        dann_A_path = DANN_A_ENCODER_TEMPLATE.format(run_i=run_i)
        if os.path.exists(dann_A_path):
            print(f"\n[DANN A] 載入: {dann_A_path}")
            X_tr, spk_tr = load_dann_embeddings(
                CSV_A_TRAIN, processor, w2v_frozen, dann_A_path)
            X_te, spk_te = load_dann_embeddings(
                CSV_A_TEST,  processor, w2v_frozen, dann_A_path)
            acc = run_probe_scenario_A(X_tr, spk_tr, X_te, spk_te)
            results["DANN_A"].append(acc)
            print(f"  → Spk Acc: {acc:.4f}  [A, unseen, 預期 ≈ 0%]")
        else:
            print(f"  ⚠️  DANN A Run {run_i} 不存在，跳過")

        # ── 6. DANN B ────────────────────────────────────────────
        dann_B_path = DANN_B_ENCODER_TEMPLATE.format(run_i=run_i)
        if os.path.exists(dann_B_path):
            print(f"\n[DANN B] 載入: {dann_B_path}")
            X_hist, spk_hist = load_dann_embeddings(
                CSV_B_TRAIN, processor, w2v_frozen, dann_B_path,
                filter_speakers=target_speakers_B)
            X_te, spk_te = load_dann_embeddings(
                CSV_B_TEST, processor, w2v_frozen, dann_B_path)
            acc = run_probe_scenario_B(X_hist, spk_hist, X_te, spk_te)
            results["DANN_B"].append(acc)
            print(f"  → Spk Acc: {acc:.4f}  [B, target-only probe]")
        else:
            print(f"  ⚠️  DANN B Run {run_i} 不存在，跳過")

        # ── 7. DANN-FT A ─────────────────────────────────────────
        dann_ft_A_ckpt = _find_best_checkpoint(DANN_FT_A_DIR_TEMPLATE.format(run_i=run_i))
        if dann_ft_A_ckpt is not None and _has_model_weights(dann_ft_A_ckpt):
            print(f"\n[DANN-FT A] 載入: {dann_ft_A_ckpt}")
            X_tr, spk_tr = load_dann_ft_embeddings(
                CSV_A_TRAIN, processor, dann_ft_A_ckpt)
            X_te, spk_te = load_dann_ft_embeddings(
                CSV_A_TEST,  processor, dann_ft_A_ckpt)
            acc = run_probe_scenario_A(X_tr, spk_tr, X_te, spk_te)
            results["DANN_FT_A"].append(acc)
            print(f"  → Spk Acc: {acc:.4f}  [A, unseen, 預期 ≈ 0%]")
        else:
            print(f"  ⏳  DANN-FT A Run {run_i} 尚無 checkpoint，跳過")

        # ── 8. DANN-FT B ─────────────────────────────────────────
        dann_ft_B_ckpt = _find_best_checkpoint(DANN_FT_B_DIR_TEMPLATE.format(run_i=run_i))
        if dann_ft_B_ckpt is not None and _has_model_weights(dann_ft_B_ckpt):
            print(f"\n[DANN-FT B] 載入: {dann_ft_B_ckpt}")
            X_hist, spk_hist = load_dann_ft_embeddings(
                CSV_B_TRAIN, processor, dann_ft_B_ckpt,
                filter_speakers=target_speakers_B)
            X_te, spk_te = load_dann_ft_embeddings(
                CSV_B_TEST, processor, dann_ft_B_ckpt)
            acc = run_probe_scenario_B(X_hist, spk_hist, X_te, spk_te)
            results["DANN_FT_B"].append(acc)
            print(f"  → Spk Acc: {acc:.4f}  [B, target-only probe]")
        else:
            print(f"  ⏳  DANN-FT B Run {run_i} 尚無 checkpoint，跳過")

    # ----------------------------------------------------------
    # 彙總輸出
    # ----------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"📊 Speaker Probe 彙總結果")
    print(f"{'─'*65}")
    print(f"  Scenario A：probe train=151 control，test=38 unseen → 預期 ≈ 0%")
    print(f"  Scenario B：probe train=38 target Historical，test=38 target current")
    print(f"{'='*65}")
    print(f"{'模型':<14} {'情境':<6} {'有效runs':<10} {'平均 Spk Acc':<16} 標準差")
    print(f"{'─'*60}")

    scenario_label = {
        "Huang_A":   "A", "Huang_B":   "B",
        "Linear_A":  "A", "Linear_B":  "B",
        "DANN_A":    "A", "DANN_B":    "B",
        "DANN_FT_A": "A", "DANN_FT_B": "B",
    }

    summary_rows = []
    for name, accs in results.items():
        scen = scenario_label[name]
        if len(accs) == 0:
            print(f"{name:<14} {scen:<6} {'0':<10} {'N/A':<16} N/A")
        else:
            arr  = np.array(accs)
            mean, std = arr.mean(), arr.std()
            print(f"{name:<14} {scen:<6} {len(accs):<10} {mean:.4f}{'':>10} ± {std:.4f}")
            summary_rows.append({
                "model": name, "scenario": scen,
                "valid_runs": len(accs),
                "spk_acc_mean": round(mean, 4),
                "spk_acc_std":  round(std,  4),
            })

    if summary_rows:
        out_path = "speaker_probe_summary.csv"
        pd.DataFrame(summary_rows).to_csv(out_path, index=False)
        print(f"\n✅ 彙總已儲存至 {out_path}")

    # DANN-FT 若有 run 被跳過，提醒補跑
    ft_a_done = len(results["DANN_FT_A"])
    ft_b_done = len(results["DANN_FT_B"])
    if ft_a_done < TOTAL_RUNS or ft_b_done < TOTAL_RUNS:
        print(f"\n⏳ DANN-FT A: {ft_a_done}/{TOTAL_RUNS} runs 完成")
        print(f"⏳ DANN-FT B: {ft_b_done}/{TOTAL_RUNS} runs 完成")
        print(f"   訓練完成後重新執行此腳本，會自動補上剩餘 runs 的結果。")
