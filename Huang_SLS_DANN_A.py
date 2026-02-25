"""
File — Huang + SLS + DANN（無 fine-tune，Scenario A）
=====================================================
架構說明：
  - Wav2Vec2 主幹：完全凍結（包含 CNN feature extractor + Transformer encoder）
  - SLS (Stochastic Layer Selection)：可學習的加權融合所有 hidden states
  - DANN domain classifier：對抗式訓練，以 speaker 為 domain
  - dep_classifier：二元憂鬱分類 (binary)

與 replicate_huang (File 2/3) 的差異：
  → wav2vec2 全部凍結（非只凍結 CNN），只有 SLS 權重 + 兩個分類頭可訓練
  → 加入 GRL + speaker domain classifier
  → 執行 TOTAL_RUNS 次，每次重新初始化 SLS/分類頭權重

修正清單（相對於使用者提供的草稿）：
  1. import 語法分行
  2. forward 回傳 SpeechClassifierOutput（ModelOutput 子類），Trainer 才能正確提取 loss
  3. speaker_labels 從 CSV 抽取後存入 dataset，讓 Trainer 能傳入 forward
  4. 補上 AUDIO_ROOT
  5. alpha 動態調整：在自訂 CTCTrainer.training_step 中依全局 step 計算
  6. config 設定 output_hidden_states=True
  7. 補完 TOTAL_RUNS 迴圈
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Union, Any
from math import sqrt, exp

from torch.autograd import Function
from datasets import Dataset as HFDataset
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2Config,
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    set_seed,
)
from transformers.file_utils import ModelOutput
from packaging import version
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    mean_squared_error,
)

# ============================================================
#  設定區
# ============================================================
TRAIN_CSV  = "./experiment_sisman_scientific/scenario_A_screening/train.csv"
TEST_CSV   = "./experiment_sisman_scientific/scenario_A_screening/test.csv"
AUDIO_ROOT = "/export/fs05/hyeh10/depression/daic_5utt_full/merged_5"  # ← 修正 1：補上 AUDIO_ROOT

MODEL_NAME = "facebook/wav2vec2-base"
OUTPUT_DIR = "./output_huang_sls_dann_A"

SEED       = 42
TOTAL_RUNS = 5
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
BATCH_SIZE    = 4
GRAD_ACCUM    = 2
EVAL_STEPS    = 50
SAVE_STEPS    = 50
LOGGING_STEPS = 50
SAVE_TOTAL_LIMIT = 2
FP16 = torch.cuda.is_available()

LABEL_MAP   = {"non": 0, "0": 0, 0: 0, "dep": 1, "1": 1, 1: 1}
LABEL_NAMES = ["non-depressed", "depressed"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
#  模型定義
# ============================================================

@dataclass
class SpeechClassifierOutput(ModelOutput):
    # 修正 2：回傳 ModelOutput 子類，Trainer 才能正確提取 loss / logits
    loss:           Optional[torch.FloatTensor] = None
    logits:         torch.FloatTensor           = None
    speaker_logits: Optional[torch.FloatTensor] = None
    hidden_states:  Optional[Tuple[torch.FloatTensor]] = None
    attentions:     Optional[Tuple[torch.FloatTensor]] = None


class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class Wav2Vec2_SLS_DANN(Wav2Vec2PreTrainedModel):
    """
    Wav2Vec2 + SLS + DANN，無 fine-tune
    - wav2vec2 全部凍結
    - SLS 加權融合所有 hidden states（13 層，含 CNN embedding 輸出）
    - dep_classifier：binary classification
    - spk_classifier：speaker domain classifier（透過 GRL 對抗）
    - alpha 由外部 CTCTrainer 動態注入（存於 self._alpha）
    """
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)

        # 凍結 wav2vec2 全部參數（含 CNN + Transformer）
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

        num_layers = config.num_hidden_layers + 1  # +1 for CNN embedding output (layer 0)
        self.sls_weights = nn.Parameter(torch.ones(num_layers))

        self.down_proj = nn.Sequential(
            nn.Linear(config.hidden_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.dep_classifier = nn.Linear(128, config.num_labels)
        self.spk_classifier = nn.Linear(128, 200)

        # alpha 由 CTCTrainer 在每個 step 前更新
        self._alpha = 0.0

        self.init_weights()

    def freeze_feature_extractor(self):
        # 相容 train.py 呼叫習慣，此處 wav2vec2 已全凍，保留此介面即可
        pass

    def forward(
        self,
        input_values,
        attention_mask=None,
        labels=None,
        speaker_labels=None,  # 修正 3：由 dataset 欄位傳入
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # 修正 6：output_hidden_states 必須為 True，SLS 才能取到所有層
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,  # 強制開啟
            output_attentions=output_attentions,
            return_dict=True,
        )

        # hidden_states: tuple of (num_layers+1) tensors, each [B, T, H]
        hidden_states = torch.stack(outputs.hidden_states)  # [L, B, T, H]
        weights = torch.softmax(self.sls_weights, dim=0)    # [L]
        fused = (hidden_states * weights.view(-1, 1, 1, 1)).sum(0)  # [B, T, H]

        # Mean pooling over time
        shared = self.down_proj(torch.mean(fused, dim=1))   # [B, 128]

        dep_logits = self.dep_classifier(shared)             # [B, num_labels]
        spk_logits = self.spk_classifier(
            GradientReversalFn.apply(shared, self._alpha)   # alpha 動態注入
        )                                                    # [B, num_speakers]

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            dep_loss = loss_fct(dep_logits, labels)
            loss = dep_loss
        if speaker_labels is not None:
            mask = speaker_labels >= 0  # 過濾 test 的陌生人（-1）
            if mask.sum() > 0:
                spk_loss = nn.CrossEntropyLoss()(
                    spk_logits[mask].view(-1, spk_logits.size(-1)),
                    speaker_labels[mask].view(-1)
                )
                loss = loss + self._alpha * spk_loss if loss is not None else spk_loss

        return SpeechClassifierOutput(
            loss=loss,
            logits=dep_logits,
            speaker_logits=spk_logits,
            hidden_states=None,
            attentions=None,
        )


# ============================================================
#  DataCollator（含 speaker_labels）
# ============================================================

@dataclass
class DataCollatorWithSpeaker:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features   = [f["labels"]          for f in features]
        speaker_features = [f["speaker_labels"]  for f in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        batch["labels"]         = torch.tensor(label_features,   dtype=torch.long)
        batch["speaker_labels"] = torch.tensor(speaker_features, dtype=torch.long)
        return batch


# ============================================================
#  compute_metrics
# ============================================================

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    acc = accuracy_score(p.label_ids, preds)
    f1  = f1_score(p.label_ids, preds, average="macro")
    return {"accuracy": acc, "f1": f1}


# ============================================================
#  CTCTrainer — 修正 5：動態注入 alpha
# ============================================================

if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast


class CTCTrainer(Trainer):
    """
    在每個 training_step 前，依全局訓練進度動態更新 model._alpha。
    alpha 公式與 run_dann.py 一致：alpha = 2/(1+exp(-10*p)) - 1，p ∈ [0,1]
    """
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:

        # 計算當前訓練進度 p ∈ [0, 1]
        total_steps = self.args.max_steps if self.args.max_steps > 0 else (
            len(self.train_dataset) // (
                self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps
            ) * int(self.args.num_train_epochs)
        )
        current_step = self.state.global_step
        p = float(current_step) / max(total_steps, 1)
        alpha = 2.0 / (1.0 + exp(-10.0 * p)) - 1.0

        # 注入 alpha 到模型
        model._alpha = alpha

        model.train()
        inputs = self._prepare_inputs(inputs)

        is_amp_used = self.args.fp16 or self.args.bf16
        if is_amp_used:
            with torch.amp.autocast("cuda"):
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if is_amp_used:
            if hasattr(self, "scaler") and self.scaler is not None:
                self.scaler.scale(loss).backward()
            elif hasattr(self, "accelerator"):
                self.accelerator.backward(loss)
            else:
                loss.backward()
        else:
            loss.backward()

        return loss.detach()


# ============================================================
#  資料載入與預處理
# ============================================================

def extract_speaker_id(filepath: str) -> str:
    return os.path.basename(filepath).split("_")[0]


def load_audio_dataset(csv_path: str, speaker_to_idx: dict = None, is_train: bool = True):
    """
    載入 CSV，回傳 HFDataset 及（訓練時建立的）speaker_to_idx。
    修正 3：speaker_labels 存入 dataset，Trainer 可以直接傳給 forward。
    """
    df = pd.read_csv(csv_path)
    print(f"📂 讀取 {csv_path}，共 {len(df)} 筆資料")

    # 建立 speaker → index 對照表（只在訓練集建立）
    if is_train and speaker_to_idx is None:
        all_wav_paths = df["path"].tolist()
        all_speakers  = sorted(set(extract_speaker_id(p) for p in all_wav_paths))
        speaker_to_idx = {spk: idx for idx, spk in enumerate(all_speakers)}
        print(f"🔍 偵測到 {len(speaker_to_idx)} 位 speaker")

    records = []
    skipped = 0
    for _, row in df.iterrows():
        wav_path  = os.path.join(AUDIO_ROOT, row["path"])
        raw_label = str(row["label"]).strip().lower()

        if raw_label not in LABEL_MAP:
            skipped += 1
            continue
        if not os.path.exists(wav_path):
            print(f"⚠️ 檔案不存在: {wav_path}")
            skipped += 1
            continue

        spk_str = extract_speaker_id(wav_path)
        spk_idx = speaker_to_idx.get(spk_str, -1)  # 陌生人(test) → -1，forward 裡 mask 過濾

        records.append({
            "path":           wav_path,
            "label":          LABEL_MAP[raw_label],
            "speaker_labels": spk_idx,
        })

    if skipped:
        print(f"⚠️ 跳過 {skipped} 筆無效/不存在的資料")
    print(f"✅ 成功載入 {len(records)} 筆資料")

    dataset = HFDataset.from_dict({
        "path":           [r["path"]           for r in records],
        "label":          [r["label"]          for r in records],
        "speaker_labels": [r["speaker_labels"] for r in records],
    })
    return dataset, speaker_to_idx


def speech_file_to_array_fn(batch, processor):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    if speech_array.shape[0] > 1:
        speech_array = torch.mean(speech_array, dim=0, keepdim=True)
    speech_array = speech_array.squeeze().numpy()
    if sampling_rate != 16000:
        import librosa
        speech_array = librosa.resample(speech_array, orig_sr=sampling_rate, target_sr=16000)
    batch["speech"] = speech_array
    return batch


def preprocess_function(batch, processor):
    result = processor(batch["speech"], sampling_rate=16000, return_tensors="np", padding=False)
    batch["input_values"] = result.input_values[0]
    batch["labels"]       = batch["label"]
    return batch


def split_train_valid(dataset: HFDataset, valid_ratio: float = 0.15, seed: int = 42):
    split = dataset.train_test_split(test_size=valid_ratio, seed=seed)
    return split["train"], split["test"]


# ============================================================
#  評估與報告
# ============================================================

def full_evaluation(trainer, test_dataset, output_dir, run_i):
    predictions = trainer.predict(test_dataset)
    preds  = predictions.predictions
    if isinstance(preds, tuple):
        preds = preds[0]
    y_pred = np.argmax(preds, axis=1)
    y_true = predictions.label_ids

    print("\n" + "=" * 60)
    print(f"📊 [Run {run_i}] Classification Report")
    print("=" * 60)
    report = classification_report(y_true, y_pred, target_names=LABEL_NAMES,
                                   zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print(report_df)

    cm     = confusion_matrix(y_true, y_pred)
    cm_df  = pd.DataFrame(cm, index=LABEL_NAMES, columns=LABEL_NAMES)
    print("\n📊 Confusion Matrix:")
    print(cm_df)

    mse  = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    report_df["MSE"]  = mse
    report_df["RMSE"] = rmse

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc     = auc(fpr, tpr)

    results_path = os.path.join(output_dir, f"results_run{run_i}")
    os.makedirs(results_path, exist_ok=True)

    report_df.to_csv(os.path.join(results_path, "clsf_report.csv"), sep="\t")
    cm_df.to_csv(    os.path.join(results_path, "conf_matrix.csv"), sep="\t")

    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC={roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - Scenario A Run {run_i}")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(results_path, "roc_curve.png"))
    plt.close()

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro")
    print(f"\n🎯 Test Accuracy: {acc:.4f} | F1 (macro): {f1:.4f} | AUC: {roc_auc:.4f}")
    print(f"✅ 結果已儲存至 {results_path}")
    return {"accuracy": acc, "f1": f1, "auc": roc_auc}


# ============================================================
#  主程式 — TOTAL_RUNS 次迴圈（修正 7）
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Huang + SLS + DANN（無 fine-tune）— Scenario A")
    print("   wav2vec2：全部凍結")
    print("   SLS 權重 + dep/spk classifier：可訓練")
    print("   DANN alpha：動態遞增")
    print("=" * 60)

    set_seed(SEED)
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)

    # ── 資料只準備一次 ──────────────────────────────────────
    print("\n📦 載入資料集（只執行一次）...")
    train_dataset_full, speaker_to_idx = load_audio_dataset(
        TRAIN_CSV, is_train=True
    )
    test_dataset_raw, _ = load_audio_dataset(
        TEST_CSV, speaker_to_idx=speaker_to_idx, is_train=False
    )
    num_speakers = len(speaker_to_idx)
    print(f"👥 共 {num_speakers} 位 speaker")

    print("\n🔊 預處理音訊（只執行一次）...")
    train_dataset_full = train_dataset_full.map(
        speech_file_to_array_fn, fn_kwargs={"processor": processor}
    )
    test_dataset_raw = test_dataset_raw.map(
        speech_file_to_array_fn, fn_kwargs={"processor": processor}
    )
    train_dataset_full = train_dataset_full.map(
        preprocess_function, fn_kwargs={"processor": processor}
    )
    test_dataset = test_dataset_raw.map(
        preprocess_function, fn_kwargs={"processor": processor}
    )

    # ── TOTAL_RUNS 次實驗迴圈（修正 7） ────────────────────
    all_results = []
    for run_i in range(1, TOTAL_RUNS + 1):
        print(f"\n{'='*60}")
        print(f"🎬 Run {run_i} / {TOTAL_RUNS}")
        print(f"{'='*60}")

        set_seed(SEED + run_i)  # 每次 run 用不同 seed，確保隨機性

        train_dataset = train_dataset_full
        eval_dataset  = test_dataset
        print(f"📊 Train: {len(train_dataset)} | Test(eval): {len(test_dataset)}")

        # 每次重新初始化模型（修正 6：config 設定 output_hidden_states）
        config = Wav2Vec2Config.from_pretrained(
            MODEL_NAME,
            num_labels=2,
            num_speakers=num_speakers,
            final_dropout=0.1,
            output_hidden_states=True,   # 修正 6：讓 wav2vec2 輸出所有 hidden states
        )
        model = Wav2Vec2_SLS_DANN.from_pretrained(MODEL_NAME, config=config)

        # 確認凍結狀態
        frozen = sum(1 for p in model.wav2vec2.parameters() if not p.requires_grad)
        total  = sum(1 for p in model.wav2vec2.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"❄️  wav2vec2 凍結: {frozen}/{total} 個參數組")
        print(f"🔥 可訓練參數總量: {trainable:,}")

        data_collator = DataCollatorWithSpeaker(processor=processor, padding=True)

        run_output_dir = os.path.join(OUTPUT_DIR, f"run_{run_i}")
        os.makedirs(run_output_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=run_output_dir,
            per_device_train_batch_size=BATCH_SIZE,
            label_names=["labels"],
            dataloader_drop_last=True,
            per_device_eval_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            evaluation_strategy="steps",
            num_train_epochs=NUM_EPOCHS,
            fp16=FP16,
            save_steps=SAVE_STEPS,
            eval_steps=EVAL_STEPS,
            logging_steps=LOGGING_STEPS,
            learning_rate=LEARNING_RATE,
            save_total_limit=SAVE_TOTAL_LIMIT,
            seed=SEED + run_i,
            data_seed=SEED + run_i,
            load_best_model_at_end=True,
            # metric_for_best_model 未設定 → 預設用 validation loss，對齊 train.py
            report_to="none",
            remove_unused_columns=False,  # 🔥 防止 Trainer 刪掉 speaker_labels
        )

        trainer = CTCTrainer(
            model=model,
            data_collator=data_collator,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=processor.feature_extractor,
        )

        print("⚔️ 開始訓練...")
        try:
            trainer.train()
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("⚠️ GPU 記憶體不足，清除快取後跳過此 run")
                torch.cuda.empty_cache()
                continue
            raise e

        # 儲存最佳模型
        best_path = os.path.join(run_output_dir, "best_model")
        trainer.save_model(best_path)
        processor.save_pretrained(best_path)
        print(f"💾 最佳模型儲存至: {best_path}")

        pth_path = os.path.join(OUTPUT_DIR, f"huang_sls_dann_A_shared_encoder_run_{run_i}.pth")
        torch.save(trainer.model.down_proj.state_dict(), pth_path)
        print(f"🔑 down_proj .pth 儲存至: {pth_path}")

        # 完整評估
        results = full_evaluation(trainer, test_dataset, OUTPUT_DIR, run_i)
        all_results.append(results)

    # ── 跨 run 統計 ────────────────────────────────────────
    if all_results:
        print("\n" + "=" * 60)
        print("📈 跨 Run 統計摘要")
        print("=" * 60)
        for metric in ["accuracy", "f1", "auc"]:
            vals = [r[metric] for r in all_results]
            print(f"  {metric.upper():10s}  mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  "
                  f"min={np.min(vals):.4f}  max={np.max(vals):.4f}")

    print("\n🏁 Scenario A 實驗完成！")