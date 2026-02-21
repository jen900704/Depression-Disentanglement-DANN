"""
DANN with Fine-tuned Transformer — Scenario B (Longitudinal / Speaker Overlap)
===============================================================================
架構說明：
  - Wav2Vec2 CNN：凍結（同 Huang）
  - Wav2Vec2 Transformer：可訓練（同 Huang）
  - Mean Pooling → 768 維
  - Shared Encoder：FC(768→128), BN, ReLU, Dropout(0.3)（同舊 DANN）
  - Depression Classifier：FC(128→64), ReLU, FC(64→2)
  - Speaker Classifier：GRL + FC(128→64), ReLU, FC(64→N_spk)
  - Total Loss = L_dep + L_spk

與舊 DANN (run_dann_scenario_B_v2.py) 的差異：
  → Wav2Vec2 Transformer 開放微調（舊版完全凍結）
  → 使用 HuggingFace Trainer 框架（對齊 Huang 的訓練方式）
  → lr=1e-5, batch=4, grad_accum=2, epochs=10（對齊 Huang yaml）
  → 說話者對抗 loss 在 training_step 內計算

與 Huang (replicate_huang_partial_finetuning.py) 的差異：
  → 多了 Shared Encoder（768→128）
  → 多了 Speaker Classifier + GRL
  → Total Loss = L_dep + L_spk（舊版只有 L_dep）
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Union, Any
from math import sqrt
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
TRAIN_CSV  = "./experiment_sisman_scientific/scenario_B_monitoring/train.csv"
TEST_CSV   = "./experiment_sisman_scientific/scenario_B_monitoring/test.csv"
AUDIO_ROOT = ""   # CSV 內已是絕對路徑

MODEL_NAME = "facebook/wav2vec2-base"
OUTPUT_DIR = "./output_dann_finetune_B"

SEED                        = 103
NUM_EPOCHS                  = 10
LEARNING_RATE               = 1e-5
PER_DEVICE_TRAIN_BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH_SIZE  = 4
GRADIENT_ACCUMULATION_STEPS = 2
FP16                        = torch.cuda.is_available()
EVAL_STEPS                  = 300
SAVE_STEPS                  = 300
LOGGING_STEPS               = 50
SAVE_TOTAL_LIMIT            = 2
TOTAL_RUNS                  = 5

LABEL_MAP   = {"non": 0, "0": 0, 0: 0, "dep": 1, "1": 1, 1: 1}
LABEL_NAMES = ["non-depressed", "depressed"]
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

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
#  ModelOutput
# ============================================================
@dataclass
class DANNOutput(ModelOutput):
    loss:          Optional[torch.FloatTensor] = None
    loss_dep:      Optional[torch.FloatTensor] = None
    loss_spk:      Optional[torch.FloatTensor] = None
    logits:        torch.FloatTensor           = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions:    Optional[Tuple[torch.FloatTensor]] = None

# ============================================================
#  模型定義
# ============================================================
class Wav2Vec2DANNFinetune(Wav2Vec2PreTrainedModel):
    """
    Wav2Vec2 (CNN frozen, Transformer trainable)
    + Shared Encoder (768→128)
    + Depression Classifier
    + Speaker Classifier with GRL
    """
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2      = Wav2Vec2Model(config)
        hidden             = config.hidden_size   # 768
        num_labels         = config.num_labels    # 2
        num_speakers       = getattr(config, "num_speakers", 38)
        self.pooling_mode  = getattr(config, "pooling_mode", "mean")

        # Shared Encoder：768 → 128
        self.shared_encoder = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Depression Classifier
        self.dep_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_labels),
        )

        # Speaker Classifier（接 GRL）
        self.spk_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_speakers),
        )

        self.grl = GradientReversalLayer()
        self.init_weights()

    def freeze_feature_extractor(self):
        """只凍結 CNN，Transformer 保持可訓練"""
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy(self, hidden_states):
        if self.pooling_mode == "mean":
            return torch.mean(hidden_states, dim=1)
        elif self.pooling_mode == "max":
            return torch.max(hidden_states, dim=1)[0]
        raise Exception("Pooling: 'mean' or 'max'")

    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        speaker_labels=None,
        alpha=1.0,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Mean pooling → 768 維
        pooled = self.merged_strategy(outputs[0])

        # Shared Encoder → 128 維
        shared = self.shared_encoder(pooled)

        # Depression 分支
        dep_logits = self.dep_classifier(shared)

        # Speaker 分支（GRL）
        rev = self.grl(shared, alpha)
        spk_logits = self.spk_classifier(rev)

        # Loss 計算
        loss = None
        loss_dep = None
        loss_spk = None

        if labels is not None:
            loss_dep = nn.CrossEntropyLoss()(
                dep_logits.view(-1, self.config.num_labels), labels.view(-1)
            )
            loss = loss_dep

        if speaker_labels is not None:
            num_speakers = spk_logits.size(-1)
            loss_spk = nn.CrossEntropyLoss()(
                spk_logits.view(-1, num_speakers), speaker_labels.view(-1)
            )
            loss = loss_dep + loss_spk if loss_dep is not None else loss_spk

        if not return_dict:
            output = (dep_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return DANNOutput(
            loss=loss,
            loss_dep=loss_dep,
            loss_spk=loss_spk,
            logits=dep_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# ============================================================
#  DataCollator（含 speaker_labels）
# ============================================================
@dataclass
class DataCollatorDANN:
    processor:          Wav2Vec2Processor
    padding:            Union[bool, str] = True
    max_length:         Optional[int]    = None
    pad_to_multiple_of: Optional[int]    = None

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features  = [{"input_values": f["input_values"]} for f in features]
        label_features  = [f["labels"]          for f in features]
        speaker_features = [f["speaker_label"]  for f in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch["labels"]         = torch.tensor(label_features,   dtype=torch.long)
        batch["speaker_labels"] = torch.tensor(speaker_features, dtype=torch.long)
        return batch

# ============================================================
#  compute_metrics（憂鬱症分類，加入 Tuple 防呆）
# ============================================================
def compute_metrics(p: EvalPrediction):
    # p.predictions 可能會把 loss_dep, loss_spk, logits 通通裝在 tuple 裡
    if isinstance(p.predictions, tuple):
        # 我們只找維度是 2 的那個陣列（也就是 logits，形狀為 [batch_size, 2]）
        for pred_array in p.predictions:
            if isinstance(pred_array, np.ndarray) and pred_array.ndim == 2:
                preds = pred_array
                break
    else:
        preds = p.predictions

    preds = np.argmax(preds, axis=1)
    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

# ============================================================
#  DANNTrainer — 在 training_step 加入 speaker loss 和動態 alpha
# ============================================================
if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True

class DANNTrainer(Trainer):
    def __init__(self, *args, total_steps=1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_steps  = total_steps
        self.current_step = 0

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        # 動態 alpha（從 0 漸增到 1）
        p     = self.current_step / max(self.total_steps, 1)
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1
        self.current_step += 1

        # 注入 alpha
        inputs["alpha"] = alpha

        is_amp = self.args.fp16 or self.args.bf16
        if is_amp:
            with torch.amp.autocast("cuda"):
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if is_amp:
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
def extract_speaker_id(filepath):
    return os.path.basename(str(filepath)).split('_')[0]

def build_speaker_map(csv_path):
    df = pd.read_csv(csv_path)
    speakers = sorted(df['path'].apply(extract_speaker_id).unique())
    return {spk: idx for idx, spk in enumerate(speakers)}

def load_audio_dataset(csv_path: str, speaker_map: dict) -> HFDataset:
    df = pd.read_csv(csv_path)
    print(f"📂 讀取 {csv_path}，共 {len(df)} 筆資料")

    records, skipped = [], 0
    for _, row in df.iterrows():
        wav_path  = os.path.join(AUDIO_ROOT, row["path"])
        raw_label = str(row["label"]).strip().lower()
        spk_id    = extract_speaker_id(row["path"])

        if raw_label not in LABEL_MAP:
            skipped += 1
            continue
        if not os.path.exists(wav_path):
            print(f"⚠️ 檔案不存在: {wav_path}")
            skipped += 1
            continue

        records.append({
            "path":         wav_path,
            "label":        LABEL_MAP[raw_label],
            "speaker_label": speaker_map.get(spk_id, 0),
        })

    if skipped > 0:
        print(f"⚠️ 跳過 {skipped} 筆")
    print(f"✅ 成功載入 {len(records)} 筆")

    return HFDataset.from_dict({
        "path":          [r["path"]          for r in records],
        "label":         [r["label"]         for r in records],
        "speaker_label": [r["speaker_label"] for r in records],
    })

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
    result = processor(
        batch["speech"],
        sampling_rate=16000,
        return_tensors="np",
        padding=False,
        return_attention_mask=False,
    )
    batch["input_values"] = result.input_values[0]
    batch["labels"]       = batch["label"]
    batch["speaker_label"] = batch["speaker_label"]
    return batch

# ============================================================
#  評估 (加入 Tuple 防呆)
# ============================================================
def full_evaluation(trainer, test_dataset, output_dir, run_i):
    predictions = trainer.predict(test_dataset)
    preds = predictions.predictions
    
    # --- 加入防呆邏輯 ---
    if isinstance(preds, tuple):
        for pred_array in preds:
            if isinstance(pred_array, np.ndarray) and pred_array.ndim == 2:
                preds = pred_array
                break
    # ------------------
    
    y_pred = np.argmax(preds, axis=1)
    y_true = predictions.label_ids

    report    = classification_report(y_true, y_pred, target_names=LABEL_NAMES,
                                      zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print(report_df)

    cm    = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=LABEL_NAMES, columns=LABEL_NAMES)
    print("\n📊 Confusion Matrix:")
    print(cm_df)

    mse  = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    report_df["MSE"]  = mse
    report_df["RMSE"] = rmse

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc     = auc(fpr, tpr)

    results_path = os.path.join(output_dir, f"results_run_{run_i}")
    os.makedirs(results_path, exist_ok=True)
    report_df.to_csv(os.path.join(results_path, "clsf_report.csv"), sep="\t")
    cm_df.to_csv(os.path.join(results_path,     "conf_matrix.csv"), sep="\t")

    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2,
             label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - DANN Finetune B (Run {run_i})")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(results_path, "roc_curve.png"))
    plt.close()
    print(f"✅ Run {run_i} 結果已儲存至 {results_path}")

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro")
    return {"accuracy": acc, "f1": f1, "auc": roc_auc}

# ============================================================
#  主程式
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 DANN + Fine-tuned Transformer — Scenario B")
    print("   CNN: Frozen | Transformer: Trainable | GRL: On")
    print(f"   實驗次數: {TOTAL_RUNS} 次")
    print("=" * 60)

    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    map_kwargs = {"fn_kwargs": {"processor": processor}}

    # Speaker map 從 train set 建立
    print("\n🔍 建立 Speaker Map...")
    speaker_map  = build_speaker_map(TRAIN_CSV)
    num_speakers = len(speaker_map)
    print(f"   共 {num_speakers} 位說話者")

    # 資料只載入一次
    print("\n📦 載入並預處理資料集...")
    train_raw = load_audio_dataset(TRAIN_CSV, speaker_map)
    test_raw  = load_audio_dataset(TEST_CSV,  speaker_map)

    train_raw = train_raw.map(speech_file_to_array_fn, **map_kwargs)
    test_raw  = test_raw.map(speech_file_to_array_fn,  **map_kwargs)

    train_dataset = train_raw.map(preprocess_function, **map_kwargs)
    test_dataset  = test_raw.map(preprocess_function,  **map_kwargs)

    print(f"📊 Train: {len(train_dataset)} 筆 | Test: {len(test_dataset)} 筆")

    data_collator = DataCollatorDANN(processor=processor, padding=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = []

    for run_i in range(1, TOTAL_RUNS + 1):
        print(f"\n{'='*60}")
        print(f"🎬 Run {run_i} / {TOTAL_RUNS}")
        print(f"{'='*60}")

        run_seed = SEED + run_i - 1
        set_seed(run_seed)

        # 模型初始化
        config = Wav2Vec2Config.from_pretrained(
            MODEL_NAME,
            num_labels=2,
            final_dropout=0.1,
            pooling_mode="mean",
        )
        config.num_speakers = num_speakers  # <--- 🔴 強制寫入 config，不讓它被忽略！
        model = Wav2Vec2DANNFinetune.from_pretrained(MODEL_NAME, config=config)
        model.freeze_feature_extractor()   # 只凍結 CNN
        print(f"❄️ CNN 已凍結，Transformer 可訓練")

        run_output_dir = os.path.join(OUTPUT_DIR, f"run_{run_i}")
        os.makedirs(run_output_dir, exist_ok=True)

        # 計算 total_steps 供 alpha 動態調整
        steps_per_epoch = len(train_dataset) // (PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
        total_steps     = steps_per_epoch * NUM_EPOCHS

        training_args = TrainingArguments(
            output_dir=run_output_dir,
            per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            evaluation_strategy="steps",
            num_train_epochs=NUM_EPOCHS,
            fp16=FP16,
            save_steps=SAVE_STEPS,
            eval_steps=EVAL_STEPS,
            logging_steps=LOGGING_STEPS,
            learning_rate=LEARNING_RATE,
            save_total_limit=SAVE_TOTAL_LIMIT,
            seed=run_seed,
            remove_unused_columns=False,
            data_seed=run_seed,
            load_best_model_at_end=True,
            report_to="none",
        )

        trainer = DANNTrainer(
            model=model,
            data_collator=data_collator,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=processor.feature_extractor,
            total_steps=total_steps,
        )

        print("⚔️ 開始訓練...")
        try:
            trainer.train()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("⚠️ OOM！清除快取...")
                torch.cuda.empty_cache()
            else:
                raise e

        best_model_path = os.path.join(run_output_dir, "best_model")
        trainer.save_model(best_model_path)
        processor.save_pretrained(best_model_path)
        print(f"💾 Run {run_i} 最佳模型已儲存: {best_model_path}")

        # 同時存 shared_encoder 供 speaker probe 使用
        torch.save(
            model.shared_encoder.state_dict(),
            f"dann_finetune_B_shared_encoder_run_{run_i}.pth"
        )

        print(f"\n📊 Run {run_i} 測試集評估...")
        results = full_evaluation(trainer, test_dataset, OUTPUT_DIR, run_i)
        results["run"] = run_i
        all_results.append(results)
        print(f"Run {run_i} → Acc: {results['accuracy']:.4f} | F1: {results['f1']:.4f} | AUC: {results['auc']:.4f}")

        import gc
        del model, trainer
        torch.cuda.empty_cache()
        gc.collect()

    # 彙總
    print(f"\n{'='*60}")
    print(f"📈 DANN + Finetune Transformer — Scenario B — {TOTAL_RUNS} 次彙總")
    print(f"{'='*60}")

    results_df = pd.DataFrame(all_results)
    print(results_df.to_string(index=False))

    summary = {
        "accuracy_mean": results_df["accuracy"].mean(),
        "accuracy_std":  results_df["accuracy"].std(),
        "f1_mean":       results_df["f1"].mean(),
        "f1_std":        results_df["f1"].std(),
        "auc_mean":      results_df["auc"].mean(),
        "auc_std":       results_df["auc"].std(),
    }

    print(f"\n🎯 Accuracy : {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}")
    print(f"🎯 F1 (macro): {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
    print(f"📈 AUC       : {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}")

    summary_path = os.path.join(OUTPUT_DIR, "summary_5runs.csv")
    results_df.to_csv(summary_path, index=False)
    print(f"\n✅ 彙總結果已儲存至 {summary_path}")
    print("\n🏁 DANN + Finetune Transformer Scenario B 完成！")

