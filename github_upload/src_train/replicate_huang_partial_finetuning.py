"""
新版 File 2 — Scenario A (Strict Speaker Split)
==============================================
目標：使用與 HXS572 (6+7+8) 完全一致的訓練方法論
- 模型架構：Wav2Vec2ForSpeechClassification (mean pooling + frozen CNN)
- 訓練框架：HuggingFace Trainer (AdamW + linear LR scheduler)
- 資料處理：完整語音長度（不截斷）+ Wav2Vec2Processor.pad
- 評估：validation set 選最佳模型 + classification_report + confusion_matrix + ROC
- 可重現性：設定 random seed
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Union, Any
from math import sqrt

from datasets import Dataset as HFDataset
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2Config,
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model,
    AutoConfig,
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
#  設定區 — 依據實驗環境修改
# ============================================================
TRAIN_CSV = "./experiment_sisman_scientific/scenario_A_screening/train.csv"
TEST_CSV = "./experiment_sisman_scientific/scenario_A_screening/test.csv"
AUDIO_ROOT = "/export/fs05/hyeh10/depression/daic_5utt_full/merged_5"

MODEL_NAME = "facebook/wav2vec2-base"
OUTPUT_DIR = "./output_scenario_A_v2"

# 訓練超參數 — 與 6+7+8 pipeline 對齊
SEED = 42
NUM_EPOCHS = 20
LEARNING_RATE = 5e-5
PER_DEVICE_TRAIN_BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2
FP16 = torch.cuda.is_available()
EVAL_STEPS = 50
SAVE_STEPS = 50
LOGGING_STEPS = 50
SAVE_TOTAL_LIMIT = 3
WARMUP_RATIO = 0.1

# Label 對照表
LABEL_MAP = {"non": 0, "0": 0, 0: 0, "dep": 1, "1": 1, 1: 1}
LABEL_NAMES = ["non-depressed", "depressed"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
#  模型定義 — 完全對應 build_model.py (File 6)
# ============================================================

@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class Wav2Vec2ClassificationHead(nn.Module):
    """分類頭 — 與 build_model.py 完全一致"""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    """
    模型類 — 與 build_model.py (File 6) 完全一致
    設定 pooling_mode="mean" 時等同於 HuangForSpeechClassification
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config

        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)
        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy(self, hidden_states, mode="mean"):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception("Pooling methods: 'mean', 'max'")
        return outputs

    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# ============================================================
#  DataCollator — 完全對應 build_model.py (File 6)
# ============================================================

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator — 使用 Wav2Vec2Processor.pad 進行動態 padding
    與 build_model.py 完全一致
    """
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]
        label_features = [feature["labels"] for feature in features]

        d_type = (
            torch.long if isinstance(label_features[0], int) else torch.float
        )

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor(label_features, dtype=d_type)
        return batch


# ============================================================
#  compute_metrics — 完全對應 build_model.py (File 6)
# ============================================================

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    acc = accuracy_score(p.label_ids, preds)
    f1 = f1_score(p.label_ids, preds, average="macro")
    return {"accuracy": acc, "f1": f1}


# ============================================================
#  CTCTrainer — 完全對應 train.py (File 7)
# ============================================================

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast


class CTCTrainer(Trainer):
    """
    自訂 Trainer — 與 train.py (File 7) 的 CTCTrainer 完全一致
    支援 AMP 混合精度訓練
    """
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.use_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.detach()


# ============================================================
#  資料載入與預處理
# ============================================================

def load_audio_dataset(csv_path: str) -> HFDataset:
    """從 CSV 載入資料集，轉換為 HuggingFace Dataset 格式"""
    df = pd.read_csv(csv_path)
    print(f"📂 讀取 {csv_path}，共 {len(df)} 筆資料")

    records = []
    skipped = 0
    for _, row in df.iterrows():
        wav_path = os.path.join(AUDIO_ROOT, row["path"])
        raw_label = str(row["label"]).strip().lower()

        if raw_label not in LABEL_MAP:
            skipped += 1
            continue
        if not os.path.exists(wav_path):
            print(f"⚠️ 檔案不存在: {wav_path}")
            skipped += 1
            continue

        records.append({
            "path": wav_path,
            "label": LABEL_MAP[raw_label],
        })

    if skipped > 0:
        print(f"⚠️ 跳過 {skipped} 筆無效/不存在的資料")
    print(f"✅ 成功載入 {len(records)} 筆資料")

    return HFDataset.from_dict({
        "path": [r["path"] for r in records],
        "label": [r["label"] for r in records],
    })


def speech_file_to_array_fn(batch, processor):
    """
    將音訊檔案讀取並轉換為 array — 不截斷，使用完整語音長度
    """
    speech_array, sampling_rate = torchaudio.load(batch["path"])

    # 多聲道轉單聲道
    if speech_array.shape[0] > 1:
        speech_array = torch.mean(speech_array, dim=0, keepdim=True)
    speech_array = speech_array.squeeze().numpy()

    # 重取樣至 16kHz
    if sampling_rate != 16000:
        import librosa
        speech_array = librosa.resample(
            speech_array, orig_sr=sampling_rate, target_sr=16000
        )

    batch["speech"] = speech_array
    return batch


def preprocess_function(batch, processor):
    """將 speech array 轉為 input_values"""
    result = processor(
        batch["speech"],
        sampling_rate=16000,
        return_tensors="np",
        padding=False,
    )
    batch["input_values"] = result.input_values[0]
    batch["labels"] = batch["label"]
    return batch


def split_train_valid(dataset: HFDataset, valid_ratio: float = 0.15, seed: int = 42):
    """
    從訓練集中分出驗證集 — 對應 6+7+8 的 train/valid/test 三分結構
    """
    split = dataset.train_test_split(test_size=valid_ratio, seed=seed)
    return split["train"], split["test"]


# ============================================================
#  評估與報告 — 對應 evaluate.py (File 8)
# ============================================================

def full_evaluation(trainer, test_dataset, config_obj, output_dir):
    """
    完整評估 — 與 evaluate.py (File 8) 一致
    包含：classification_report + confusion_matrix + ROC curve
    """
    predictions = trainer.predict(test_dataset)
    preds = predictions.predictions
    if isinstance(preds, tuple):
        preds = preds[0]
    y_pred = np.argmax(preds, axis=1)
    y_true = predictions.label_ids

    # Classification Report
    print("\n" + "=" * 60)
    print("📊 Classification Report")
    print("=" * 60)
    report = classification_report(
        y_true, y_pred,
        target_names=LABEL_NAMES,
        zero_division=0,
        output_dict=True,
    )
    report_df = pd.DataFrame(report).transpose()
    print(report_df)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=LABEL_NAMES, columns=LABEL_NAMES)
    print("\n📊 Confusion Matrix:")
    print(cm_df)

    # MSE / RMSE
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    report_df["MSE"] = mse
    report_df["RMSE"] = rmse

    # ROC Curve (binary)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    # 儲存結果
    results_path = os.path.join(output_dir, "results")
    os.makedirs(results_path, exist_ok=True)

    report_df.to_csv(os.path.join(results_path, "clsf_report.csv"), sep="\t")
    cm_df.to_csv(os.path.join(results_path, "conf_matrix.csv"), sep="\t")

    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2,
             label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Scenario A")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(results_path, "roc_curve.png"))
    plt.close()
    print(f"\n✅ 結果已儲存至 {results_path}")

    # 額外輸出 accuracy 與 F1
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\n🎯 Test Accuracy: {acc:.4f}")
    print(f"🎯 Test F1 (macro): {f1:.4f}")
    print(f"📈 AUC: {roc_auc:.4f}")

    return {"accuracy": acc, "f1": f1, "auc": roc_auc}


# ============================================================
#  主訓練流程
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Scenario A — 使用 6+7+8 等效訓練管線")
    print("   模型: Wav2Vec2ForSpeechClassification (mean pooling)")
    print("   CNN: Frozen (feature extractor)")
    print("   分類: Binary (depressed / non-depressed)")
    print("   音訊: 完整長度（不截斷）")
    print("=" * 60)

    # 1. 設定 seed — 對應 train.py 的 set_seed()
    set_seed(SEED)
    print(f"🎲 Random seed: {SEED}")

    # 2. 載入 processor 與 config
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    config = Wav2Vec2Config.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        final_dropout=0.1,
        pooling_mode="mean",  # 明確設定 mean pooling
    )

    # 3. 載入模型
    model = Wav2Vec2ForSpeechClassification.from_pretrained(
        MODEL_NAME, config=config
    )

    # 4. 凍結 feature extractor (CNN) — 對應 train.py 的 freeze_feature_extractor
    model.freeze_feature_extractor()
    print("❄️ Feature Extractor (CNN) 已凍結")
    print(f"🔍 Transformer 第一層梯度: "
          f"{model.wav2vec2.encoder.layers[0].attention.k_proj.weight.requires_grad}")

    # 5. 載入資料
    print("\n📦 載入資料集...")
    train_dataset_full = load_audio_dataset(TRAIN_CSV)
    test_dataset = load_audio_dataset(TEST_CSV)

    # 6. 預處理音訊 — 使用完整語音長度（不截斷）
    print("\n🔊 預處理音訊檔案...")
    train_dataset_full = train_dataset_full.map(
        speech_file_to_array_fn,
        fn_kwargs={"processor": processor},
    )
    test_dataset = test_dataset.map(
        speech_file_to_array_fn,
        fn_kwargs={"processor": processor},
    )

    # 7. 轉換為 input_values
    print("🔄 轉換為模型輸入格式...")
    train_dataset_full = train_dataset_full.map(
        preprocess_function,
        fn_kwargs={"processor": processor},
    )
    test_dataset = test_dataset.map(
        preprocess_function,
        fn_kwargs={"processor": processor},
    )

    # 8. 分割 train/valid — 對應 6+7+8 的三分結構
    train_dataset, eval_dataset = split_train_valid(
        train_dataset_full, valid_ratio=0.15, seed=SEED
    )
    print(f"📊 Train: {len(train_dataset)} 筆 | Valid: {len(eval_dataset)} 筆 | "
          f"Test: {len(test_dataset)} 筆")

    # 9. 設定 DataCollator — 對應 build_model.py 的 DataCollatorCTCWithPadding
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    # 10. 設定 TrainingArguments — 對應 train.py (File 7)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
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
        seed=SEED,
        data_seed=SEED,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        warmup_ratio=WARMUP_RATIO,
        report_to="none",
    )

    # 11. 初始化 Trainer — 使用 CTCTrainer (對應 train.py)
    trainer = CTCTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.feature_extractor,
    )

    # 12. 開始訓練
    print("\n⚔️ 開始訓練...")
    try:
        trainer.train()
    except RuntimeError as exception:
        if "out of memory" in str(exception):
            print("⚠️ GPU 記憶體不足！嘗試清除快取...")
            if hasattr(torch.cuda, "empty_cache"):
                torch.cuda.empty_cache()
        else:
            raise exception

    # 13. 儲存最佳模型
    best_model_path = os.path.join(OUTPUT_DIR, "best_model")
    trainer.save_model(best_model_path)
    processor.save_pretrained(best_model_path)
    print(f"\n💾 最佳模型已儲存至: {best_model_path}")

    # 14. 完整評估 — 對應 evaluate.py (File 8)
    print("\n📊 在測試集上進行完整評估...")
    results = full_evaluation(trainer, test_dataset, config, OUTPUT_DIR)

    print("\n🏁 Scenario A 實驗完成！")
