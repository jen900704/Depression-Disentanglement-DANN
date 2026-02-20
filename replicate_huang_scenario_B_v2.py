"""
File 3 — Scenario B (Partial Speaker Overlap) ── v3 速度 & OOM 修正版
=======================================================================
對齊 File 6+7+8 (build_model.py / train.py / evaluate.py) 的訓練管線
- 模型架構：Wav2Vec2ForSpeechClassification (mean pooling + frozen CNN)
- 訓練框架：HuggingFace Trainer (AdamW + linear LR scheduler)
- checkpoint 選擇：eval_loss（對齊 yaml 預設，不指定 metric_for_best_model）
- compute_metrics：accuracy only（對齊 build_model.py File 6）
- return_attention_mask：False（對齊 yaml）
- 五次實驗迴圈，最後輸出平均與標準差

v3 修改摘要（速度 & OOM 防護）：
  ① eval_steps / save_steps / logging_steps：10 → 100（減少 eval 次數 10x）
  ② dataloader_num_workers=0 + pin_memory=False：避免多 worker 佔用額外 RAM
  ③ OOM 捕捉：whole training loop + empty_cache + gc.collect()
  ④ 每次 eval / predict 前後加 torch.cuda.empty_cache() 防止評估時 OOM
  ⑤ 每次 run 結束 del model/trainer + empty_cache，防止 5 次實驗累積記憶體
  ⑥ gradient_checkpointing 保留（省 GPU 記憶體，代價是輕微減速）
  ⑦ 保留 fp16=True（CUDA 環境下省記憶體且加速）
  ⑧ LengthGroupedSampler：訓練時按音訊長度排序，減少 padding 浪費 → 防 OOM
  ⑨ CTCTrainer 覆寫 get_train_dataloader / get_eval_dataloader 使用長度排序
  ※ 音訊長度不截斷，完整保留原始資料

對齊 daic-c2-rmse-roc.yaml 設定（不動）：
  seed=103 | lr=1e-5 | epochs=10 | batch=1+grad_accum=8 (eff=8)
  freeze_feature_extractor=True | pooling_mode=mean
  return_attention_mask=False | metric_for_best_model → eval_loss (預設)
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
from math import sqrt

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
#  設定區 — 路徑為 scenario_B_monitoring，其餘對齊 yaml
# ============================================================
TRAIN_CSV  = "./experiment_sisman_scientific/scenario_B_monitoring/train.csv"
TEST_CSV   = "./experiment_sisman_scientific/scenario_B_monitoring/test.csv"
AUDIO_ROOT = "/export/fs05/hyeh10/depression/daic_5utt_full/merged_5"

MODEL_NAME = "facebook/wav2vec2-base"
OUTPUT_DIR = "./output_scenario_B_v2"

SEED                         = 103
NUM_EPOCHS                   = 10
LEARNING_RATE                = 1e-5
PER_DEVICE_TRAIN_BATCH_SIZE  = 1
PER_DEVICE_EVAL_BATCH_SIZE   = 1
GRADIENT_ACCUMULATION_STEPS  = 8
FP16                         = torch.cuda.is_available()
# ▼ v3 修正：eval 頻率降低 10 倍，大幅縮短總訓練時間
EVAL_STEPS                   = 100
SAVE_STEPS                   = 100
LOGGING_STEPS                = 50
SAVE_TOTAL_LIMIT             = 2

TOTAL_RUNS = 5  # 五次實驗取平均

LABEL_MAP   = {"non": 0, "0": 0, 0: 0, "dep": 1, "1": 1, 1: 1}
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
    def __init__(self, config):
        super().__init__()
        self.dense    = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout  = nn.Dropout(config.final_dropout)
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
    def __init__(self, config):
        super().__init__(config)
        self.num_labels   = config.num_labels
        self.pooling_mode = getattr(config, "pooling_mode", "mean")
        self.config       = config

        self.wav2vec2   = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)
        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy(self, hidden_states, mode="mean"):
        if mode == "mean":
            return torch.mean(hidden_states, dim=1)
        elif mode == "max":
            return torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception("Pooling methods: 'mean', 'max'")

    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
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
    processor:                 Wav2Vec2Processor
    padding:                   Union[bool, str] = True
    max_length:                Optional[int] = None
    max_length_labels:         Optional[int] = None
    pad_to_multiple_of:        Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [f["labels"] for f in features]

        d_type = torch.long if isinstance(label_features[0], int) else torch.float

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
#  compute_metrics — 對齊 build_model.py (File 6)
#  回傳 accuracy only，邏輯與 File 6 一致
# ============================================================

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}


# ============================================================
#  CTCTrainer — 完全對應 train.py (File 7)
# ============================================================

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True


def _length_sorted_indices(dataset) -> list:
    """
    回傳按 input_values 長度排序的 index 清單。
    長度相近的樣本會被排在一起，大幅減少 batch 內 padding，防止 OOM。
    """
    lengths = [len(dataset[i]["input_values"]) for i in range(len(dataset))]
    return sorted(range(len(lengths)), key=lambda i: lengths[i])


class CTCTrainer(Trainer):
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
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

    def get_train_dataloader(self):
        """
        ▼ v3 OOM 防護：訓練時按音訊長度排序（長度相近的排在一起），
          大幅減少 batch 內 padding 大小，是 A100 上最有效的非截斷 OOM 對策。
          batch_size=1 時每筆獨立，padding 由 collator 做到 batch 內最長，
          排序後最長音訊不會與最短混在同一 batch，避免暴增的 padding tensor。
        """
        from torch.utils.data import DataLoader, Subset, SequentialSampler
        dataset = self.train_dataset
        sorted_indices = _length_sorted_indices(dataset)
        sorted_dataset = Subset(dataset, sorted_indices)
        return DataLoader(
            sorted_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=SequentialSampler(sorted_dataset),
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        """
        ▼ v3 OOM 防護：評估時同樣按長度排序，防止超長音訊在評估中觸發 OOM。
        """
        from torch.utils.data import DataLoader, Subset, SequentialSampler
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        sorted_indices = _length_sorted_indices(dataset)
        sorted_dataset = Subset(dataset, sorted_indices)
        return DataLoader(
            sorted_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            sampler=SequentialSampler(sorted_dataset),
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


# ============================================================
#  資料載入與預處理
# ============================================================

def load_audio_dataset(csv_path: str) -> HFDataset:
    df = pd.read_csv(csv_path)
    print(f"📂 讀取 {csv_path}，共 {len(df)} 筆資料")

    records, skipped = [], 0
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

        records.append({"path": wav_path, "label": LABEL_MAP[raw_label]})

    if skipped > 0:
        print(f"⚠️ 跳過 {skipped} 筆無效/不存在的資料")
    print(f"✅ 成功載入 {len(records)} 筆資料")

    return HFDataset.from_dict({
        "path":  [r["path"]  for r in records],
        "label": [r["label"] for r in records],
    })


def speech_file_to_array_fn(batch, processor):
    speech_array, sampling_rate = torchaudio.load(batch["path"])

    if speech_array.shape[0] > 1:
        speech_array = torch.mean(speech_array, dim=0, keepdim=True)
    speech_array = speech_array.squeeze().numpy()

    if sampling_rate != 16000:
        import librosa
        speech_array = librosa.resample(
            speech_array, orig_sr=sampling_rate, target_sr=16000
        )

    batch["speech"] = speech_array
    return batch


def preprocess_function(batch, processor):
    """return_attention_mask=False — 對齊 yaml"""
    result = processor(
        batch["speech"],
        sampling_rate=16000,
        return_tensors="np",
        padding=False,
        return_attention_mask=False,  # 對齊 yaml: return_attention_mask: False
    )
    batch["input_values"] = result.input_values[0]
    batch["labels"]       = batch["label"]
    return batch


# ============================================================
#  評估與報告 — 對應 evaluate.py (File 8)
# ============================================================

def full_evaluation(trainer, test_dataset, output_dir, run_i):
    predictions = trainer.predict(test_dataset)
    preds = predictions.predictions
    if isinstance(preds, tuple):
        preds = preds[0]
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
    plt.title(f"ROC Curve - Scenario B (Run {run_i})")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(results_path, "roc_curve.png"))
    plt.close()
    print(f"✅ Run {run_i} 結果已儲存至 {results_path}")

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro")
    return {"accuracy": acc, "f1": f1, "auc": roc_auc}


# ============================================================
#  主訓練流程 — 五次迴圈取平均
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Scenario B — 對齊 daic-c2-rmse-roc.yaml 訓練管線")
    print("   模型: Wav2Vec2ForSpeechClassification (mean pooling)")
    print("   CNN: Frozen | 分類: Binary | seed: 103 | lr: 1e-5")
    print("   checkpoint 選擇: eval_loss (對齊 yaml 預設)")
    print(f"   實驗次數: {TOTAL_RUNS} 次，最後輸出平均與標準差")
    print("=" * 60)

    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    map_kwargs = {"fn_kwargs": {"processor": processor}}

    # 資料只載入一次
    print("\n📦 載入並預處理資料集（只執行一次）...")
    train_dataset_raw = load_audio_dataset(TRAIN_CSV)
    test_dataset_raw  = load_audio_dataset(TEST_CSV)

    train_dataset_raw = train_dataset_raw.map(speech_file_to_array_fn, **map_kwargs)
    test_dataset_raw  = test_dataset_raw.map(speech_file_to_array_fn,  **map_kwargs)

    train_dataset = train_dataset_raw.map(preprocess_function, **map_kwargs)
    test_dataset  = test_dataset_raw.map(preprocess_function,  **map_kwargs)

    print(f"📊 Train: {len(train_dataset)} 筆 | Test: {len(test_dataset)} 筆")

    data_collator_obj = DataCollatorCTCWithPadding(processor=processor, padding=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = []

    for run_i in range(1, TOTAL_RUNS + 1):
        print(f"\n{'='*60}")
        print(f"🎬 開始第 {run_i} / {TOTAL_RUNS} 次實驗")
        print(f"{'='*60}")

        # 每次 seed 遞增確保隨機性：103, 104, 105, 106, 107
        run_seed = SEED + run_i - 1
        set_seed(run_seed)
        print(f"🎲 Run {run_i} seed: {run_seed}")

        # 每次重新初始化模型
        config = Wav2Vec2Config.from_pretrained(
            MODEL_NAME,
            num_labels=2,
            final_dropout=0.1,
            pooling_mode="mean",   # 對齊 yaml: pooling_mode: mean
        )
        model = Wav2Vec2ForSpeechClassification.from_pretrained(MODEL_NAME, config=config)
        model.freeze_feature_extractor()  # 對齊 yaml: freeze_feature_extractor: True
        print(f"❄️ Feature Extractor (CNN) 已凍結")

        run_output_dir = os.path.join(OUTPUT_DIR, f"run_{run_i}")
        os.makedirs(run_output_dir, exist_ok=True)

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
            data_seed=run_seed,
            load_best_model_at_end=True,
            # metric_for_best_model 不設定 → 預設 eval_loss（對齊 yaml）
            report_to="none",
            gradient_checkpointing=True,
            # ▼ v3：關閉多 worker，避免 DataLoader 佔用額外 RAM
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
        )

        trainer = CTCTrainer(
            model=model,
            data_collator=data_collator_obj,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,   # 無獨立 valid set，對齊論文做法
            tokenizer=processor.feature_extractor,
        )

        print("⚔️ 開始訓練...")
        # ▼ v3：完整 OOM 防護 — 捕捉整個 train()，清 cache 後提示
        try:
            trainer.train()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("\n⚠️  OOM！嘗試清除 GPU cache 後繼續評估（訓練未完整完成）...")
                torch.cuda.empty_cache()
                import gc; gc.collect()
            else:
                raise e

        best_model_path = os.path.join(run_output_dir, "best_model")
        trainer.save_model(best_model_path)
        processor.save_pretrained(best_model_path)
        print(f"💾 Run {run_i} 最佳模型已儲存至: {best_model_path}")

        # ▼ v3：評估前釋放 GPU 記憶體
        torch.cuda.empty_cache()
        import gc; gc.collect()
        print(f"\n📊 Run {run_i} 測試集評估...")
        results = full_evaluation(trainer, test_dataset, OUTPUT_DIR, run_i)
        results["run"] = run_i
        all_results.append(results)
        print(f"Run {run_i} → Acc: {results['accuracy']:.4f} | F1: {results['f1']:.4f} | AUC: {results['auc']:.4f}")

        # ▼ v3：每次 run 結束後釋放模型記憶體，避免多次實驗累積 OOM
        del model, trainer
        torch.cuda.empty_cache()
        import gc; gc.collect()

    # ============================================================
    #  輸出五次平均與標準差
    # ============================================================
    print(f"\n{'='*60}")
    print(f"📈 Scenario B — {TOTAL_RUNS} 次實驗彙總")
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
    print("\n🏁 Scenario B 全部實驗完成！")
