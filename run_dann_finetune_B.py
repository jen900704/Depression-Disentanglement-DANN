"""
DANN + Fine-tuned Transformer — Scenario B (Longitudinal)
===============================================================================
修正版 v6：修復 remove_unused_columns 導致的 KeyError，保留 Spk Acc，並恢復合理的 eval_steps
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Union, Any
from torch.autograd import Function
from datasets import Dataset as HFDataset
from transformers import (Wav2Vec2Processor, Wav2Vec2Config, Wav2Vec2PreTrainedModel, Wav2Vec2Model, Trainer, TrainingArguments, EvalPrediction, set_seed)
from transformers.file_utils import ModelOutput
from sklearn.metrics import accuracy_score, f1_score, classification_report

TRAIN_CSV  = "./experiment_sisman_scientific/scenario_B_monitoring/train.csv"
TEST_CSV   = "./experiment_sisman_scientific/scenario_B_monitoring/test.csv"
MODEL_NAME = "facebook/wav2vec2-base"
OUTPUT_DIR = "./output_dann_finetune_B_v6"

SEED, NUM_EPOCHS, LEARNING_RATE, PER_DEVICE_TRAIN_BATCH_SIZE = 103, 10, 1e-5, 4
GRADIENT_ACCUMULATION_STEPS, EVAL_STEPS, SAVE_STEPS, TOTAL_RUNS = 2, 300, 300, 5
LABEL_MAP, DEVICE = {"non": 0, "0": 0, 0: 0, "dep": 1, "1": 1, 1: 1}, "cuda" if torch.cuda.is_available() else "cpu"

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
        self.shared_encoder = nn.Sequential(nn.Linear(config.hidden_size, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3))
        self.dep_classifier = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, config.num_labels))
        self.spk_classifier = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, getattr(config, "num_speakers", 38)))
        self.grl = GradientReversalLayer()
        self.init_weights()

    def freeze_feature_extractor(self): self.wav2vec2.feature_extractor._freeze_parameters()

    def forward(self, input_values, attention_mask=None, return_dict=None, labels=None, speaker_labels=None, alpha=1.0):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        shared = self.shared_encoder(torch.mean(outputs[0], dim=1))
        dep_logits = self.dep_classifier(shared)
        spk_logits = self.spk_classifier(self.grl(shared, alpha))

        loss, loss_dep, loss_spk = None, None, None
        if labels is not None:
            loss_dep = nn.CrossEntropyLoss()(dep_logits.view(-1, self.config.num_labels), labels.view(-1))
            loss = loss_dep
        if speaker_labels is not None:
            mask = speaker_labels >= 0
            if mask.sum() > 0:
                loss_spk = nn.CrossEntropyLoss()(spk_logits[mask].view(-1, spk_logits.size(-1)), speaker_labels[mask].view(-1))
                loss = loss_dep + loss_spk if loss_dep is not None else loss_spk

        if not (return_dict if return_dict is not None else self.config.use_return_dict): return ((loss,) + (dep_logits, spk_logits)) if loss is not None else (dep_logits, spk_logits)
        return DANNOutput(loss=loss, logits=dep_logits, spk_logits=spk_logits)

@dataclass
class DataCollatorDANN:
    processor: Wav2Vec2Processor
    def __call__(self, features):
        batch = self.processor.pad([{"input_values": f["input_values"]} for f in features], return_tensors="pt")
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=torch.long)
        batch["speaker_labels"] = torch.tensor([f["speaker_label"] for f in features], dtype=torch.long)
        return batch

def compute_metrics(p: EvalPrediction):
    preds = p.predictions
    if isinstance(preds, tuple):
        for arr in preds:
            if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] == 2: preds = arr; break
    true_labels = p.label_ids[0] if isinstance(p.label_ids, tuple) else p.label_ids
    preds = np.argmax(preds, axis=1)
    return {"accuracy": accuracy_score(true_labels, preds), "f1": f1_score(true_labels, preds, average="macro")}

class DANNTrainer(Trainer):
    def __init__(self, *args, total_steps=1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_steps, self.current_step = total_steps, 0
    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)
        inputs["alpha"] = 2.0 / (1.0 + np.exp(-10 * (self.current_step / max(self.total_steps, 1)))) - 1
        self.current_step += 1
        with torch.amp.autocast("cuda") if self.args.fp16 else torch.autocast("cpu", enabled=False): loss = self.compute_loss(model, inputs) / self.args.gradient_accumulation_steps
        if self.args.fp16 and hasattr(self, "scaler"): self.scaler.scale(loss).backward()
        else: loss.backward()
        return loss.detach()

def load_audio_dataset(csv_path, speaker_map):
    df = pd.read_csv(csv_path)
    records = [{"path": r["path"], "label": LABEL_MAP[str(r["label"]).strip().lower()], "speaker_label": speaker_map.get(os.path.basename(r["path"]).split('_')[0], -1)} for _, r in df.iterrows() if str(r["label"]).strip().lower() in LABEL_MAP and os.path.exists(r["path"])]
    return HFDataset.from_dict({"path": [r["path"] for r in records], "label": [r["label"] for r in records], "speaker_label": [r["speaker_label"] for r in records]})

def preprocess_function(batch, processor):
    speech, sr = torchaudio.load(batch["path"])
    if speech.shape[0] > 1: speech = torch.mean(speech, dim=0, keepdim=True)
    if sr != 16000: speech = torchaudio.transforms.Resample(sr, 16000)(speech)
    batch["input_values"] = processor(speech.squeeze().numpy(), sampling_rate=16000, return_tensors="np", padding=False).input_values[0]
    return batch

def full_evaluation(trainer, test_dataset, output_dir, run_i, num_speakers):
    pred_obj = trainer.predict(test_dataset)
    dep_preds, spk_preds = None, None
    if isinstance(pred_obj.predictions, tuple):
        for arr in pred_obj.predictions:
            if isinstance(arr, np.ndarray) and arr.ndim == 2:
                if arr.shape[1] == 2: dep_preds = arr
                elif arr.shape[1] == num_speakers: spk_preds = arr
    y_true, spk_true = (pred_obj.label_ids[0], pred_obj.label_ids[1]) if isinstance(pred_obj.label_ids, tuple) else (pred_obj.label_ids, None)
    
    y_pred = np.argmax(dep_preds, axis=1)
    acc, f1 = accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro")
    spk_acc = accuracy_score(spk_true, np.argmax(spk_preds, axis=1)) if spk_preds is not None and spk_true is not None else 0.0
    
    os.makedirs(f"{output_dir}/results_run_{run_i}", exist_ok=True)
    pd.DataFrame(classification_report(y_true, y_pred, zero_division=0, output_dict=True)).transpose().to_csv(f"{output_dir}/results_run_{run_i}/clsf_report.csv", sep="\t")
    return {"accuracy": acc, "f1": f1, "spk_acc": spk_acc}

if __name__ == "__main__":
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    speaker_map = {spk: i for i, spk in enumerate(sorted(pd.read_csv(TEST_CSV)['path'].apply(lambda x: os.path.basename(x).split('_')[0]).unique()))}
    num_speakers = len(speaker_map)
    train_ds = load_audio_dataset(TRAIN_CSV, speaker_map).map(lambda b: preprocess_function(b, processor))
    test_ds = load_audio_dataset(TEST_CSV, speaker_map).map(lambda b: preprocess_function(b, processor))

    all_results = []
    for run_i in range(1, TOTAL_RUNS + 1):
        set_seed(SEED + run_i - 1)
        config = Wav2Vec2Config.from_pretrained(MODEL_NAME, num_labels=2, num_speakers=num_speakers)
        model = Wav2Vec2DANNFinetune.from_pretrained(MODEL_NAME, config=config)
        model.freeze_feature_extractor()

        training_args = TrainingArguments(
            output_dir=f"{OUTPUT_DIR}/run_{run_i}", per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS, evaluation_strategy="steps", num_train_epochs=NUM_EPOCHS,
            save_steps=SAVE_STEPS, eval_steps=EVAL_STEPS, learning_rate=LEARNING_RATE, load_best_model_at_end=True,
            metric_for_best_model="f1", dataloader_drop_last=True, report_to="none",
            remove_unused_columns=False  # 🔥 關鍵修復：防止 Trainer 刪掉 label 和 speaker_label
        )
        
        trainer = DANNTrainer(model=model, data_collator=DataCollatorDANN(processor=processor), args=training_args,
                              compute_metrics=compute_metrics, train_dataset=train_ds, eval_dataset=test_ds,
                              total_steps=(len(train_ds) // (PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)) * NUM_EPOCHS)
        trainer.train()
        res = full_evaluation(trainer, test_ds, OUTPUT_DIR, run_i, num_speakers)
        print(f"Run {run_i} → Acc: {res['accuracy']:.4f} | F1: {res['f1']:.4f} | Spk Acc: {res['spk_acc']:.4f}")
        all_results.append(res)
        
    df = pd.DataFrame(all_results)
    print(f"\n🎯 Avg F1: {df['f1'].mean():.4f} ± {df['f1'].std():.4f} | Avg Spk Acc: {df['spk_acc'].mean():.4f} ± {df['spk_acc'].std():.4f}")
