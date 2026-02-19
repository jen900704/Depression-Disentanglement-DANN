import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torchaudio
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Union
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2Config, Wav2Vec2Model, Wav2Vec2PreTrainedModel
from transformers.file_utils import ModelOutput

# ================= 🔧 1. 設定區 (Scenario B - Random Split) =================
# 🔥 修改這裡：指向 Scenario B 的資料夾
TRAIN_CSV = "./experiment_sisman_scientific/scenario_B_monitoring/train.csv"
TEST_CSV = "./experiment_sisman_scientific/scenario_B_monitoring/test.csv"
AUDIO_ROOT = "/export/fs05/hyeh10/depression/daic_5utt_full/merged_5"

MODEL_NAME = "facebook/wav2vec2-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 保持跟 Scenario A 一樣的設定
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 5e-5 

# ================= 🧠 2. Huang et al. 架構 (Partial Fine-tuning) =================
# ... (這裡的 class 定義完全不用動，跟上一個腳本一模一樣) ...

@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None

class Wav2Vec2ClassificationHead(nn.Module):
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

class HuangForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = "mean" 
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)
        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def forward(self, input_values, attention_mask=None, labels=None):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return SpeechClassifierOutput(loss=loss, logits=logits)

# ================= 📂 3. 資料載入 =================

class ScenarioBDataset(Dataset):
    def __init__(self, csv_path, processor):
        self.df = pd.read_csv(csv_path)
        self.processor = processor
        self.label_map = {'non': 0, '0': 0, 0: 0, 'dep': 1, '1': 1, 1: 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav_path = os.path.join(AUDIO_ROOT, row['path'])
        
        try:
            speech, sr = torchaudio.load(wav_path)
            if sr != 16000:
                speech = torchaudio.transforms.Resample(sr, 16000)(speech)
            
            # 🔥 保持跟 Scenario A 一樣的長度限制 (8秒)
            MAX_LEN = 16000 * 8 
            if speech.shape[1] > MAX_LEN:
                 speech = speech[:, :MAX_LEN]
    
            input_values = self.processor(speech.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_values[0]
            
            raw_label = str(row['label']).strip().lower()
            label_int = self.label_map.get(raw_label, 0)
            label = torch.tensor(label_int, dtype=torch.long)
            
            return {"input_values": input_values, "labels": label}
        except Exception as e:
            print(f"Error loading {wav_path}: {e}")
            return {"input_values": torch.zeros(16000), "labels": torch.tensor(0, dtype=torch.long)}

def collate_fn(batch):
    input_values = [item['input_values'] for item in batch]
    labels = [item['labels'] for item in batch]
    input_values = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True)
    labels = torch.stack(labels)
    return {"input_values": input_values, "labels": labels}

# ================= 🚀 4. 執行實驗 =================

if __name__ == "__main__":
    print(f"🚀 啟動實驗: 複製 Huang et al. 架構 (Partial Fine-tuning) 在 Scenario B (Random Split)")
    
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    config = Wav2Vec2Config.from_pretrained(MODEL_NAME, num_labels=2, final_dropout=0.1)
    config.gradient_checkpointing = True 

    model = HuangForSpeechClassification.from_pretrained(MODEL_NAME, config=config).to(DEVICE)
    model.freeze_feature_extractor()
    print("❄️ Wav2Vec2 Feature Encoder (CNN) 已凍結。Transformer Layers 保持可訓練！")
    
    train_ds = ScenarioBDataset(TRAIN_CSV, processor)
    test_ds = ScenarioBDataset(TEST_CSV, processor)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            inputs = batch['input_values'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 評估
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch['input_values'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                logits = model(inputs).logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        print(f"✅ Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f} | Test Acc: {acc:.4f} | Test F1: {f1:.4f}")
        
    print("\n🏁 實驗結束。")
    print(classification_report(all_labels, all_preds))