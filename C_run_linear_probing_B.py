import os
import pandas as pd
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ==========================================
# 1. 設定區 (Config) - 對齊 Scenario A
# ==========================================

TRAIN_CSV_PATH = "./experiment_sisman_scientific/scenario_B_monitoring/train.csv"
TEST_CSV_PATH = "./experiment_sisman_scientific/scenario_B_monitoring/test.csv"
AUDIO_ROOT = "/export/fs05/hyeh10/depression/daic_5utt_full/merged_5"

MODEL_NAME = "facebook/wav2vec2-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOTAL_RUNS = 5

print(f"🖥️ 運算設備: {DEVICE}")

# ==========================================
# 2. 資料處理工具 (直接從音檔抽取，確保與 DANN A 對齊)
# ==========================================
def prepare_data_for_probing(csv_path, processor, model):
    df = pd.read_csv(csv_path)
    print(f"📂 正在處理 {csv_path} (共 {len(df)} 筆)...")
    
    features_list = []
    labels_list = []
    label_map = {'dep': 1, '1': 1, 1: 1, 'non': 0, '0': 0, 0: 0}
    
    model.eval()
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Features"):
            wav_path = os.path.join(AUDIO_ROOT, row['path'])
            try:
                waveform, sample_rate = torchaudio.load(wav_path)
                if sample_rate != 16000:
                    waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
                if waveform.shape[0] > 1: 
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                raw_label = str(row['label']).strip().lower()
                if raw_label not in label_map: continue
                final_label = label_map[raw_label]

                # 提取 Wav2Vec2 凍結特徵
                inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                embeddings = model(**inputs).last_hidden_state.mean(dim=1).cpu()
                
                features_list.append(embeddings)
                labels_list.append(final_label)
                
            except Exception as e:
                print(f"⚠️ Error: {wav_path} -> {e}")
                continue

    X = torch.cat(features_list, dim=0).numpy()
    y = np.array(labels_list)
    return X, y

# ==========================================
# 3. 主程式執行
# ==========================================
if __name__ == "__main__":
    print("🧠 載入 Wav2Vec2 模型提取深層特徵...")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    w2v_model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(DEVICE)
    w2v_model.eval() # 嚴格凍結 Backbone，不更新權重

    print("\n⏳ 抽取【訓練集】特徵...")
    X_train, y_train = prepare_data_for_probing(TRAIN_CSV_PATH, processor, w2v_model)
    print("\n⏳ 抽取【測試集】特徵...")
    X_test, y_test = prepare_data_for_probing(TEST_CSV_PATH, processor, w2v_model)

    print(f"\n✅ 特徵抽取完畢！形狀: X_train={X_train.shape}, X_test={X_test.shape}")

    all_accs = []
    all_f1s = []

    print(f"\n🚀 開始執行 Linear Probing ({TOTAL_RUNS} 次實驗)...")
    
    # 4. 執行 5 次迴圈
    for run_i in range(1, TOTAL_RUNS + 1):
        print(f"\n{'-'*30}")
        print(f"🎬 Run {run_i} / {TOTAL_RUNS}")
        print(f"{'-'*30}")
        
        # 設定 Random State 確保每次訓練的隨機性不同
        current_seed = 42 + run_i
        
        clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=current_seed)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        
        all_accs.append(acc)
        all_f1s.append(f1)
        
        print(f"Run {run_i} -> Acc: {acc:.4f} | F1: {f1:.4f}")

    # ==========================================
    # 5. 輸出五次平均與標準差
    # ==========================================
    print(f"\n{'='*50}")
    print(f"🏆 Scenario B: Linear Probing ({TOTAL_RUNS} runs) 最終結果")
    print(f"{'='*50}")
    
    accs_np = np.array(all_accs)
    f1s_np = np.array(all_f1s)
    
    print(f"🎯 平均 憂鬱症 Acc (Dep Acc): {accs_np.mean():.4f} ± {accs_np.std():.4f}")
    print(f"🎯 平均 憂鬱症 F1  (Dep F1) : {f1s_np.mean():.4f} ± {f1s_np.std():.4f}")