'''
Hand Rehabilitation Dataset Training Script (參數調整版)

功能說明：
    此腳本用於訓練手部復健動作識別模型，支援參數調整和模型評估。
    包含完整的訓練流程、驗證集評估、獨立測試集評估（含 label0 邏輯）。

主要特性：
    1. 可調整的參數配置（學習率、批次大小、Dropout、模型架構等）
    2. 支援 StandardScaler 或 Normalizer 資料正規化
    3. 類別權重平衡（處理不平衡資料集）
    4. Early Stopping 和 Learning Rate Scheduler
    5. Label0 邏輯：低信心度預測自動分類為 label0
    6. 詳細的診斷資訊和測試報告

可調整參數：
    1. 學習率 (learning_rate): 0.001, 0.0005, 0.0001
    2. Batch size: 16, 32, 64
    3. Dropout rate: 0.3, 0.4, 0.5
    4. 模型架構: 不同層數和神經元數量
    5. 正規化方法: Normalizer vs StandardScaler
    6. 訓練策略: Early Stopping, Model Checkpoint, LR Scheduler
    7. 信心度閾值 (confidence_threshold): 用於 label0 分類
       - 最大機率低於此閾值時，標記為 label0
       - 同時使用熵（entropy）來判斷預測不確定性
       - 熵高於 (1 - threshold) 時，也標記為 label0

資料集說明：
    - 訓練集: data/rehab_dataset_keyframes.csv（標籤範圍：1-7）
    - 測試集: data/rehab_dataset_keyframes_testing_set.csv（標籤範圍：0-7，0 為 label0）

輸出檔案：
    - 模型: models/rehab_model_tuned/rehab_model_tuned_{timestamp}.h5
    - 配置: models/rehab_model_tuned/config_{timestamp}.json
    - Scaler: models/rehab_model_tuned/scaler_{timestamp}.pkl（如果使用 StandardScaler）
    - 驗證集混淆矩陣: models/confusion_matrices/rehab_model_tuned_{timestamp}.txt
    - 測試集預測結果: models/test_results/test_predictions_tuned_{timestamp}.csv
    - 測試集報告: models/test_results/test_report_tuned_{timestamp}.txt

使用方式：
    python scripts/training/train_rehab_model_tuned.py
'''

import os
import time
import numpy
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import json
import pandas as pd

# ========== 可調整參數配置 ==========
# 說明：修改此配置字典以調整模型訓練參數
CONFIG = {
    'name': 'config_1_high_lr',  # 配置名稱，用於識別不同的訓練配置
    'learning_rate': 0.001,  # 學習率，建議範圍：0.0001 - 0.001
    'batch_size': 32,  # 批次大小，建議：16, 32, 64
    'dropout_rate': 0.4,  # Dropout 比率，建議範圍：0.3 - 0.5
    'neurons_layer1': 128,  # 第一層神經元數量
    'neurons_layer2': 64,  # 第二層神經元數量
    'use_standard_scaler': True,  # True: 使用 StandardScaler, False: 使用 Normalizer
    'epochs': 150,  # 最大訓練輪數（實際可能因 Early Stopping 提前結束）
    'test_size': 0.1,  # 驗證集比例（從訓練集中分割，不包含獨立測試集）
    'use_class_weight': True,  # 是否使用類別權重平衡（處理不平衡資料集）
    'use_early_stopping': True,  # 是否啟用 Early Stopping
    'early_stopping_patience': 20,  # Early Stopping 的耐心值（輪數）
    'use_lr_scheduler': True,  # 是否啟用學習率調度器
    'confidence_threshold': 0.1,  # label0 的信心度閾值（預測機率低於此值將分類為 label0）
    # Label0 判斷參數（用於優化 label0 檢測）
    'label0_low_confidence_threshold': 0.35,  # 低信心度閾值（最大機率低於此值直接標記為 label0）
    'label0_medium_confidence_max': 0.95,  # 中等信心度上限（用於組合判斷）
    'label0_std_threshold': 0.07,  # 標準差閾值（機率分布分散程度）
    'label0_entropy_threshold': 0.12,  # 熵閾值（預測不確定性）
}

# ========== 輔助函數 ==========

def _apply_label0_logic(raw_predictions, config):
    """
    實現 label0 判斷邏輯
    
    說明：
        由於模型在訓練時沒有見過 label0，對 label0 資料也會給出高信心度預測，
        但這些預測通常是錯誤的。因此使用組合判斷方式：
        1. 檢查最大機率是否低於閾值（用於處理低信心度的情況）
        2. 檢查預測機率的標準差（label0 的機率分布通常更分散）
        3. 檢查預測機率的熵（label0 的熵通常略高）
    
    參數:
        raw_predictions: 原始預測機率 (n_samples, 7)
        config: 配置字典
    
    返回:
        predicted_labels: 預測標籤 (0-7，其中 0 是 label0)
        confidence_scores: 信心水準
    """
    # 檢查預測結果的有效性（處理邊緣情況）
    valid_mask = numpy.isfinite(raw_predictions).all(axis=1) & (raw_predictions.sum(axis=1) > 0)
    
    # 獲取最大機率和對應的類別
    max_probs = numpy.max(raw_predictions, axis=1)
    predicted_classes = numpy.argmax(raw_predictions, axis=1)
    
    # 計算預測機率的標準差（衡量機率分布的分散程度）
    # Label0 資料的機率分布通常比 Label1-7 更分散
    prob_std = numpy.std(raw_predictions, axis=1)
    
    # 計算熵（entropy）來衡量預測的不確定性
    epsilon = 1e-10  # 避免 log(0)
    entropy = -numpy.sum(raw_predictions * numpy.log(raw_predictions + epsilon), axis=1)
    max_entropy = numpy.log(7)  # 7 個類別的最大熵
    normalized_entropy = entropy / max_entropy  # 正規化到 0-1
    
    # 從配置中獲取閾值
    low_conf_threshold = config.get('label0_low_confidence_threshold', 0.5)
    medium_conf_max = config.get('label0_medium_confidence_max', 0.95)
    std_threshold = config.get('label0_std_threshold', 0.2)
    entropy_threshold = config.get('label0_entropy_threshold', 0.01)
    
    # Label0 判斷邏輯（組合多個條件）：
    # 1. 最大機率低於低信心度閾值（低信心度）
    # 2. 或者：最大機率在中等範圍，且標準差高且熵高（模型不確定）
    #    這表示雖然最大機率不低，但預測分布分散，可能是 label0
    # 3. 資料無效
    low_confidence = max_probs < low_conf_threshold
    medium_confidence_uncertain = (
        (max_probs >= low_conf_threshold) & 
        (max_probs < medium_conf_max) & 
        (prob_std > std_threshold) & 
        (normalized_entropy > entropy_threshold)
    )
    
    predicted_labels = numpy.where(
        low_confidence | medium_confidence_uncertain | (~valid_mask),
        0,  # label0
        predicted_classes + 1  # label1-7 (原始類別 0-6 轉換為 1-7)
    )
    
    # 計算信心水準
    # 說明：
    #   - 對於 label1-7：使用最大機率作為信心度
    #   - 對於 label0：使用 1 - max_prob 作為信心度（表示「不確定」的程度）
    #   - 無效資料的信心度設為 0
    confidence_scores = numpy.where(
        predicted_labels == 0,
        numpy.where(valid_mask, 1 - max_probs, 0.0),  # label0 的信心度（無效資料為 0）
        max_probs  # label1-7 的信心度
    )
    
    return predicted_labels, confidence_scores

# ========== 主程式 ==========
# 設定隨機種子以確保結果可重現
seed = 7
numpy.random.seed(seed)

# 資料集檔案路徑
CSV_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "rehab_dataset_keyframes.csv")

# 檢查檔案是否存在
if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"找不到資料集檔案: {CSV_FILE}")

# Load CSV dataset
print("=" * 60)
print("參數調整版訓練腳本")
print("=" * 60)
print(f"正在載入資料集: {CSV_FILE}")
raw_dataset = numpy.loadtxt(CSV_FILE, delimiter=",")
print(f"資料集載入完成，形狀: {raw_dataset.shape}")

# 提取特徵和標籤
X = raw_dataset[:, 0:42]  # 前 42 列為特徵（手部關鍵點座標）
Y = raw_dataset[:, 42] - 1  # 最後一列為標籤（原始範圍 1-7，轉換為 0-6 以符合 Keras 要求）

# 資料驗證
if X.shape[0] == 0:
    raise ValueError("資料集為空")
if X.shape[1] != 42:
    raise ValueError(f"特徵維度錯誤，預期 42，實際 {X.shape[1]}")

# 驗證標籤範圍
unique_labels = numpy.unique(Y)
if len(unique_labels) != 7 or unique_labels.min() < 0 or unique_labels.max() > 6:
    print(f"警告: 標籤範圍異常，唯一標籤: {unique_labels}")

print(f"\n訓練資料: {X.shape[0]} 筆，特徵維度: {X.shape[1]}")
print(f"標籤範圍: {Y.min()} - {Y.max()}，類別數: {len(unique_labels)}")

# 分析類別分布
print("\n類別分布分析:")
for label in sorted(unique_labels):
    count = numpy.sum(Y == label)
    percentage = count / len(Y) * 100
    print(f"  類別 {int(label)}: {count} 筆 ({percentage:.1f}%)")

# 計算類別權重（用於處理不平衡資料集）
# 說明：如果某些類別的樣本數量差異很大，使用類別權重可以讓模型更關注少數類別
class_weight_dict = None
if CONFIG['use_class_weight']:
    class_weights = compute_class_weight('balanced', classes=unique_labels, y=Y)
    class_weight_dict = {int(label): weight for label, weight in zip(unique_labels, class_weights)}
    print(f"\n類別權重: {class_weight_dict}")

# 資料正規化
# 說明：
#   - StandardScaler: 標準化（均值為 0，標準差為 1），適合大多數情況
#   - Normalizer: 正規化（將每個樣本歸一化為單位向量），適合特徵尺度差異大的情況
# 注意：scaler 物件需要保存，以便在測試時使用相同的正規化參數
print(f"\n正在進行資料正規化（使用 {'StandardScaler' if CONFIG['use_standard_scaler'] else 'Normalizer'}）...")
scaler = None  # 初始化 scaler 變數（如果使用 StandardScaler，將保存此物件）
if CONFIG['use_standard_scaler']:
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print(f"正規化後統計: mean={X.mean():.6f}, std={X.std():.6f}")
else:
    # Normalizer 是無狀態的，不需要保存
    transformer = Normalizer()
    X = transformer.fit_transform(X)

# 分割訓練集與驗證集
# 說明：此處的「測試集」實際上是「驗證集」，用於訓練過程中的模型評估
# 真正的測試集是獨立的 rehab_dataset_keyframes_testing_set.csv，不參與訓練
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=CONFIG['test_size'], random_state=0, stratify=Y
)
print(f"\n訓練集: {X_train.shape[0]} 筆，驗證集: {X_test.shape[0]} 筆（僅用於訓練過程驗證）")
print("注意：測試集使用獨立的 rehab_dataset_keyframes_testing_set.csv，將在訓練完成後進行測試")

# 建立模型
print("\n" + "=" * 60)
print("建立模型")
print("=" * 60)
print(f"架構: {CONFIG['neurons_layer1']} -> {CONFIG['neurons_layer2']} -> 7")
print(f"Dropout: {CONFIG['dropout_rate']}")

model = Sequential()
model.add(Dense(CONFIG['neurons_layer1'], input_dim=42, activation='relu'))
model.add(Dense(CONFIG['neurons_layer2'], activation='relu'))
model.add(Dropout(CONFIG['dropout_rate']))
model.add(Dropout(CONFIG['dropout_rate']))
model.add(Dense(7, activation='softmax'))

optimizer = Adam(learning_rate=CONFIG['learning_rate'])
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

print("\n模型架構:")
model.summary()

# 設置回調函數
callbacks = []
if CONFIG['use_early_stopping']:
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=CONFIG['early_stopping_patience'],
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)

# Model Checkpoint
checkpoint = ModelCheckpoint(
    'best_model_temp.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=0
)
callbacks.append(checkpoint)

if CONFIG['use_lr_scheduler']:
    lr_reducer = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=1
    )
    callbacks.append(lr_reducer)

# 訓練模型
print("\n" + "=" * 60)
print("開始訓練模型")
print("=" * 60)
print("訓練參數:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

start_time = time.time()
history = model.fit(
    X_train, Y_train,
    epochs=CONFIG['epochs'],
    batch_size=CONFIG['batch_size'],
    verbose=2,
    validation_data=(X_test, Y_test),
    shuffle=True,
    class_weight=class_weight_dict,
    callbacks=callbacks
)
training_time = time.time() - start_time

# 載入最佳模型
if os.path.exists('best_model_temp.h5'):
    from keras.models import load_model
    model = load_model('best_model_temp.h5')
    os.remove('best_model_temp.h5')

# 評估模型
print("\n" + "=" * 60)
print("評估模型")
print("=" * 60)

results = model.evaluate(X_test, Y_test, batch_size=CONFIG['batch_size'], verbose=1)
test_loss, test_acc = results

# 混淆矩陣和分類報告
Y_pred_prob = model.predict(X_test, verbose=0)
Y_pred = numpy.argmax(Y_pred_prob, axis=1)

cm = confusion_matrix(Y_test, Y_pred)
report = classification_report(Y_test, Y_pred, 
                                target_names=[f'Gesture {i+1}' for i in range(7)],
                                output_dict=True)

print("\n混淆矩陣:")
print(cm)
print("\n分類報告:")
print(classification_report(Y_test, Y_pred, 
                            target_names=[f'Gesture {i+1}' for i in range(7)]))

# 計算每個類別的準確率
per_class_acc = {}
for i in range(7):
    mask = Y_test == i
    if mask.sum() > 0:
        per_class_acc[f'Gesture {i+1}'] = (Y_pred[mask] == i).sum() / mask.sum()

print("\n各類別準確率:")
for gesture, acc in per_class_acc.items():
    print(f"  {gesture}: {acc:.4f} ({acc*100:.2f}%)")

# 儲存模型和配置
named_tuple = time.localtime()
time_string = time.strftime("%m_%d_%Y_%H_%M_%S", named_tuple)
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
models_dir = os.path.join(base_dir, "models", "rehab_model_tuned")
os.makedirs(models_dir, exist_ok=True)
model_filename = os.path.join(models_dir, f"rehab_model_tuned_{time_string}.h5")

print(f"\n正在儲存模型: {model_filename}")
model.save(model_filename)

# 儲存 scaler（如果使用 StandardScaler）
if scaler is not None:
    import pickle
    scaler_filename = os.path.join(models_dir, f"scaler_{time_string}.pkl")
    with open(scaler_filename, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler 已儲存: {scaler_filename}")

# 儲存配置和結果
config_filename = os.path.join(models_dir, f"config_{time_string}.json")
results_summary = {
    'config': CONFIG,
    'results': {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'training_time_seconds': float(training_time),
        'per_class_accuracy': {k: float(v) for k, v in per_class_acc.items()},
        'confusion_matrix': cm.tolist()
    }
}

with open(config_filename, 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, indent=2, ensure_ascii=False)

print(f"配置和結果已儲存: {config_filename}")

# ========== 使用獨立測試集進行最終測試（包含 label0 邏輯）==========
print("\n" + "=" * 60)
print("使用獨立測試集進行最終測試（包含 label0 邏輯）")
print("=" * 60)

# 載入獨立測試集
TEST_CSV_FILE = os.path.join(base_dir, "data", "rehab_dataset_keyframes_testing_set.csv")
if not os.path.exists(TEST_CSV_FILE):
    print(f"警告: 找不到測試集檔案: {TEST_CSV_FILE}")
    print("跳過最終測試（包含 label0）")
else:
    raw_test_data = numpy.loadtxt(TEST_CSV_FILE, delimiter=",")
    
    X_test_independent = raw_test_data[:, 0:42]  # 特徵
    Y_test_true = raw_test_data[:, 42]  # 標籤（0-7，其中 0 是 label0，1-7 是 label1-7）
    
    # 真實標籤處理：測試集的標籤 0 是 label0，標籤 1-7 是 label1-7
    # 不需要轉換，直接使用（0 對應 label0，1-7 對應 label1-7）
    Y_test_labels = Y_test_true.astype(int)  # 0-7，其中 0 是 label0，1-7 是 label1-7
    
    # 正規化測試資料（使用與訓練時相同的方法）
    if scaler is not None:
        X_test_normalized = scaler.transform(X_test_independent)
    else:
        transformer = Normalizer()
        X_test_normalized = transformer.fit_transform(X_test_independent)
    
    # 進行預測
    raw_predictions = model.predict(X_test_normalized, verbose=0)
    
    # 使用優化後的 label0 判斷邏輯
    predicted_labels, confidence_scores = _apply_label0_logic(raw_predictions, CONFIG)
    # 計算整體準確率
    overall_accuracy = accuracy_score(Y_test_labels, predicted_labels)
    
    # 計算各 label 準確率
    per_label_accuracy = {}
    
    # label0 的統計
    label0_predicted_count = (predicted_labels == 0).sum()
    label0_true_count = (Y_test_labels == 0).sum()
    label0_correct = ((predicted_labels == 0) & (Y_test_labels == 0)).sum()
    label0_false_positive = ((predicted_labels == 0) & (Y_test_labels > 0)).sum()
    label0_false_negative = ((predicted_labels > 0) & (Y_test_labels == 0)).sum()
    
    per_label_accuracy['label0_true_count'] = int(label0_true_count)
    per_label_accuracy['label0_predicted_count'] = int(label0_predicted_count)
    per_label_accuracy['label0_correct'] = int(label0_correct)
    per_label_accuracy['label0_false_positive'] = int(label0_false_positive)
    per_label_accuracy['label0_false_negative'] = int(label0_false_negative)
    if label0_predicted_count > 0:
        per_label_accuracy['label0_precision'] = float(label0_correct / label0_predicted_count)
    else:
        per_label_accuracy['label0_precision'] = 0.0
    if label0_true_count > 0:
        per_label_accuracy['label0_recall'] = float(label0_correct / label0_true_count)
    else:
        per_label_accuracy['label0_recall'] = 0.0
    
    # label1-7 的準確率
    for label in range(1, 8):  # label1 到 label7
        mask = Y_test_labels == label
        if mask.sum() > 0:
            correct = ((predicted_labels == label) & mask).sum()
            total = mask.sum()
            per_label_accuracy[f'label{label}'] = float(correct / total) if total > 0 else 0.0
        else:
            per_label_accuracy[f'label{label}'] = 0.0
    
    # 生成輸出 CSV
    output_data = {
        'true_label': Y_test_labels.astype(int),  # 0-7（0=label0，1-7=label1-7）
        'predicted_label': predicted_labels.astype(int),
        'confidence_score': confidence_scores
    }
    
    # 添加原始預測機率（可選）
    for i in range(7):
        output_data[f'prob_class_{i}'] = raw_predictions[:, i]
    
    # 創建 DataFrame
    df_output = pd.DataFrame(output_data)
    
    # 儲存 CSV
    output_dir = os.path.join(base_dir, "models", "test_results")
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, f"test_predictions_tuned_{time_string}.csv")
    
    df_output.to_csv(output_csv, index=False, encoding='utf-8-sig')
    
    # 生成混淆矩陣（考慮 label0）
    cm_test = confusion_matrix(Y_test_labels, predicted_labels, labels=[0, 1, 2, 3, 4, 5, 6, 7])
    
    # 生成測試報告
    report_file = os.path.join(output_dir, f"test_report_tuned_{time_string}.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("模型測試報告（包含 label0）\n")
        f.write("=" * 60 + "\n")
        f.write(f"測試時間: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模型: rehab_model_tuned_{time_string}\n")
        f.write(f"配置名稱: {CONFIG.get('name', 'N/A')}\n")
        f.write(f"測試集: {TEST_CSV_FILE}\n")
        f.write(f"測試資料筆數: {X_test_independent.shape[0]}\n")
        f.write(f"信心度閾值: {CONFIG.get('confidence_threshold', 0.1)}\n")
        f.write(f"Label0 判斷參數:\n")
        f.write(f"  - 低信心度閾值: {CONFIG.get('label0_low_confidence_threshold', 0.5)}\n")
        f.write(f"  - 中等信心度上限: {CONFIG.get('label0_medium_confidence_max', 0.95)}\n")
        f.write(f"  - 標準差閾值: {CONFIG.get('label0_std_threshold', 0.2)}\n")
        f.write(f"  - 熵閾值: {CONFIG.get('label0_entropy_threshold', 0.01)}\n\n")
        
        f.write("整體準確率:\n")
        f.write(f"  {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)\n\n")
        
        f.write("各 label 準確率:\n")
        for label, acc in per_label_accuracy.items():
            if 'count' in label or 'precision' in label or 'recall' in label or 'correct' in label or 'false' in label:
                if isinstance(acc, float):
                    f.write(f"  {label}: {acc:.4f} ({acc*100:.2f}%)\n")
                else:
                    f.write(f"  {label}: {acc}\n")
            else:
                f.write(f"  {label}: {acc:.4f} ({acc*100:.2f}%)\n")
        
        f.write("\n混淆矩陣（包含 label0）:\n")
        f.write("行: 真實標籤，列: 預測標籤\n")
        f.write("      ")
        for label in [0, 1, 2, 3, 4, 5, 6, 7]:
            f.write(f"label{label:2d}  ")
        f.write("\n")
        for i, row in enumerate(cm_test):
            f.write(f"label{i:2d}  ")
            for val in row:
                f.write(f"{val:6d}  ")
            f.write("\n")
        
        f.write("\n分類報告（包含 label0）:\n")
        report = classification_report(
            Y_test_labels, 
            predicted_labels, 
            labels=[0, 1, 2, 3, 4, 5, 6, 7],
            target_names=['label0', 'label1', 'label2', 'label3', 'label4', 'label5', 'label6', 'label7'],
            output_dict=False
        )
        f.write(report)

# ========== 保存驗證集混淆矩陣 ==========
# 說明：此處保存的是驗證集（從訓練集中分割）的混淆矩陣
# 獨立測試集的混淆矩陣已在上面的測試流程中保存
confusion_matrices_dir = os.path.join(base_dir, "models", "confusion_matrices")
os.makedirs(confusion_matrices_dir, exist_ok=True)

# 保存驗證集混淆矩陣為文本文件
cm_text_file = os.path.join(confusion_matrices_dir, f"rehab_model_tuned_{time_string}.txt")
with open(cm_text_file, 'w', encoding='utf-8') as f:
    f.write("訓練過程驗證結果 (Validation Results)\n")
    f.write("=" * 60 + "\n")
    f.write(f"模型: rehab_model_tuned_{time_string}\n")
    f.write(f"配置名稱: {CONFIG.get('name', 'N/A')}\n")
    f.write(f"驗證準確率: {test_acc:.4f} ({test_acc*100:.2f}%)\n")
    f.write(f"驗證損失: {test_loss:.4f}\n")
    f.write(f"訓練時間: {training_time:.2f} 秒\n\n")
    f.write("混淆矩陣:\n")
    f.write(str(cm) + "\n\n")
    f.write("分類報告:\n")
    f.write(classification_report(Y_test, Y_pred, 
                                  target_names=[f'Gesture {i+1}' for i in range(7)]))
    f.write("\n各類別準確率:\n")
    for gesture, acc in per_class_acc.items():
        f.write(f"  {gesture}: {acc:.4f} ({acc*100:.2f}%)\n")
print(f"驗證集混淆矩陣已儲存: {cm_text_file}")

# 嘗試保存驗證集混淆矩陣圖片
# 說明：如果安裝了 matplotlib 和 seaborn，會生成混淆矩陣的可視化圖片
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'Gesture {i+1}' for i in range(7)],
                yticklabels=[f'Gesture {i+1}' for i in range(7)])
    plt.title(f'Validation Confusion Matrix - rehab_model_tuned_{time_string}\nConfig: {CONFIG.get("name", "N/A")}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    cm_image_file = os.path.join(confusion_matrices_dir, f"rehab_model_tuned_{time_string}.png")
    plt.savefig(cm_image_file, dpi=150)
    plt.close()
    print(f"驗證集混淆矩陣圖片已儲存: {cm_image_file}")
except ImportError:
    print("（跳過混淆矩陣圖片繪製，需要安裝 matplotlib 和 seaborn）")
except Exception as e:
    print(f"（無法繪製混淆矩陣圖片: {e}）")

print("\n" + "=" * 60)
print("訓練完成總結")
print("=" * 60)
print(f"驗證準確率: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"驗證損失: {test_loss:.4f}")
print(f"訓練時間: {training_time:.2f} 秒")
print(f"模型檔案: {model_filename}")
if os.path.exists(TEST_CSV_FILE):
    try:
        print(f"\n獨立測試集結果:")
        print(f"  整體準確率: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
        print(f"  預測結果 CSV: {output_csv}")
        print(f"  測試報告: {report_file}")
    except NameError:
        pass  # 如果測試失敗，這些變數可能未定義
print("=" * 60)

