'''
系統化測試不同參數組合以找出最佳配置

可調整參數：
1. 學習率 (learning_rate): 0.001, 0.0005, 0.0001
2. Batch size: 16, 32, 64
3. Dropout rate: 0.3, 0.4, 0.5
4. 模型架構: 不同層數和神經元數量
5. 正規化方法: Normalizer vs StandardScaler
6. 訓練策略: Early Stopping, Model Checkpoint, LR Scheduler
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
from sklearn.metrics import confusion_matrix, classification_report
import json

# ========== 可調整參數配置 ==========
CONFIG = {
    'name': 'config_1_high_lr',
    'learning_rate': 0.001,
    'batch_size': 32,
    'dropout_rate': 0.4,
    'neurons_layer1': 128,
    'neurons_layer2': 64,
    'use_standard_scaler': False,
    'epochs': 150,
    'test_size': 0.1,
    'use_class_weight': True,
    'use_early_stopping': True,
    'early_stopping_patience': 20,
    'use_lr_scheduler': True,
}

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

X = raw_dataset[:, 0:42]
Y = raw_dataset[:, 42] - 1  # 轉換為 0-6

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

# 計算類別權重
class_weight_dict = None
if CONFIG['use_class_weight']:
    class_weights = compute_class_weight('balanced', classes=unique_labels, y=Y)
    class_weight_dict = {int(label): weight for label, weight in zip(unique_labels, class_weights)}
    print(f"\n類別權重: {class_weight_dict}")

# 資料正規化
print(f"\n正在進行資料正規化（使用 {'StandardScaler' if CONFIG['use_standard_scaler'] else 'Normalizer'}）...")
if CONFIG['use_standard_scaler']:
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print(f"正規化後統計: mean={X.mean():.6f}, std={X.std():.6f}")
else:
    transformer = Normalizer()
    X = transformer.fit_transform(X)

# 分割訓練集與測試集
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=CONFIG['test_size'], random_state=0, stratify=Y
)
print(f"\n訓練集: {X_train.shape[0]} 筆，測試集: {X_test.shape[0]} 筆")

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

# 保存混淆矩陣到 confusion_matrices 資料夾
confusion_matrices_dir = os.path.join(base_dir, "models", "confusion_matrices")
os.makedirs(confusion_matrices_dir, exist_ok=True)

# 保存混淆矩陣為文本文件
# cm_text_file = os.path.join(confusion_matrices_dir, f"rehab_model_tuned_{time_string}.txt")
# with open(cm_text_file, 'w', encoding='utf-8') as f:
#     f.write("混淆矩陣 (Confusion Matrix)\n")
#     f.write("=" * 60 + "\n")
#     f.write(f"模型: rehab_model_tuned_{time_string}\n")
#     f.write(f"配置名稱: {CONFIG.get('name', 'N/A')}\n")
#     f.write(f"測試準確率: {test_acc:.4f} ({test_acc*100:.2f}%)\n")
#     f.write(f"測試損失: {test_loss:.4f}\n")
#     f.write(f"訓練時間: {training_time:.2f} 秒\n\n")
#     f.write("混淆矩陣:\n")
#     f.write(str(cm) + "\n\n")
#     f.write("分類報告:\n")
#     f.write(classification_report(Y_test, Y_pred, 
#                                   target_names=[f'Gesture {i+1}' for i in range(7)]))
#     f.write("\n各類別準確率:\n")
#     for gesture, acc in per_class_acc.items():
#         f.write(f"  {gesture}: {acc:.4f} ({acc*100:.2f}%)\n")
# print(f"混淆矩陣已儲存: {cm_text_file}")

# 嘗試保存混淆矩陣圖片
# try:
#     import matplotlib.pyplot as plt
#     import seaborn as sns
    
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=[f'Gesture {i+1}' for i in range(7)],
#                 yticklabels=[f'Gesture {i+1}' for i in range(7)])
#     plt.title(f'Confusion Matrix - rehab_model_tuned_{time_string}\nConfig: {CONFIG.get("name", "N/A")}')
#     plt.ylabel('True Label')
#     plt.xlabel('Predicted LACabel')
#     plt.tight_layout()
    
#     cm_image_file = os.path.join(confusion_matrices_dir, f"rehab_model_tuned_{time_string}.png")
#     plt.savefig(cm_image_file, dpi=150)
#     plt.close()
#     print(f"混淆矩陣圖片已儲存: {cm_image_file}")
# except ImportError:
#     print("（跳過混淆矩陣圖片繪製，需要安裝 matplotlib 和 seaborn）")
# except Exception as e:
#     print(f"（無法繪製混淆矩陣圖片: {e}）")

print("\n" + "=" * 60)
print("訓練完成總結")
print("=" * 60)
# print(f"測試準確率: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"測試損失: {test_loss:.4f}")
print(f"訓練時間: {training_time:.2f} 秒")
print(f"模型檔案: {model_filename}")
print("=" * 60)