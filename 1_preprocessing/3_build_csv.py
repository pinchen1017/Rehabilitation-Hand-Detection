import csv
import json
from pathlib import Path

PROCESSED_DIR = Path("2_processed")
OUT_CSV = Path("3_result/rehab_dataset_keyframes.csv")

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

def main():
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)

        # 原本是讀 folder 裡的 .json
# 現在 folder 裡面會有很多 .json (因為一張圖一個)

        for gesture_dir in sorted(PROCESSED_DIR.iterdir()):
            if not gesture_dir.is_dir(): continue
            
            # 解析 Label (跟 preprocess 一樣)
            try:
                label = int(gesture_dir.name.split('_')[0])
            except:
                continue
                
            # 讀取該資料夾下「所有的」JSON 檔
            for json_file in gesture_dir.rglob("*.json"): # 使用 rglob 抓子資料夾
                with open(json_file, "r") as f:
                    data = json.load(f)
                    frames = data["frames"]
                    
                    # 每個 JSON 來自一張照片(擴增成30筆)
                    for vec in frames:
                            # 把 label 加到最後面
                            row = vec + [label]
                            
                            # 資料寫進 CSV
                            writer.writerow(row)

            print(f"[OK] Processed class: {gesture_dir.name}")

    print(f"\n===> FINISHED: {OUT_CSV} 已完成！")


if __name__ == "__main__":
    main()