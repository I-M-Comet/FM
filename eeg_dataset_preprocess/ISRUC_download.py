import os
import urllib.request

# 다운로드 받을 폴더 생성
save_dir = "D:/open_eeg_eval/ISRUC_Subgroup1/"
os.makedirs(save_dir, exist_ok=True)

print("다운로드를 시작합니다...")
for i in range(1, 101):
    url = f"http://dataset.isr.uc.pt/ISRUC_Sleep/subgroupI/{i}.rar"
    save_path = os.path.join(save_dir, f"{i}.rar")
    
    # 파일이 이미 있으면 건너뛰기 (이어받기 대용)
    if not os.path.exists(save_path):
        print(f"Downloading {i}.rar ...")
        urllib.request.urlretrieve(url, save_path)
    else:
        print(f"Skip {i}.rar (Already exists)")

print("다운로드 완료!")