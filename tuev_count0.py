import json

def find_subset_dp(items, target):
    """
    동적 계획법(DP)을 사용하여 정확히 target 합을 만드는 아이템 조합을 찾습니다.
    items: [(파일경로, event_count), ...] 형태의 리스트
    target: 목표 event 개수
    """
    dp = [-1] * (target + 1)
    dp[0] = -2  
    
    max_s = 0  
    
    for i, (filepath, count) in enumerate(items):
        if count > target:
            continue
            
        for s in range(max_s, -1, -1):
            if dp[s] != -1:
                new_sum = s + count
                if new_sum <= target and dp[new_sum] == -1:
                    dp[new_sum] = i
                    if new_sum > max_s:
                        max_s = new_sum
                        
        if dp[target] != -1:
            break

    if dp[target] == -1:
        return None
        
    selected = []
    curr = target
    while curr > 0:
        item_idx = dp[curr]
        selected.append(items[item_idx])
        curr -= items[item_idx][1]
        
    return selected

def main():
    # 1. JSON 파일 로드
    json_file_path = 'D:\\open_eeg_eval\\TUEV_npy\\scan_report.json'
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # 💡 [수정된 부분] 'test'는 무시하고 'train'이나 'val'인 경우만 all_files에 담습니다.
    all_files = []
    for filepath, info in data.items():
        if info.get("output_split") in ["train", "val"]:
            all_files.append((filepath, info["n_events"]))
    
    target_train = 67436
    target_val = 15634
    
    print(f"✅ 'test' 제외 필터링 완료")
    print(f"탐색 대상 파일 개수: {len(all_files)}")
    print(f"탐색 대상 Event 합계: {sum(count for _, count in all_files)}")
    print("-" * 50)
    
    # 3. Train 데이터 조합 찾기
    print(f"Train 조합 찾는 중... (목표: {target_train}개)")
    train_subset = find_subset_dp(all_files, target_train)
    
    if not train_subset:
        print("❌ Train 개수를 정확히 맞출 수 있는 조합이 존재하지 않습니다.")
        return
        
    train_paths = set(filepath for filepath, _ in train_subset)
    print(f"✅ Train 데이터 분리 완료 (파일 {len(train_subset)}개 선택됨)")
    
    # 4. 남은 데이터에서 Validation 조합 찾기
    remaining_files = [item for item in all_files if item[0] not in train_paths]
    
    print(f"\nValidation 조합 찾는 중... (목표: {target_val}개)")
    val_subset = find_subset_dp(remaining_files, target_val)
    
    if not val_subset:
        print("❌ Validation 개수를 정확히 맞출 수 있는 조합이 남은 파일 중에 존재하지 않습니다.")
        return
        
    val_paths = set(filepath for filepath, _ in val_subset)
    print(f"✅ Validation 데이터 분리 완료 (파일 {len(val_subset)}개 선택됨)")
    
    # 5. 버려지는(Drop) 파일들 정리
    dropped_files = [item for item in remaining_files if item[0] not in val_paths]
    
    print("-" * 50)
    print("[ 최종 결과 요약 ]")
    print(f"📌 Train 파일 개수: {len(train_subset)}개 / Event 합계: {sum(c for _, c in train_subset):,}")
    print(f"📌 Val 파일 개수  : {len(val_subset)}개 / Event 합계: {sum(c for _, c in val_subset):,}")
    print(f"🗑️ 버려진 파일 개수: {len(dropped_files)}개 / Event 합계: {sum(c for _, c in dropped_files):,}")
    
    # JSON 파일로 결과 내보내기 (선택사항)
    result_dict = {
        "train_files": [p for p, _ in train_subset],
        "val_files": [p for p, _ in val_subset],
        "dropped_files": [p for p, _ in dropped_files]
    }
    with open("split_result_filtered.json", "w", encoding='utf-8') as f:
        json.dump(result_dict, f, indent=4)

if __name__ == '__main__':
    main()