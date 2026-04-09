import json

def find_exact_subset(items, target_count, target_sum):
    """
    주어진 items 배열에서 정확히 target_count개를 골라 target_sum을 만듭니다.
    조건을 만족하면서 '최대한 뒤쪽에 있는 파일을 버리도록' 역추적합니다.
    """
    n = len(items)
    dp = [[0] * (target_count + 1) for _ in range(n + 1)]
    dp[0][0] = 1
    
    for i in range(1, n + 1):
        _, count = items[i-1]
        for c in range(target_count + 1):
            dp[i][c] = dp[i-1][c] 
            if c > 0 and count <= target_sum:
                dp[i][c] |= (dp[i-1][c-1] << count)
                
    if not (dp[n][target_count] & (1 << target_sum)):
        return None, None
        
    selected = []
    dropped = []
    curr_c = target_count
    curr_s = target_sum
    
    for i in range(n, 0, -1):
        _, count = items[i-1]
        can_skip = (dp[i-1][curr_c] & (1 << curr_s)) > 0
        
        if can_skip:
            dropped.append(items[i-1]) 
        else:
            selected.append(items[i-1]) 
            curr_c -= 1
            curr_s -= count
            
    selected.reverse()
    dropped.reverse()
    return selected, dropped

def get_valid_counts(items, target):
    """
    각 인덱스 k에 대해, items[:k+1] 구간에서 합이 target이 되도록 
    고를 수 있는 파일의 개수(count) 후보들을 모두 계산합니다.
    """
    n = len(items)
    valid_counts = []
    dp = {0: 1} 
    
    for i in range(n):
        _, count = items[i]
        next_dp = {c: v for c, v in dp.items()}
        for c, mask in dp.items():
            if count <= target:
                new_mask = (mask << count) & ((1 << (target + 1)) - 1)
                if new_mask > 0:
                    next_dp[c + 1] = next_dp.get(c + 1, 0) | new_mask
        dp = next_dp
        valid_counts.append([c for c, mask in dp.items() if (mask & (1 << target))])
        
    return valid_counts

def main():
    # 1. 지정해주신 경로로 수정
    json_file_path = r'D:\open_eeg_eval\TUEV_npy\scan_report.json' 
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ 경로에 파일이 없습니다: {json_file_path}")
        return
        
    # 'test' 제외 필터링 및 리스트 변환
    all_files = []
    for filepath, info in data.items():
        if info.get("output_split") in ["train", "val"]:
            all_files.append((filepath, info["n_events"]))
            
    # 원래 순서 보장 (정렬)
    all_files.sort(key=lambda x: x[0])
    
    target_train = 67436
    target_val = 15634
    
    print("1. 전체 파일 스캔 및 가능한 분할선(Pivot) 탐색 중...")
    
    prefix_train = get_valid_counts(all_files, target_train)
    
    reversed_files = all_files[::-1]
    suffix_val_reversed = get_valid_counts(reversed_files, target_val)
    suffix_val = [None] * len(all_files)
    for i in range(len(all_files)):
        suffix_val[i] = suffix_val_reversed[len(all_files) - 1 - i]
        
    best_combo = None
    
    for K in range(len(all_files) - 1):
        valid_P_list = prefix_train[K]
        valid_Q_list = suffix_val[K + 1]
        
        for P in valid_P_list:
            for Q in valid_Q_list:
                if P == int((P + Q) * 0.8):
                    best_combo = (K, P, Q)
                    break 
            if best_combo: break
        if best_combo: break
        
    if not best_combo:
        print("❌ 80:20 비율을 정확히 만족하면서 합계를 맞출 수 있는 Drop 조합이 없습니다.")
        return
        
    K, P, Q = best_combo
    print(f"✅ 조건에 맞는 분할점 발견! (살아남을 파일: Train {P}개, Val {Q}개)")
    print("-" * 50)
    
    train_pool = all_files[:K + 1]
    val_pool = all_files[K + 1:]
    
    train_selected, train_dropped = find_exact_subset(train_pool, P, target_train)
    val_selected, val_dropped = find_exact_subset(val_pool, Q, target_val)
    
    total_kept = train_selected + val_selected
    total_dropped = train_dropped + val_dropped
    
    print("[ 검증 및 최종 결과 요약 ]")
    print(f"📌 남은 전체 파일: {len(total_kept)}개 (Train {len(train_selected)}개 + Val {len(val_selected)}개)")
    print(f"📌 인덱스 기준 80% 계산: int({len(total_kept)} * 0.8) = {int(len(total_kept) * 0.8)}개 -> Train 개수와 완벽 일치!")
    print(f"📌 Train Event 합계: {sum(c for _, c in train_selected):,} (목표: {target_train:,})")
    print(f"📌 Val Event 합계  : {sum(c for _, c in val_selected):,} (목표: {target_val:,})")
    print(f"🗑️ 버려진 파일 수 : {len(total_dropped)}개")
    print("-" * 50)

    # 2. 버려야 할 파일 목록을 JSON으로 출력
    drop_export_path = 'files_to_drop.json'
    
    # total_dropped는 [(filepath, count), ...] 형태이므로 filepath만 추출
    dropped_filepaths = [filepath for filepath, count in total_dropped]
    
    drop_result_dict = {
        "description": "Files to drop before 80:20 Train/Val split",
        "drop_count": len(dropped_filepaths),
        "files_to_drop": dropped_filepaths
    }
    
    with open(drop_export_path, 'w', encoding='utf-8') as f:
        json.dump(drop_result_dict, f, indent=4, ensure_ascii=False)
        
    print(f"📁 버려야 할 파일 목록이 '{drop_export_path}' 파일로 저장되었습니다.")

if __name__ == '__main__':
    main()