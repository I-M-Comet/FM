import wandb
import json

# 1. API 초기화
api = wandb.Api()

# 💡 본인의 WandB 엔티티(계정명)와 프로젝트명으로 반드시 변경해 줘!
ENTITY = "minsukim207"   
PROJECT = "eeg_fm"
STARTS_WITH = "B0_0"  # ★ 여기에 지정했던 그룹명을 입력해!
runs = api.runs(f"{ENTITY}/{PROJECT}") 

# 추가로 뽑고 싶은 특정 지표들
other_metrics = ['_step','loss_scaled', 'grad_norm', 'loss_tgt_mean', 'tgt_tokens_sum', 'ctx_tokens_sum']
output_filename = f"{STARTS_WITH}_runs_metrics.jsonl"

print("🔥 JSONL 형식으로 데이터 추출을 시작합니다...")

# 2. 파일을 쓰기 모드('w')로 열기
with open(output_filename, 'w', encoding='utf-8') as f:
    
    run_count = 0
    for run in runs:
        if run.name.startswith(STARTS_WITH):
            print(f"[{run.name}] 데이터 다운로드 및 저장 중...")
            run_count += 1
            
            # 4. scan_history()로 전체 스텝 순회
            for row in run.scan_history():
                
                # 5. 조건에 맞는 컬럼만 딕셔너리로 새로 구성 (proxy 시작 또는 지정된 지표)
                filtered_row = {
                    k: v for k, v in row.items() 
                    if str(k).startswith('proxy') or k in other_metrics
                }
                
                # 추출할 데이터가 있다면 파일에 쓰기
                if filtered_row:
                    # 나중에 어떤 Run의 데이터인지 알 수 있게 이름 추가
                    filtered_row['run_name'] = run.name
                    
                    # 딕셔너리를 JSON 문자열로 변환하고 줄바꿈(\n)을 더해서 파일에 바로 쓰기
                    f.write(json.dumps(filtered_row) + '\n')

if run_count > 0:
    print(f"\n✅ 추출 완료! '{output_filename}' 파일이 성공적으로 생성되었어.")
else:
    print("\n❌ 조건에 맞는 Run을 찾지 못했어. 프로젝트나 런 이름을 확인해 줘.")