import wandb
import pandas as pd

# 1. W&B API 초기화
api = wandb.Api()

# 2. 본인의 정보 및 타겟 그룹명 입력
entity = "minsukim207"      # 계정명 또는 팀명
project = "eeg_fm"    # 프로젝트명
starts_with = "eval_A3"   

runs = api.runs(
    f"{entity}/{project}"
)

all_results = []

# 4. 필터링된 Run들을 순회하며 테이블 추출
for run in runs:
    if run.name.startswith(starts_with):
        for artifact in run.logged_artifacts():
            if "lp_results" in artifact.name:
                table = artifact.get("lp_results")
                
                if table is not None:
                    # Pandas DataFrame으로 변환
                    df = table.get_dataframe()
                    
                    # 출처 구분을 위한 정보 추가
                    df["run_id"] = run.id
                    df["run_name"] = run.name
                    df["group"] = run.group
                    
                    all_results.append(df)
                    break # 테이블을 찾았으면 현재 run의 나머지 artifact는 건너뜀 (속도 향상)

# 5. 병합 및 저장
if all_results:
    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(f"{starts_with}_lp_results.csv", index=False)
    print(f"성공적으로 데이터를 합쳐서 '{starts_with}_lp_results.csv'로 저장했어!")
else:
    print("해당 그룹에서 'lp_results' 테이블을 찾지 못했어.")