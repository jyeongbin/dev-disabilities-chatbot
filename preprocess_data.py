import os
import json
import glob
from tqdm import tqdm
import re # 정규 표현식 사용 (필요시)
# 전처리 코드 

def get_style_info(domain, proficiency):
    """이미지의 스타일과 숙련도 정보를 텍스트로 반환합니다."""
    # 요구사항에 따라 proficiency가 '상'인 경우만 처리되므로, '상'에 대한 설명만 남겨도 무방합니다.
    style = ""
    if domain == "ILLUSTRATION":
        style += "일러스트레이션"
    elif domain == "SKETCH":
        style += "스케치"
    else:
        style += "이미지"

    # proficiency는 항상 '상'이므로 해당 설명 추가
    style += "(숙련도: 상, 상세 묘사)"
    return style

def get_scale_info(scale):
    """이미지의 객체 크기 정보를 텍스트로 반환합니다."""
    if scale == "대":
        return "크게"
    elif scale == "중":
        return "중간 크기로"
    elif scale == "소":
        return "작게"
    else:
        return ""

def generate_factual_description_with_path(json_data):
    """
    JSON 데이터의 정보를 바탕으로 이미지 경로를 포함한
    VectorDB 학습용 '사실 정보' 텍스트를 생성합니다.
    이미지 경로, 내용 부분, 전체 텍스트를 반환합니다.
    """
    try:
        # 1. 이미지 경로 추출
        abstract_img = json_data.get('abstract_image', {})
        img_path = abstract_img.get('abs_path')

        # --- 요구사항 2: proficiency가 '상' 인지 확인 ---
        proficiency = abstract_img.get('proficiency')
        if proficiency != '상':
            # proficiency가 '상'이 아니면 처리하지 않음
            return None, None, None, False # 마지막 False는 proficiency 조건 미달 의미

        if not img_path:
             return None, None, None, True # proficiency는 '상'이지만 경로 없음

        # 2. 필수 정보 추출 (카테고리)
        category = json_data.get('category', {})
        level1 = category.get('ctg_nm_level1')
        level2 = category.get('ctg_nm_level2')
        level3 = category.get('ctg_nm_level3')

        if not level3:
            return img_path, None, None, True # proficiency는 '상'이지만 level3 없음

        # 3. 부가 정보 추출 (이미지 특징)
        domain = abstract_img.get('abs_domain')
        scale = abstract_img.get('object_scale')
        # proficiency는 이미 '상'으로 확인됨

        # 4. 정보 조합
        description_content = f"개체명: {level3}."

        if level1 and level2:
            description_content += f" 분류: {level1} > {level2}."
        elif level1:
            description_content += f" 분류: {level1}."

        style_info = get_style_info(domain, proficiency) # proficiency는 '상'
        scale_info = get_scale_info(scale)

        image_details = []
        if style_info: # 항상 '상' 설명이 포함됨
            image_details.append(f"표현 방식: {style_info}")
        if scale_info:
             image_details.append(f"표현 크기: {scale_info} 표현됨")

        if image_details:
            description_content += f" ({', '.join(image_details)})."

        full_description = f"이미지 경로: {img_path}, {description_content}\n"

        return img_path, description_content, full_description, True # 마지막 True는 proficiency 조건 만족 의미

    except Exception as e:
        # print(f"파일 처리 중 예상치 못한 오류 발생: {e}") # 디버깅 시 주석 해제
        img_path = json_data.get('abstract_image', {}).get('abs_path')
        # 오류 시에도 proficiency 조건은 알 수 없으므로 일단 None 반환
        return img_path, None, None, None

def process_data(dataset_path, output_file):
    """
    데이터셋 폴더(하위 폴더 포함)를 순회하며 proficiency가 '상'인 JSON 파일만
    전처리하고 내용 기반 중복을 제거하여 결과를 파일에 저장합니다.
    """
    # --- 요구사항 1: 하위 폴더 포함 모든 JSON 파일 검색 ---
    # glob의 recursive=True 옵션을 사용하여 하위 폴더까지 모두 검색
    json_files = glob.glob(os.path.join(dataset_path, '**', '*.json'), recursive=True)

    print(f"총 {len(json_files)}개의 JSON 파일을 '{dataset_path}' 및 하위 폴더에서 찾았습니다.")
    print("proficiency가 '상'인 파일만 처리하여 VectorDB 학습 데이터 생성을 시작합니다...")

    processed_content = set() # 이미 처리한 '내용 부분' 저장용 집합
    processed_count = 0
    skipped_proficiency_count = 0 # proficiency 조건 미달로 건너뛴 횟수
    skipped_content_duplicates_count = 0 # 내용 중복으로 건너뛴 횟수
    error_count = 0

    with open(output_file, 'w', encoding='utf-8') as out_f:
        for file_path in tqdm(json_files, desc="데이터 전처리 중"):
            current_content_part = None
            is_proficiency_ok = None
            try:
                with open(file_path, 'r', encoding='utf-8') as in_f:
                    content = in_f.read()
                    if not content.strip():
                        error_count += 1
                        continue

                    data = json.loads(content)

                    # 이미지 경로, 내용 부분, 전체 텍스트, proficiency 조건 만족 여부 받아옴
                    _, current_content_part, full_processed_text, is_proficiency_ok = generate_factual_description_with_path(data)

                    # proficiency 조건 미달 시 건너<0xEB><0x9A><0xB4>
                    if is_proficiency_ok is False:
                        skipped_proficiency_count += 1
                        continue
                    # 오류 발생 시 (is_proficiency_ok가 None일 수 있음) 또는 내용 생성 실패 시
                    elif current_content_part is None:
                        error_count += 1
                        continue
                    # 내용 부분을 기준으로 중복 체크
                    elif current_content_part in processed_content:
                        skipped_content_duplicates_count += 1
                        continue

                    # 유효한 전체 텍스트가 생성되었고, 새로운 내용이라면 파일에 쓰고 내용 기록
                    if full_processed_text: # is_proficiency_ok가 True인 경우만 여기까지 옴
                        out_f.write(full_processed_text)
                        processed_content.add(current_content_part)
                        processed_count += 1
                    else: # 이론상 여기까지 오기 어려움 (content_part가 None 아니면서 full_text가 None)
                        error_count += 1
                        if current_content_part:
                            processed_content.add(current_content_part)


            except json.JSONDecodeError:
                # print(f"경고: {file_path} 파일은 유효한 JSON이 아닙니다.")
                error_count += 1
                continue
            except Exception as e:
                # print(f"경고: {file_path} 처리 중 오류 발생 - {e}")
                error_count += 1
                if current_content_part: # 오류 시에도 content_part 알면 기록 시도
                     processed_content.add(current_content_part)
                continue

    print(f"\n데이터 전처리 완료!")
    print(f"- 총 {len(json_files)}개 JSON 파일 확인")
    print(f"- proficiency '상' 조건 만족 및 고유 내용 {processed_count}개 처리 성공 (파일 저장됨)")
    print(f"- proficiency 조건 미달 {skipped_proficiency_count}개 건너<0xEB><0x9B><0x84>")
    print(f"- 내용 중복 {skipped_content_duplicates_count}개 건너<0xEB><0x9B><0x84>")
    print(f"- 처리 실패 또는 오류 {error_count}개")
    print(f"결과가 {output_file}에 저장되었습니다.")

# --- 실행 부분 ---
if __name__ == "__main__":
    # --- 요구사항 1: 최상위 폴더 경로 설정 ---
    # 예시: 'training' 폴더 전체를 처리하도록 경로 수정
    DATASET_ROOT_PATH = "C:/Users/TTACCTV/Downloads/download/049.스케치,_아이콘_인식용_다양한_추상_이미지_데이터/01.데이터/1.Training/라벨링데이터" # training 폴더 경로로 가정
    
    # 최종 결과물 파일 이름 (proficiency '상'만 포함됨을 명시)
    OUTPUT_TXT_FILE = "vectordb_knowledge1.txt"

    process_data(DATASET_ROOT_PATH, OUTPUT_TXT_FILE)