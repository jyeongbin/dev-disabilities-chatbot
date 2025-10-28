import re
import os

# --- 설정 ---
# 1. 원본 vectordb_knowledge.txt 파일 경로 (★ "이미지 경로:" 포함된 원본 사용!)
knowledge_file_path = "C:/Users/TTACCTV/OneDrive - 한국정보통신기술협회/바탕 화면/배정빈/dev-disabilities-chatbot-master_V2/vectordb_knowledge1.txt" # 실제 원본 파일 경로로 수정

# 2. 결과 JavaScript 파일 경로 (옵션)
output_js_file = "pictogramData.js"

# 3. 이미지 웹 기본 경로 (★ 수정됨: '/static' 까지만 정의)
static_base_path = "/static"
# --- 설정 끝 ---

# --- get_role_from_category 함수 (이전과 동일) ---
def get_role_from_category(category_str, entity_name):
    """분류 문자열과 개체명을 바탕으로 문법적 역할(role)을 추정합니다."""
    category_lower = category_str.lower()
    entity_lower = entity_name.lower()

    if '행동' in category_lower or entity_lower in ['가다', '오다', '먹다', '마시다', '자다', '씻다', '입다', '벗다', '보다', '듣다', '만들다', '말하다', '도와주다', '받다', '주다', '게임하기', '영상보기']:
        return 'verb'
    elif '감정' in category_lower or entity_lower in ['좋아', '슬퍼', '화나', '무섭다', '아프다', '행복']:
        if len(entity_name.split()) > 1: return 'phrase'
        else: return 'adjective'
    elif '상태' in category_lower or entity_lower in ['깨끗하다', '더럽다', '춥다', '덥다', '다르다', '같다', '작다', '크다', '알다', '모르다']:
         return 'adjective'
    elif '인물' in category_lower or '인체/직업' in category_lower or entity_lower in ['나', '너', '엄마', '아빠', '선생님', '친구']:
         if entity_lower == '나': return 'subject'
         else: return 'subject/object'
    elif '시간' in category_lower or entity_lower in ['오늘', '어제', '내일', '월', '주']: return 'when'
    elif '장소' in category_lower or '건축물' in category_lower or '랜드마크' in category_lower: return 'where'
    elif '취미' in category_lower: return 'hobby'
    elif '기호' in category_lower or entity_lower in ['!', '?', '.']: return 'end'
    elif entity_lower in ['갖고 싶은거 때문에 싸웠어']: return 'phrase'

    return 'object'


def parse_knowledge_file(file_path):
    """vectordb_knowledge.txt 파일을 파싱하여 pictogram 정보를 리스트로 반환합니다."""
    pictogram_list = []
    processed_texts = set()

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 정규표현식: 이미지 경로, 개체명, 분류 추출
        pattern = re.compile(r"이미지 경로:\s*(?P<path>[^,]+),\s*개체명:\s*(?P<name>[^.]+)\.\s*분류:\s*(?P<category>[^.(]+)")

        for line in lines:
            line = line.strip()
            if not line: continue

            match = pattern.search(line)
            if match:
                original_path_from_txt = match.group('path').strip().replace('\\', '/') # 슬래시 통일
                entity_name = match.group('name').strip()
                category = match.group('category').strip()

                if entity_name in processed_texts: continue
                processed_texts.add(entity_name)

                role = get_role_from_category(category, entity_name)

                # ▼▼▼▼▼ 최종 이미지 웹 경로 생성 로직 (수정됨) ▼▼▼▼▼
                image_path = ""
                # 1. 취미인 경우: '/static/images/' + 파일명
                if role == 'hobby':
                    file_name = os.path.basename(original_path_from_txt)
                    image_path = f"{static_base_path}/images/{file_name}"

                # 2. 'ABSTRACT_ILLUSTRATION' 경로인 경우: '/static/button/' + 경로
                elif 'ABSTRACT_ILLUSTRATION' in original_path_from_txt:
                    start_index = original_path_from_txt.find('ABSTRACT_ILLUSTRATION')
                    if start_index != -1:
                        # --- 안전하게 ts_folder 찾기 ---
                        path_before = original_path_from_txt[:start_index].strip('/') # 앞부분 경로 추출 및 양 끝 '/' 제거
                        path_parts = path_before.split('/') # '/' 기준으로 나누기

                        # '/' 기준으로 나눴을 때 유효한 부분이 있고, 그 마지막 부분이 비어있지 않은 경우
                        if path_parts and path_parts[-1]:
                            ts_folder = path_parts[-1] # 마지막 부분을 ts_folder로 간주 (예: 'TS1')
                            # ts_folder/ABSTRACT_ILLUSTRATION 으로 시작하는 부분부터 경로 생성
                            relevant_part_start_str = ts_folder + '/ABSTRACT_ILLUSTRATION'
                            relevant_part_start_index = original_path_from_txt.find(relevant_part_start_str)

                            if relevant_part_start_index != -1:
                                relevant_part = original_path_from_txt[relevant_part_start_index:]
                                image_path = f"{static_base_path}/button/{relevant_part}"
                            else: # 혹시 모를 예외: 그냥 ABSTRACT_ILLUSTRATION 부터 사용
                                relevant_part = original_path_from_txt[start_index:]
                                image_path = f"{static_base_path}/button/{relevant_part}"
                        else: # ABSTRACT_ILLUSTRATION으로 시작하거나 바로 앞에 폴더가 없는 경우
                            relevant_part = original_path_from_txt[start_index:] # ABSTRACT_ILLUSTRATION 부터 사용
                            image_path = f"{static_base_path}/button/{relevant_part}"
                        # --- 안전하게 ts_folder 찾기 끝 ---
                    else: # 'ABSTRACT_ILLUSTRATION'이 없는 fallback (이 경우는 거의 없음)
                        file_name = os.path.basename(original_path_from_txt)
                        image_path = f"{static_base_path}/button/{file_name}"

                # 3. 그 외 (새로 만든 픽토그램 등): '/static/button/' + 파일명
                else:
                    file_name = os.path.basename(original_path_from_txt)
                    image_path = f"{static_base_path}/button/{file_name}"
                # ▲▲▲▲▲ 최종 이미지 웹 경로 생성 로직 (수정됨) ▲▲▲▲▲


                pictogram_dict = {
                    "text": entity_name,
                    "category": category.split('>')[0].strip(),
                    "role": role,
                    "image": image_path # 이미 슬래시(/)로 통일됨
                }
                pictogram_list.append(pictogram_dict)
            else:
                 print(f"경고: 다음 줄을 파싱하지 못했습니다: {line}") # 형식 안 맞는 줄 경고

    except FileNotFoundError:
        print(f"오류: '{file_path}' 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"파일 처리 중 오류 발생: {e}")

    return pictogram_list

# --- 스크립트 실행 ---
pictogram_data = parse_knowledge_file(knowledge_file_path)

if pictogram_data:
    js_output = "const pictogramData = [\n"
    for item in pictogram_data:
        js_output += f"  {{ text: '{item['text']}', category: '{item['category']}', role: '{item['role']}', image: '{item['image']}' }},\n"
    js_output += "];\n"

    print("\n--- JavaScript pictogramData 배열 생성 결과 ---")
    print(js_output)

    try:
        with open(output_js_file, 'w', encoding='utf-8') as f:
            f.write(js_output)
        print(f"\n결과를 '{output_js_file}' 파일로 저장했습니다.")
    except Exception as e:
        print(f"\n파일 저장 중 오류 발생: {e}")
else:
    print("처리할 데이터를 찾지 못했습니다.")