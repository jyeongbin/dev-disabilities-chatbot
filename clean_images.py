import os
import re
import shutil
# 전처리 파일 이미지 분류 코드


# --- 설정 ---
# 1. 이미지 경로가 포함된 원본 txt 파일 경로
knowledge_file_path = "C:/Users/TTACCTV/Downloads/download/vectordb_knowledge.txt"
# 2. 실제 이미지 파일들이 있는 최상위 폴더 경로 (매우 중요!)
# 예시: 'C:/project/images' 또는 './images' 등
# txt 파일에 있는 경로가 이 폴더를 기준으로 한 상대 경로여야 합니다.
image_base_dir = "C:/Users/TTACCTV/Downloads/download/049.스케치,_아이콘_인식용_다양한_추상_이미지_데이터/01.데이터/1.Training/원천데이터 - 복사본/TS1/ABSTRACT_ILLUSTRATION" # 현재 폴더를 기준으로 설정 (실제 환경에 맞게 수정하세요)

# 3. 삭제 대신 이동할 폴더 경로 (None으로 두면 즉시 삭제)
# 예: './unnecessary_images' (이 폴더가 미리 존재해야 합니다)
move_to_dir = './unnecessary_images' 
# move_to_dir = None # 주석 해제 시 즉시 삭제 모드

# 4. 처리할 이미지 확장자
allowed_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp') 
# --- 설정 끝 ---

def extract_paths_from_file(file_path):
    """txt 파일에서 '이미지 경로: ' 다음의 파일 경로들을 추출합니다."""
    paths = set()
    prefix_to_remove = "ABSTRACT_ILLUSTRATION/" # txt 경로에 이게 포함되어 있다면 제거
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            found_paths = re.findall(r"이미지 경로:\s*([^\s,]+)", content)
            for path in found_paths:
                # 경로 정규화 (운영체제에 맞게 슬래시 변환, 역슬래시를 슬래시로 통일)
                normalized_path = os.path.normpath(path.strip()).replace('\\', '/')

                # 경로 앞부분 제거 (만약 있다면)
                if normalized_path.startswith(prefix_to_remove):
                     normalized_path = normalized_path[len(prefix_to_remove):]
                
                paths.add(normalized_path)
    except FileNotFoundError:
        print(f"오류: '{file_path}' 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"파일 읽기 중 오류 발생: {e}")
    return paths

def find_all_images(base_dir, extensions):
    """지정된 폴더 및 하위 폴더에서 모든 이미지 파일 경로를 찾습니다."""
    image_files = set()
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(extensions):
                full_path = os.path.join(root, file)
                # 기본 디렉토리를 기준으로 상대 경로 생성 및 정규화
                relative_path = os.path.relpath(full_path, base_dir)
                normalized_path = os.path.normpath(relative_path).replace('\\', '/')
                image_files.add(normalized_path)
    return image_files

# 1. txt 파일에서 필요한 이미지 경로 목록 추출
print(f"'{knowledge_file_path}'에서 필요한 이미지 경로를 추출합니다...")
required_paths = extract_paths_from_file(knowledge_file_path)
if not required_paths:
    print("추출된 경로가 없습니다. 스크립트를 종료합니다.")
    exit()
print(f"총 {len(required_paths)}개의 필요한 이미지 경로를 찾았습니다.")

# 2. 실제 이미지 폴더에서 모든 이미지 파일 목록 찾기
print(f"'{image_base_dir}' 폴더 및 하위 폴더에서 모든 이미지 파일을 검색합니다...")
existing_images = find_all_images(image_base_dir, allowed_extensions)
print(f"총 {len(existing_images)}개의 이미지 파일을 찾았습니다.")

print("\n--- 경로 샘플 비교 ---")
print("TXT에서 추출된 경로 샘플 (required_paths):")
count = 0
for path in required_paths:
    print(f" - {path}")
    count += 1
    if count >= 5: # 5개 샘플만 출력
        break

print("\n실제 폴더에서 찾은 경로 샘플 (existing_images):")
count = 0
for path in existing_images:
    print(f" - {path}")
    count += 1
    if count >= 5: # 5개 샘플만 출력
        break
print("--- 경로 샘플 비교 끝 ---\n")
# 테스트용 

# 3. 불필요한 이미지 파일 식별 (실제 파일 목록에는 있지만, txt 목록에는 없는 파일)
unnecessary_images = existing_images - required_paths
necessary_images_found = existing_images.intersection(required_paths)
missing_images = required_paths - existing_images

print("\n--- 결과 ---")
print(f"필요한 이미지 파일 수 (txt 기준): {len(required_paths)}")
print(f"실제 찾은 이미지 파일 수: {len(existing_images)}")
print(f"유지될 이미지 파일 수 (실제 파일 중 필요한 것): {len(necessary_images_found)}")
if missing_images:
    print(f"경고: txt 파일에는 있지만 실제 폴더에 없는 이미지 수: {len(missing_images)}")
    # 필요시 누락된 파일 목록 출력 (주석 해제)
    # print("누락된 파일 목록:")
    # for img in sorted(list(missing_images)):
    #     print(f" - {img}")
print(f"정리 대상(불필요한) 이미지 파일 수: {len(unnecessary_images)}")

# 4. 불필요한 파일 처리 (삭제 또는 이동)
if unnecessary_images:
    print("\n--- 작업 시작 ---")
    confirm = input(f"총 {len(unnecessary_images)}개의 불필요한 이미지 파일을 처리하시겠습니까? "
                    f"({'이동' if move_to_dir else '삭제'}) [y/N]: ")

    if confirm.lower() == 'y':
        processed_count = 0
        if move_to_dir:
            # 이동 모드
            if not os.path.exists(move_to_dir):
                try:
                    os.makedirs(move_to_dir)
                    print(f"'{move_to_dir}' 폴더를 생성했습니다.")
                except OSError as e:
                    print(f"오류: 이동 대상 폴더 생성 실패 - {e}. 스크립트를 종료합니다.")
                    exit()
            
            print(f"불필요한 파일을 '{move_to_dir}' 폴더로 이동합니다...")
            for img_rel_path in unnecessary_images:
                source_path = os.path.join(image_base_dir, img_rel_path)
                # 대상 경로에 하위 폴더 구조 유지
                dest_path = os.path.join(move_to_dir, img_rel_path)
                dest_dir = os.path.dirname(dest_path)
                
                try:
                    if not os.path.exists(dest_dir):
                        os.makedirs(dest_dir) # 대상 하위 폴더 생성
                    shutil.move(source_path, dest_path)
                    # print(f"이동: {source_path} -> {dest_path}") # 상세 로그 필요시 주석 해제
                    processed_count += 1
                except Exception as e:
                    print(f"오류: '{source_path}' 이동 실패 - {e}")
            print(f"총 {processed_count}개의 파일을 이동했습니다.")

        else:
            # 삭제 모드
            print("불필요한 파일을 삭제합니다...")
            for img_rel_path in unnecessary_images:
                file_to_delete = os.path.join(image_base_dir, img_rel_path)
                try:
                    os.remove(file_to_delete)
                    # print(f"삭제: {file_to_delete}") # 상세 로그 필요시 주석 해제
                    processed_count += 1
                except Exception as e:
                    print(f"오류: '{file_to_delete}' 삭제 실패 - {e}")
            print(f"총 {processed_count}개의 파일을 삭제했습니다.")
    else:
        print("작업이 취소되었습니다.")
else:
    print("\n정리할 불필요한 이미지 파일이 없습니다.")

print("\n스크립트 완료.")