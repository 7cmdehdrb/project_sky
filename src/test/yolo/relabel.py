import os
import glob


def main():
    # 1. 클래스 ID 매핑 딕셔너리 (기존 ID -> 새 ID)
    # 누락된 기존 ID 11 (cup_4_purple)은 이 딕셔너리에 없으므로 라벨에서 삭제됩니다.
    mapping = {
        4: 0,  # can_1_cola -> coca_cola
        5: 1,  # can_2_sikhye -> sikhye
        6: 2,  # can_3_peach -> yello_peach
        7: 3,  # can_4_catata -> cantata
        8: 4,  # cup_1_sky -> cup_sky
        9: 5,  # cup_2_white -> cup_white
        10: 6,  # cup_3_blue -> cup_blue
        12: 7,  # cup_5_green -> cup_green
        13: 8,  # mug_1_black -> mug_black
        14: 9,  # mug_2_gray -> mug_gray
        15: 10,  # mug_3_yellow -> mug_yello
        16: 11,  # mug_4_orange -> mug_orange
        0: 12,  # bottle_1_alive -> alive
        1: 13,  # bottle_2_greentea -> green_tea
        2: 14,  # bottle_3_yellow -> yello_smoothie
        3: 15,  # bottle_4_red -> bottle_red
        17: 16,  # soda -> cyder
    }

    # 2. 라벨 파일이 있는 폴더 경로 (실제 환경에 맞게 상위 폴더 경로를 지정하세요)
    # 예: dataset 폴더 안에 train, valid, test가 있는 경우
    base_dir = "/home/min/7cmdehdrb/project_sky/src/test/yolo/yolo_dataset"  # data.yaml이 있는 최상위 경로로 맞춰주세요.
    target_folders = ["train/labels", "valid/labels"]

    changed_files_count = 0
    deleted_labels_count = 0

    # 3. 각 폴더를 순회하며 라벨 수정
    for folder in target_folders:
        folder_path = os.path.join(base_dir, folder)
        if not os.path.exists(folder_path):
            print(f"⚠️ 폴더를 찾을 수 없어 건너뜁니다: {folder_path}")
            continue

        # 폴더 내 모든 txt 파일 검색
        txt_files = glob.glob(os.path.join(folder_path, "*.txt"))

        for txt_file in txt_files:
            with open(txt_file, "r") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue

                old_id = int(parts[0])

                # 매핑 딕셔너리에 있는 ID만 변환하여 저장
                if old_id in mapping:
                    new_id = mapping[old_id]
                    # 클래스 ID만 새 ID로 바꾸고 나머지 좌표(폴리곤)는 그대로 이어붙임
                    new_line = f"{new_id} " + " ".join(parts[1:]) + "\n"
                    new_lines.append(new_line)
                else:
                    # 매핑에 없는 ID(예: 11번 cup_4_purple)는 제외됨
                    deleted_labels_count += 1

            # 변환된 내용으로 파일 덮어쓰기
            with open(txt_file, "w") as f:
                f.writelines(new_lines)

            changed_files_count += 1

    print("✅ 라벨 변환 작업이 완료되었습니다!")
    print(f"총 처리한 파일 수: {changed_files_count}개")
    print(f"삭제된 라벨(매핑되지 않은 클래스) 수: {deleted_labels_count}개")


if __name__ == "__main__":
    main()
