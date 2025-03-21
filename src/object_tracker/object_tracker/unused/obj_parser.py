import os
import sys
import math
import json


def get_obj_bounds(file_path):
    vertices = []

    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("v "):  # 정점 정보만 파싱
                parts = line.strip().split()
                if len(parts) >= 4:
                    x, y, z = map(float, parts[1:4])
                    vertices.append((x, y, z))

    if not vertices:
        raise ValueError("OBJ 파일에 유효한 정점 정보가 없습니다.")

    # x, y, z 좌표 각각 리스트로 분리
    xs, ys, zs = zip(*vertices)

    bounds = {
        "min_x": min(xs),
        "max_x": max(xs),
        "min_y": min(ys),
        "max_y": max(ys),
        "min_z": min(zs),
        "max_z": max(zs),
    }

    return bounds


# 사용 예:
root_dir = "/home/min/7cmdehdrb/ros2_ws/src/object_tracker/resource/models"
folders = os.listdir(root_dir)

result = {}

for folder in folders:
    obj_file = os.path.join(root_dir, folder, f"{folder}.obj")
    bounds = get_obj_bounds(obj_file)
    result[folder] = {
        "x": abs(bounds["max_x"] - bounds["min_x"]),
        "y": abs(bounds["max_y"] - bounds["min_y"]),
        "z": abs(bounds["max_z"] - bounds["min_z"]),
    }

with open("obj_bounds.json", "w") as f:
    json.dump(result, f)

    f.close()
