import os
import re
import numpy as np

base_dir = "output/scannet_0421_mlp"

scenes = [
    "scene0000_00",
    "scene0062_00",
    "scene0070_00",
    "scene0097_00",
    "scene0140_00",
    "scene0347_00",
    "scene0590_00",
    "scene0645_00",
]

overall_acc_list = []
macc_list = []

for scene in scenes:
    file_path = os.path.join(base_dir, scene, "eval_3d_10", "eval3d_results.txt")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    with open(file_path, "r") as f:
        content = f.read()

    # 用正则提取数值
    overall_match = re.search(r"Reported Overall Accuracy:\s*([0-9.]+)", content)
    macc_match = re.search(r"Reported mAcc:\s*([0-9.]+)", content)

    # block = re.search(r"Volume-aware Accuracy results:(.*?)Standard Accuracy results", content, re.S)
    # if block:
    #     section = block.group(1)
    #     overall_match = re.search(r"Overall Accuracy:\s*([0-9.]+)", section)
    #     macc_match = re.search(r"Mean Class Accuracy \(mAcc\):\s*([0-9.]+)", section)

    if overall_match and macc_match:
        overall_acc = float(overall_match.group(1))
        macc = float(macc_match.group(1))

        overall_acc_list.append(overall_acc)
        macc_list.append(macc)

        print(f"{scene}: Overall={overall_acc:.4f}, mAcc={macc:.4f}")
    else:
        print(f"Failed to parse: {file_path}")

# 计算均值
if overall_acc_list and macc_list:
    mean_overall = np.mean(overall_acc_list)
    mean_macc = np.mean(macc_list)

    print("\n===== Final Results =====")
    print(f"Mean Standard Overall Accuracy: {mean_overall:.4f}")
    print(f"Mean Standard mAcc: {mean_macc:.4f}")
else:
    print("No valid data found.")