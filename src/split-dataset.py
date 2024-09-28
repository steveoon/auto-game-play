import os
import random
import shutil

# 源目录
src_images = "/Users/rensiwen/Documents/LLMs/auto-game-play/original-label-data/project-2-at-2024-09-28-13-19-b06df912/images"
src_labels = "/Users/rensiwen/Documents/LLMs/auto-game-play/original-label-data/project-2-at-2024-09-28-13-19-b06df912/labels"

# 目标目录
dst_base = "/Users/rensiwen/Documents/LLMs/auto-game-play/dataset"


def split_dataset(images_dir, labels_dir, dst_base, train_ratio=0.8):
    # 确保目标目录存在
    for subset in ["train", "val"]:
        os.makedirs(os.path.join(dst_base, "images", subset), exist_ok=True)
        os.makedirs(os.path.join(dst_base, "labels", subset), exist_ok=True)

    # 获取所有图片文件
    images = [
        f
        for f in os.listdir(images_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    random.shuffle(images)

    # 计算分割点
    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    # 复制文件
    for subset, subset_images in [("train", train_images), ("val", val_images)]:
        for img in subset_images:
            # 复制图片
            shutil.copy2(
                os.path.join(images_dir, img),
                os.path.join(dst_base, "images", subset, img),
            )

            # 复制对应的标签文件
            label = img.rsplit(".", 1)[0] + ".txt"
            if os.path.exists(os.path.join(labels_dir, label)):
                shutil.copy2(
                    os.path.join(labels_dir, label),
                    os.path.join(dst_base, "labels", subset, label),
                )
            else:
                print(f"警告: 标签文件 {label} 不存在")

    print(f"数据集分割完成。训练集: {len(train_images)}张, 验证集: {len(val_images)}张")


# 执行数据集分割
split_dataset(src_images, src_labels, dst_base)
