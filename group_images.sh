#!/bin/zsh

# 设置源目录
source_dir="./screenshots"

# 检查源目录是否存在
if [ ! -d "$source_dir" ]; then
    echo "错误：$source_dir 目录不存在。"
    exit 1
fi

# 使用 find 命令查找 PNG 文件
images=($(find "$source_dir" -type f -name "frame_*_*.png"))

# 检查是否找到图片文件
if [ ${#images[@]} -eq 0 ]; then
    echo "错误：$source_dir 目录下没有找到符合格式的 PNG 文件。"
    exit 1
fi

# 定义每组的图片数量
group_size=10
group_number=1
counter=0

# 遍历图片数组进行分组和复制
for image in "${images[@]}"; do
    if [ $((counter % group_size)) -eq 0 ]; then
        folder_name="group_${group_number}"
        mkdir -p "$folder_name"
        group_number=$((group_number + 1))
    fi
    cp "$image" "$folder_name/"
    counter=$((counter + 1))
done

echo "图片已成功分组并复制。共处理 $counter 张图片。"