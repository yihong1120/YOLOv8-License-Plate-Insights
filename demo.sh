# wget 

if [ "$(uname)" = "Linux" ]; then
    filename="Car-License-Plate.zip"

    # 移除 .zip 延伸檔，取得檔案名稱
    foldername="${filename%.*}"

    # 建立同名資料夾
    mkdir -p "$foldername"

    # 解壓縮 .zip 檔案到該資料夾
    unzip "$filename" -d "$foldername"

elif [ "$(uname)" = "Darwin" ]; then
    filename="Car-License-Plate.zip"
    dirname="${filename%.zip}"

    # 创建与 ZIP 文件同名的目录（如果尚不存在）
    mkdir -p "$dirname"

    # 将文件解压缩到该目录中
    unzip "$filename" -d "$dirname"

elif [ "$(expr substr $(uname -s) 1 5)" = "MINGW" ] || [ "$(expr substr $(uname -s) 1 5)" = "MSYS" ]; then
    OS="Windows"
fi

python xml2txt.py

python split.py

python train.py

# python inference.py
