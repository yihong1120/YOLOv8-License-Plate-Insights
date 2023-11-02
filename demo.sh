# wget 

# 假設你的 .zip 檔案名稱為 file.zip
filename="Car-License-Plate.zip"

# 移除 .zip 延伸檔，取得檔案名稱
foldername="${filename%.*}"

# 建立同名資料夾
mkdir -p "$foldername"

# 解壓縮 .zip 檔案到該資料夾
unzip "$filename" -d "$foldername"

python xml2txt.py

python split.py

python train.py

python inference.py
