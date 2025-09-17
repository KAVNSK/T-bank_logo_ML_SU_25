import os
import shutil
from sys import exit


source_folder = "C:/Users/alexk/OneDrive/Документы/GitHub/proj/dataset_full/data_sirius" # откуда берутся изображения
layout_folder = "C:/Users/alexk/OneDrive/Документы/GitHub/proj/dataset_full/labels" # где искать текстовые лейблы
images_folder = "C:/Users/alexk/OneDrive/Документы/GitHub/proj/dataset_pre/post_images/train" # куда копировать изображения
labels_folder = "C:/Users/alexk/OneDrive/Документы/GitHub/proj/dataset_pre/post_labels/train" # куда копировать/создавать лейблы

import ctypes
from functools import cmp_to_key
stop_file = "1bc165558633443c29254861baa8a621.jpg"

StrCmpLogicalW = ctypes.windll.shlwapi.StrCmpLogicalW
def windows_compare(a, b):
    return StrCmpLogicalW(a, b)

layout_files = []
if os.path.isdir(layout_folder):
    layout_files = [f for f in os.listdir(layout_folder)
                    if os.path.isfile(os.path.join(layout_folder, f))]
else:
    print(f"Внимание: папка layout '{layout_folder}' не найдена. Все лейблы будут создаваться пустыми.")
    exit(0)

for filename in sorted(os.listdir(source_folder), key=cmp_to_key(windows_compare)):
    if filename == stop_file:
        print(f"Встречен файл '{stop_file}', копирование остановлено.")
        break

    src_image_path = os.path.join(source_folder, filename)

    if not os.path.isfile(src_image_path):
        continue

    dest_image_path = os.path.join(images_folder, filename)
    shutil.copy2(src_image_path, dest_image_path)
    print(f"Скопировано изображение: {filename}")


    stem = os.path.splitext(filename)[0]

    candidates = [f for f in layout_files if os.path.splitext(f)[0] == stem]

    if candidates:
        txt_candidate = next((f for f in candidates if f.lower().endswith('.txt')), None)
        chosen = txt_candidate if txt_candidate else candidates[0]
        src_label_path = os.path.join(layout_folder, chosen)

        chosen_ext = os.path.splitext(chosen)[1]
        dest_label_name = f"{stem}{chosen_ext}"
        dest_label_path = os.path.join(labels_folder, dest_label_name)

        shutil.copy2(src_label_path, dest_label_path)
        print(f"  -> Найден и скопирован лейбл: {chosen} -> {dest_label_name}")

    else:

        dest_label_path = os.path.join(labels_folder, f"{stem}.txt")
        open(dest_label_path, "w", encoding="utf-8").close() # suka
        print(f"  -> Лейбл не найден. Создан пустой файл: {os.path.basename(dest_label_path)}")


print("Готово.", num)
