from clean_text import clean_text
from claster import remove_noise_boilerplate
from tagging import pos_tag
import os
from tqdm import tqdm
# Основная функция обработки текстов
def process_articles(input_dir, output_dir):
    # Сначала собираем список всех файлов для обработки
    all_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".txt"):  # Обрабатываем только текстовые файлы
                all_files.append(os.path.join(root, file))
    
    # Инициализируем tqdm для отслеживания процесса
    for file_path in tqdm(all_files, desc="Processing articles", unit="file"):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Очистка текста
        cleaned_text = clean_text(text)

        # Удаление шума
        # noise_free_text = remove_noise_boilerplate(cleaned_text)

        # Разметка частей речи
        # pos_tagged_text = pos_tag(noise_free_text)

        # Создание выходной директории
        relative_path = os.path.relpath(os.path.dirname(file_path), input_dir)
        output_path = os.path.join(output_dir, relative_path)
        os.makedirs(output_path, exist_ok=True)

        # Сохранение обработанного текста
        output_file_path = os.path.join(output_path, os.path.basename(file_path))
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)

# Запуск обработки
input_directory = "./dataset"
output_directory = "./processed_dataset"
process_articles(input_directory, output_directory)