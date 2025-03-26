import requests
from tqdm import tqdm
import re
import csv

# === ПАРАМЕТРЫ ===
SEARCH_CONCEPT_IDs = {
# "C154945302":"Artificial Intelligence", +
# "C138885662":"Philosophy", +
# "C132651083":"Climate Change", +
# "C36289849":"Social science", +
# "C144133560":"Neuroscience", +
# "C119857082":"Machine learning", +
# "C199360897":"Programming language", +
# "C185592680":"Cybersecurity", +
# "C111919701":"Operating system", +
# "C142362112":"Precision Medicine", +
# "C33923547":"Mathematics", +
# "C31972630":"Computer vision", +
# "C121332964":"Physics", +
# "C74650414":"Classical mechanics", +
# "C62520636":"Quantum mechanics", +
# "C134362201":"Mental health", +
}
OUTPUT_FILE = "openalex_dataset.csv"
NUM_PAPERS = 10000  # Количество статей (изменяй по необходимости)

# === ФУНКЦИЯ ДЛЯ ЗАПРОСА ДАННЫХ ===
def fetch_openalex_data(concept_id, num_papers=10000):
    per_page=50
    url = "https://api.openalex.org/works"
    params = {
        "filter": f"concepts.id:{concept_id},has_abstract:true",
        "per-page":per_page,  # Максимальное число за один запрос
        "page": 1,
        "mailto": "nasonovnicolai@gmail.com"
    }

    all_papers = []
    
    for papers in tqdm(range(num_papers//per_page)):
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print("Ошибка запроса:", response.text)
            break

        data = response.json()
        works = data.get("results", [])

        if not works:
            break  # Если статьи закончились

        all_papers.extend(works)
        params["page"] += 1  # Переход на следующую страницу

    return all_papers[:num_papers]  # Обрезаем до нужного количества


# === ФУНКЦИЯ СОХРАНЕНИЯ В CSV ===
def save_to_csv(papers, filename):
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ID", 
                         "Title", 
                         "Abstract", 
                         "Concepts", 
                         "Year", 
                         "Referenced_works", 
                         "Related_works", 
                         "Cited_by_count",
                         "Best_oa_location"])

        for paper in papers:
            writer.writerow([
                paper["id"],
                paper["title"] or "N/A",
                reconstruct_abstract(paper.get("abstract_inverted_index", {})),
                paper.get("concepts"),
                paper["publication_year"],
                "; ".join(paper.get("referenced_works", [])),  # Список цитируемых работ
                "; ".join(paper.get("related_works", [])),  # Список связанных работ
                paper.get("cited_by_count", 0),
                paper["best_oa_location"] or "N/A"
            ])

def clean_text(text):
    text = text.strip()  # Удаление пробелов в начале и в конце строки
    text = text.replace('\n', ' ')  # Замена переносов строк на пробел
    text = text.replace('\r', ' ')  # Удаление символов возврата каретки
    text = text.replace('\t', ' ')  # Удаление табуляций
    text = re.sub(r'\s+', ' ', text)  # Замена нескольких пробелов на один
    return text

def reconstruct_abstract(abstract_inverted_index):
    if not abstract_inverted_index:
        return "No abstract available"
    
    words = []
    for word, positions in abstract_inverted_index.items():
        for pos in positions:
            words.append((pos, word))
    
    words.sort()  # Сортируем по позициям
    abstract = " ".join(word for _, word in words)
    abstract=clean_text(abstract)
    return abstract

# === СКАЧИВАЕМ И СОХРАНЯЕМ ДАННЫЕ ===
for search_id in SEARCH_CONCEPT_IDs:
    papers = fetch_openalex_data(search_id, NUM_PAPERS)
    out_file=f"{SEARCH_CONCEPT_IDs[search_id]}.csv"
    if papers:
        save_to_csv(papers, out_file)
        print(f"Датасет сохранён в {out_file}")
    else:
        print("Не удалось получить данные.")
