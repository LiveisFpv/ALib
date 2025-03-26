import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import os
import pandas as pd
import os
import tqdm
from src.lib.logger import Logger

class Model:
    def __init__(self, model_name, vector_dir, logger:Logger):
        self.__model_name = model_name
        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Загрузка модели и токенизатора
        logger.info(f"Starting {model_name}")
        self.__model = AutoModel.from_pretrained(self.__model_name).to(self.__device)
        self.__tokenizer = AutoTokenizer.from_pretrained(self.__model_name)
        
        logger.info(f"Started model {model_name}")
        # Загружаем ВСЕ `.pth` файлы из директории
        self.__vectors = self.__load_vectors(vector_dir)

    def __average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """Среднее пуллингование скрытых состояний с учетом маски."""
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def __tokenize_text(self, text):
        """Препроцессинг текста и получение токенов."""
        inputs = self.__tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(self.__device) for k, v in inputs.items()}
        return inputs
    
    def __get_embedding(self, text):
        """Получение эмбеддинга текста."""
        encoded_input = self.__tokenize_text(text)
        with torch.no_grad():
            model_output = self.__model(**encoded_input)
        
        embeddings = self.__average_pool(model_output.last_hidden_state, encoded_input["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)  # Нормализация
        
        return embeddings.cpu()

    def __load_vectors(self, vector_dir):
        """Загрузка предобработанных векторов из ВСЕХ `.pth` файлов в директории."""
        all_vectors = []
        
        for file_name in os.listdir(vector_dir):
            if file_name.endswith(".pth"):  # Проверяем, что файл - .pth
                file_path = os.path.join(vector_dir, file_name)
                loaded_vectors = torch.load(file_path)
                
                # Добавляем данные из файла в общий список
                for item in loaded_vectors:
                    all_vectors.append({
                        "id": item["id"],
                        "vector": item["vector"],
                        "file_path": item["file_path"],  
                        "source_file": file_name  # Добавляем имя файла, из которого загружен вектор
                    })

        return all_vectors
    
    def get_similar_texts(self, query_text, top_n=5):
        """Поиск наиболее похожих текстов."""
        query_embedding = self.__get_embedding(query_text)  # Эмбеддинг запроса
        query_embedding = query_embedding.squeeze(0)  # Убираем лишнию размерность
        
        # Вычисляем косинусное сходство
        similarities = []
        for item in self.__vectors:
            vector = item["vector"]
            similarity = F.cosine_similarity(query_embedding, vector, dim=0)
            similarities.append((item["id"], item["file_path"], item["source_file"], similarity.item()))
        
        # Сортируем по убыванию схожести
        similarities.sort(key=lambda x: x[3], reverse=True)
        
        # Возвращаем top_n наиболее похожих
        return similarities[:top_n]

    def add_paper(self, paper, data_path, vec_path):
        """Добавление новой публикации."""
        # TODO: реализовать добавление новой публикации

    # Обработка csv файлов в pth

    def __load_articles_from_csv(self,file_path):
        """Загружает статьи из CSV-файла."""
        articles = []
        df = pd.read_csv(file_path)
        
        required_columns = {"ID", "Title", "Abstract"}
        if not required_columns.issubset(df.columns):
            print(f"⚠️ Пропущен файл {file_path}: отсутствуют нужные столбцы")
            return []
        
        for _, row in df.iterrows():
            text = f"{row['Abstract']}"
            articles.append({
                "id": row["ID"],
                "text": text,
                "file_path": file_path
            })
        return articles

    def __tokenize_articles(self,articles, batch_size=16):
        """Токенизация и векторизация статей."""
        all_vectors = []
        
        for i in tqdm(range(0, len(articles), batch_size), desc="Vectorizing articles"):
            batch = articles[i:i+batch_size]
            texts = [article["text"] for article in batch]
            
            encoded_input = self.__tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
            encoded_input = {k: v.to(self.__device) for k, v in encoded_input.items()}
            
            with torch.no_grad():
                model_output = model(**encoded_input)
            
            embeddings = self.__average_pool(model_output.last_hidden_state, encoded_input["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            for j, article in enumerate(batch):
                all_vectors.append({
                    "id": article["id"],
                    "vector": embeddings[j].cpu(),
                    "file_path": article["file_path"]
                })
        
        return all_vectors

    def __save_vectors_as_pth(vectors, output_file):
        """Сохраняет вектора в отдельный PTH-файл."""
        torch.save(vectors, output_file)

    def process_directory(self,dataset_path, output_path):
        """Обрабатывает все CSV-файлы в папке и сохраняет вектора отдельно."""
        os.makedirs(output_path, exist_ok=True)
        
        for file_name in os.listdir(dataset_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(dataset_path, file_name)
                output_file_pth = os.path.join(output_path, f"{os.path.splitext(file_name)[0]}.pth")
                
                articles = self.__load_articles_from_csv(file_path)
                if articles:
                    vectors = self.__tokenize_articles(articles)
                    self.__save_vectors_as_pth(vectors, output_file_pth)
                    print(f"✅ Вектора для {file_name} сохранены в {output_file_pth}")
                else:
                    print(f"⚠️ Нет статей для обработки в {file_name}")
        

if __name__ == "__main__":
    vector_directory = "./data/vectorized/openAlex/"  # Путь к директории с `.pth` файлами
    model = Model("intfloat/multilingual-e5-large", vector_directory)

    query = "Machine learning for scientific research"
    top_matches = model.get_similar_texts(query, top_n=5)

    print("🔍 Наиболее похожие тексты:")
    for match in top_matches:
        print(f"📄 ID: {match[0]}, Файл: {match[1]}, Источник: {match[2]}, Сходство: {match[3]:.4f}")
