import re
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
# Загрузим стоп-слова для русского языка
nltk.download('stopwords')
russian_stop_words = stopwords.words('russian')
def remove_noise_boilerplate(input_text, min_cluster_size=2, num_clusters=5, max_noise_ratio=0.3):
    # Разбиение текста на предложения
    sentences = re.split(r'[.!?]\s|\n', input_text)
    
    if len(sentences) < 2:  # Пропуск коротких текстов
        return input_text

    # Преобразование предложений в матрицу эмбеддингов
    embeddings_matrix = text_vectorize(sentences)
    
    if embeddings_matrix.shape[0] < num_clusters:  # Проверка на достаточное количество предложений
        num_clusters = max(1, embeddings_matrix.shape[0])  # Устанавливаем минимально возможное значение кластеров
    
    # KMeans кластеризация
    kmeans_model = KMeans(n_clusters=num_clusters)
    kmeans_model.fit(embeddings_matrix)
    model_labels = kmeans_model.labels_
    model_centroids = kmeans_model.cluster_centers_
    cluster_sizes = np.bincount(model_labels)
    
    # Идентификация шумных кластеров
    is_noise = np.zeros(num_clusters, dtype=bool)
    for i, centroid in enumerate(model_centroids):
        if cluster_sizes[i] < min_cluster_size:
            continue
        distances = np.linalg.norm(embeddings_matrix[model_labels == i] - centroid, axis=1)
        median_distance = np.median(distances)
        if np.count_nonzero(distances > median_distance) / cluster_sizes[i] > max_noise_ratio:
            is_noise[i] = True
    
    # Удаление шумных предложений
    filtered_sentences = [sentence for i, sentence in enumerate(sentences) if not is_noise[model_labels[i]]]
    
    filtered_text = ' '.join(filtered_sentences)
    return filtered_text


def text_vectorize(input_text):
    # Использование TF-IDF Vectorizer для русского языка
    vectorizer = TfidfVectorizer(
        stop_words=russian_stop_words
    )
    
    # Преобразование текста в TF-IDF матрицу
    tfidf_matrix = vectorizer.fit_transform(input_text)
    
    # Преобразуем матрицу в массив numpy
    return tfidf_matrix.toarray()
