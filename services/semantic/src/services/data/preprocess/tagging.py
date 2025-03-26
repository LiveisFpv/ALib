import spacy

# Загрузка модели SpaCy для русского языка
nlp = spacy.load('ru_core_news_lg')

def pos_tag(input_text):
    """
    Аннотирует каждое слово в тексте частью речи для анализа научных текстов.
    """
    # Преобразование текста в документ SpaCy
    spacy_doc = nlp(input_text)
    
    # Создание строки с токенами и их частями речи
    tagged_string = []
    for token in spacy_doc:
        # Пропускаем стоп-слова (если требуется)
        if not token.is_stop:
            tagged_string.append(f"{token.text}_{token.pos_}")

    # Соединение результатов в одну строку
    tagged_result = ' '.join(tagged_string)
    return tagged_result
