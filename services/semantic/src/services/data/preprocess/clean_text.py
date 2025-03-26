import re
import emoji
import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

# Функция для очистки текста
def clean_text(input_text):    
    # Удаление HTML-тегов
    clean_text = input_text.replace("-\n","")
    clean_text = re.sub('<[^<]+?>', '', clean_text)
    
    # Удаление URL
    clean_text = re.sub(r'http\S+', '', clean_text)

    # Преобразование эмоджи в текст
    clean_text = emojis_words(clean_text)
    
    # Приведение текста к нижнему регистру
    clean_text = clean_text.lower()

    # Удаление лишних пробелов
    clean_text = re.sub(r'\s+', ' ', clean_text)

    # Преобразование символов с диакритическими знаками
    clean_text = unicodedata.normalize('NFKD', clean_text)

    # # Удаление специальных символов и знаков препинания
    # clean_text = re.sub(r'[^а-яА-ЯёЁ0-9\s]', '', clean_text)

    # Удаление стоп-слов
    stop_words = set(stopwords.words('russian'))
    tokens = word_tokenize(clean_text)
    tokens = [token for token in tokens if token not in stop_words]
    clean_text = ' '.join(tokens)

    # Удаление повторных пробелов
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    return clean_text

# Функция для преобразования эмоджи в слова
def emojis_words(text):
    # Преобразование эмоджи в словесные описания
    clean_text = emoji.demojize(text, delimiters=(" ", " "))
    # Удаление лишних символов
    clean_text = clean_text.replace(":", "").replace("_", " ")
    return clean_text

# Функция для замены цифр на текст
def replace_numbers_with_words(text):
    # Замена цифр на текст на русском языке
    from num2words import num2words
    words = []
    for word in text.split():
        try:
            # Проверяем, можно ли преобразовать слово в число
            if word.isdigit():
                try:
                    num2words(word, lang='ru')
                except:
                    words.append(word)
                    continue
                words.append(num2words(word, lang='ru'))
            else:
                words.append(word)
        except ValueError as e:
            # Если возникла ошибка при преобразовании числа, просто добавляем слово без изменений
            words.append(word)
    return ' '.join(words)
