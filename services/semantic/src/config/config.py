import os
from dotenv import load_dotenv

# Загружаем переменные окружения из .env
load_dotenv()

# Читаем переменные
LOGSTASH_HOST = os.getenv("LOGSTASH_HOST", "localhost")
LOGSTASH_PORT = int(os.getenv("LOGSTASH_PORT", 5044))
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")
