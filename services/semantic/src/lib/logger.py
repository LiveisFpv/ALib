import logging
import sys
import traceback
from logstash_async.handler import AsynchronousLogstashHandler


class CustomFormatter(logging.Formatter):
    """Кастомный JSON-форматтер для логов Logstash"""
    
    def format(self, record):
        log_record = {
            "@timestamp": self.formatTime(record),
            "@version": 1,
            "message": record.getMessage(),
            "logger_name": record.name,
            "level": record.levelname,
            "level_value": record.levelno,
            "thread_name": record.threadName,
            "host": record.pathname,
            "lineno": record.lineno,
        }

        # Добавляем extra-данные, если они есть
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            log_record.update(record.extra)

        # Обработка исключений
        if record.exc_info:
            exc_type, exc_value, exc_traceback = record.exc_info
            log_record.update({
                "error_type": str(exc_type.__name__),
                "error_message": str(exc_value),
                "stack_trace": traceback.format_exc(),
            })

        return str(log_record)  # Logstash понимает JSON в виде строки


class Logger:
    """Логгер для всего проекта с поддержкой Logstash и JSON"""

    def __init__(self, logstash_host: str, logstash_port: int, logger_name: str, log_level:str|int=logging.INFO):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(log_level)

        # Консольный логгер
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(CustomFormatter())
        self.logger.addHandler(console_handler)
        if logstash_host !='' and logstash_port!=0:
            # Logstash логгер
            try:
                logstash_handler = AsynchronousLogstashHandler(logstash_host, logstash_port, database_path=None)
                logstash_handler.setFormatter(CustomFormatter())
                self.logger.addHandler(logstash_handler)
            except Exception:
                RuntimeWarning("Failed to connect to Logstash")
        else:
            RuntimeWarning("Failed to connect to Logstash")

    def log(self, level, message, **extra):
        """Формирует лог с дополнительными полями"""
        self.logger.log(level, message, extra={"extra": extra})

    def info(self, message, **extra):
        """Wrapper for logging info messages"""
        self.log(logging.INFO, message, **extra)

    def warning(self, message, **extra):
        """Wrapper for logging warning messages"""
        self.log(logging.WARNING, message, **extra)

    def error(self, message, **extra):
        """Wrapper for logging error messages"""
        self.log(logging.ERROR, message, **extra)


