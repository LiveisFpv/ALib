# Система семантического поиска научных публикаций по большим базам научных публикаций на основе информации о внешнем цитировании с использованием нейросетевых моделей
### Цель: 
### Создать систему, которая объединяет семантический анализ текста и анализ цитирований для улучшения релевантности поиска.
### Задачи:
1. Сбор и обработка данных (публикации + цитирования).
2. Построение нейросетевой модели, учитывающей семантику и цитирования.
3. Разработка архитектуры поискового движка.
4. Оценка качества поиска.
### BACKEND services
1. gateway (Точка входа API) https://github.com/LiveisFpv/VKR_gateway_service 
2. semantic_service (Основная бизнес логика) https://github.com/LiveisFpv/VKR_SemanticSearch_service
3. ELK_service (Логи и метрики) https://github.com/LiveisFpv/VKR_ELK_service
4. authorization_service (Авторизация пользователей) https://github.com/LiveisFpv/authorization_service
### Frontend services
