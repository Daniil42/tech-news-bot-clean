# Tech News Bot

Telegram бот для автоматической публикации технологических новостей.

## Функции

- Парсинг RSS-лент техно-новостей
- AI-суммаризация через Gemini
- Fallback перевод через MyMemory
- Публикация в Telegram-канал

## Деплой на Railway

1. Форкните репозиторий
2. Создайте проект на Railway
3. Добавьте переменные окружения:
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHANNEL_ID`
   - `GEMINI_API_KEY` (опционально)
4. Деплойте из GitHub

## Переменные окружения

```
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHANNEL_ID=@your_channel
GEMINI_API_KEY=your_gemini_key (опционально)
PARSER_INTERVAL=30 (минуты между циклами)
```

## Источники новостей

- Hacker News
- TechCrunch
- Ars Technica
- MIT Technology Review
- Engadget