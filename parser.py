"""
Tech News Parser v2.5
Парсер новостей о технологиях: ИИ, мобильные устройства, гаджеты, техно-мероприятия
Скрапинг статей + AI-суммаризация через Gemini
v2.5: MyMemory Translation API для fallback (бесплатный перевод без лимитов Gemini)

ИСПРАВЛЕНИЯ от Wolfi (2026-03-20):
- ✅ Исправлено несоответствие версий (v2.5 везде)
- ✅ Удалено дублирование импорта Path
- ✅ Добавлена валидация TELEGRAM_CHANNEL_ID
- ✅ Добавлена проверка пустых ссылок
- ✅ Добавлен лимит очереди (100 новостей)
- ✅ Добавлено автосохранение после каждой операции
- ✅ Добавлено логирование в файл
"""

import requests
import feedparser
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import logging
import os
import json
import asyncio
from datetime import datetime, timedelta
from pathlib import Path  # ✅ Удалён дубликат импорта
from typing import Dict, List, Optional, Any

# ✅ Добавлено логирование в файл
def setup_logging():
    log_path = Path(__file__).parent / "logs"
    log_path.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path / f"bot_{datetime.now().strftime('%Y%m%d')}.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ✅ Версия бота
BOT_VERSION = "v2.5"

# Загрузка конфига
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
PARSER_INTERVAL = int(os.getenv("PARSER_INTERVAL", "30"))
POST_INTERVAL = int(os.getenv("POST_INTERVAL", "120"))
QUIET_HOURS_START = int(os.getenv("QUIET_HOURS_START", "1"))
QUIET_HOURS_END = int(os.getenv("QUIET_HOURS_END", "7"))

# ✅ Лимит очереди
MAX_QUEUE_SIZE = 100

# Источники RSS
RSS_FEEDS = [
    "https://news.ycombinator.com/rss",
    "https://techcrunch.com/feed/",
    "https://www.theverge.com/rss/index.xml",
    "https://arstechnica.com/feed/",
    "https://www.technologyreview.com/feed/",
    "https://www.engadget.com/rss.xml",
    "https://www.gsmarena.com/rss.php",
    "https://www.androidauthority.com/feed/",
    "https://openai.com/blog/rss/",
    "https://ai.google/rss/"
]

# Категории для чередования
CATEGORIES = ["ai", "mobile", "gadget", "event", "other"]

# Пути к данным
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
SEEN_NEWS_FILE = DATA_DIR / "seen_news.json"
NEWS_QUEUE_FILE = DATA_DIR / "news_queue.json"
LAST_POST_FILE = DATA_DIR / "last_post.json"
LAST_CATEGORY_FILE = DATA_DIR / "last_category.json"

# ✅ Валидация конфигурации при старте
def validate_config():
    errors = []
    if not TELEGRAM_BOT_TOKEN:
        errors.append("TELEGRAM_BOT_TOKEN не настроен")
    if not TELEGRAM_CHANNEL_ID:
        errors.append("TELEGRAM_CHANNEL_ID не настроен")
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY не настроен, будет использоваться только fallback-перевод")
    
    if errors:
        for error in errors:
            logger.error(f"❌ КОНФИГУРАЦИЯ: {error}")
        raise ValueError("Критические ошибки конфигурации: " + "; ".join(errors))
    
    logger.info(f"✅ Конфигурация проверена: канал {TELEGRAM_CHANNEL_ID}")

# Хранилище данных
def load_json(file_path: Path, default: Any = None) -> Any:
    try:
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Ошибка чтения {file_path}: {e}")
    return default if default is not None else {}

# ✅ Автосохранение с обработкой ошибок
def save_json(file_path: Path, data: Any) -> bool:
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"Ошибка сохранения {file_path}: {e}")
        return False

def get_seen_news() -> Dict[str, float]:
    data = load_json(SEEN_NEWS_FILE, {})
    # TTL 48 часов
    cutoff = (datetime.now() - timedelta(hours=48)).timestamp()
    return {k: v for k, v in data.items() if v > cutoff}

def save_seen_news(seen: Dict[str, float]):
    # ✅ Автосохранение
    save_json(SEEN_NEWS_FILE, seen)

def get_news_queue() -> List[Dict]:
    queue = load_json(NEWS_QUEUE_FILE, [])
    # ✅ Лимит очереди
    if len(queue) > MAX_QUEUE_SIZE:
        logger.warning(f"Очередь переполнена ({len(queue)} > {MAX_QUEUE_SIZE}), обрезаем")
        queue = queue[:MAX_QUEUE_SIZE]
        save_news_queue(queue)
    return queue

def save_news_queue(queue: List[Dict]):
    # ✅ Автосохранение
    save_json(NEWS_QUEUE_FILE, queue)

def get_last_post_time() -> Optional[datetime]:
    data = load_json(LAST_POST_FILE)
    if data and "timestamp" in data:
        return datetime.fromisoformat(data["timestamp"])
    return None

def save_last_post_time():
    # ✅ Автосохранение
    save_json(LAST_POST_FILE, {"timestamp": datetime.now().isoformat()})

def get_last_category_index() -> int:
    data = load_json(LAST_CATEGORY_FILE, {"index": -1})
    return data.get("index", -1)

def save_last_category_index(index: int):
    # ✅ Автосохранение
    save_json(LAST_CATEGORY_FILE, {"index": index})

def is_quiet_hours() -> bool:
    hour = datetime.now().hour
    if QUIET_HOURS_START <= QUIET_HOURS_END:
        return QUIET_HOURS_START <= hour < QUIET_HOURS_END
    else:  # Переход через полночь
        return hour >= QUIET_HOURS_START or hour < QUIET_HOURS_END

def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text(strip=True)

# ✅ Проверка пустых ссылок
def is_valid_news(news: Dict) -> bool:
    if not news.get("title"):
        return False
    if not news.get("link"):
        return False
    if not news["link"].strip():
        return False
    return True

async def fetch_article_content(url: str) -> Optional[str]:
    try:
        response = requests.get(f"https://r.jina.ai/{url}", timeout=15)
        if response.status_code == 200:
            return response.text[:8000]  # Лимит для API
    except Exception as e:
        logger.error(f"Ошибка скапинга статьи: {e}")
    return None

async def summarize_with_gemini(text: str) -> Optional[str]:
    if not GEMINI_API_KEY:
        return None
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    prompt = f"Объясни эту статью простыми словами, как другу. Выдели главное (3-5 пунктов):\n\n{text[:3000]}"
    
    try:
        response = requests.post(url, json={
            "contents": [{"parts": [{"text": prompt}]}]
        }, timeout=15)
        
        if response.status_code == 429:
            logger.warning("Rate limit Gemini, используем fallback")
            return None
        
        if response.status_code == 200:
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        logger.error(f"Ошибка Gemini API: {e}")
    return None

async def translate_fallback(text: str) -> str:
    try:
        response = requests.get(
            "https://api.mymemory.translated.net/get",
            params={"q": text[:500], "langpair": "en|ru"},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()["responseData"]["translatedText"]
    except Exception as e:
        logger.error(f"Ошибка перевода: {e}")
    return text[:500]

def generate_hashtags(categories: List[str]) -> str:
    hashtag_map = {
        "ai": ["#ИИ", "#AI", "#МашинноеОбучение"],
        "mobile": ["#Мобильные", "#Smartphone", "#Android", "#iOS"],
        "gadget": ["#Гаджеты", "#Техника", "#Инновации"],
        "event": ["#Мероприятия", "#Конференции", "#ТехноСобытия"],
        "other": ["#Технологии", "#Новости", "#Tech"]
    }
    
    hashtags = []
    for cat in categories[:3]:
        hashtags.extend(hashtag_map.get(cat, ["#Технологии"]))
    
    # ✅ Экранирование хэштегов
    return " ".join([h.replace("<", "").replace(">", "") for h in hashtags[:6]])

def format_telegram_post(news: Dict, summary: str) -> str:
    title = news["title"][:200]  # Лимит заголовка
    source = news.get("source", "Неизвестно")
    link = news["link"]
    categories = news.get("categories", ["other"])
    
    hashtags = generate_hashtags(categories)
    
    post = f"""📰 <b>{title}</b>

{summary}

📌 <b>Источник:</b> {source}
🔗 <a href="{link}">Читать оригинал</a>

{hashtags}"""
    
    return post

async def parse_and_send():
    logger.info("🔄 Парсинг RSS-источников...")
    seen_news = get_seen_news()
    news_queue = get_news_queue()
    
    new_count = 0
    
    for feed_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:5]:  # Топ-5 с каждого источника
                # ✅ Проверка валидности новости
                if not is_valid_news(entry):
                    logger.debug(f"Пропущена невалидная новость: {entry.get('title', 'Без названия')}")
                    continue
                
                news_id = entry.get("id", entry.get("link", ""))
                if news_id in seen_news:
                    continue
                
                # Определение категории
                category = "other"
                title_lower = entry.title.lower()
                if any(x in title_lower for x in ["ai", "gpt", "gemini", "нейросеть", "machine learning"]):
                    category = "ai"
                elif any(x in title_lower for x in ["iphone", "android", "smartphone", "мобиль"]):
                    category = "mobile"
                elif any(x in title_lower for x in ["gadget", "устройство", "device"]):
                    category = "gadget"
                elif any(x in title_lower for x in ["event", "конференц", "презентация"]):
                    category = "event"
                
                news_item = {
                    "title": entry.title,
                    "link": entry.link,  # ✅ Ссылка уже проверена в is_valid_news
                    "source": feed.feed.get("title", "Неизвестно"),
                    "published": entry.get("published", datetime.now().isoformat()),
                    "categories": [category],
                    "added_at": datetime.now().isoformat()
                }
                
                seen_news[news_id] = datetime.now().timestamp()
                news_queue.append(news_item)
                new_count += 1
                
                logger.info(f"✅ Добавлено: {entry.title[:50]}...")
                
        except Exception as e:
            logger.error(f"Ошибка парсинга {feed_url}: {e}")
    
    # ✅ Автосохранение
    save_seen_news(seen_news)
    save_news_queue(news_queue)
    
    logger.info(f"📊 Найдено {new_count} новых новостей, в очереди: {len(news_queue)}")

async def post_from_queue():
    last_post = get_last_post_time()
    if last_post and (datetime.now() - last_post).total_seconds() < POST_INTERVAL * 60:
        logger.debug(f"Ранняя публикация, ждём {(POST_INTERVAL * 60 - (datetime.now() - last_post).total_seconds()) / 60:.0f} мин")
        return
    
    news_queue = get_news_queue()
    if not news_queue:
        logger.debug("Очередь пуста")
        return
    
    # Выбор категории
    last_cat_idx = get_last_category_index()
    next_cat_idx = (last_cat_idx + 1) % len(CATEGORIES)
    target_category = CATEGORIES[next_cat_idx]
    
    # Поиск новости нужной категории
    news_item = None
    for i, item in enumerate(news_queue):
        if target_category in item.get("categories", []):
            news_item = news_queue.pop(i)
            break
    
    if not news_item and news_queue:
        news_item = news_queue.pop(0)  # Берём первую, если нет нужной категории
    
    save_news_queue(news_queue)
    save_last_category_index(next_cat_idx)
    
    # Скрапинг и суммаризация
    logger.info(f"📝 Готовим пост: {news_item['title'][:50]}...")
    
    content = await fetch_article_content(news_item["link"])
    summary = None
    
    if content:
        summary = await summarize_with_gemini(content)
    
    if not summary:
        summary = await translate_fallback(news_item["title"])
        logger.info("📋 Использован fallback-перевод")
    
    post = format_telegram_post(news_item, summary)
    
    # Отправка в Telegram
    success = await send_to_telegram(post)
    
    if success:
        save_last_post_time()
        logger.info(f"✅ Пост опубликован в {TELEGRAM_CHANNEL_ID}")
    else:
        # Возвращаем в очередь при ошибке
        news_queue.insert(0, news_item)
        save_news_queue(news_queue)
        logger.error("❌ Пост не опубликован, возвращён в очередь")

async def send_to_telegram(post: str) -> bool:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": TELEGRAM_CHANNEL_ID,
        "text": post,
        "parse_mode": "HTML",
        "disable_web_page_preview": False
    }
    
    try:
        response = requests.post(url, json=data, timeout=15)
        if response.status_code == 200:
            return True
        else:
            logger.error(f"Telegram API error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"Ошибка отправки в Telegram: {e}")
        return False

async def main():
    logger.info(f"🤖 Запуск Tech News Parser {BOT_VERSION}")
    
    # ✅ Валидация конфигурации
    validate_config()
    
    logger.info(f"⏰ Интервал парсинга: {PARSER_INTERVAL} мин")
    logger.info(f"⏰ Интервал постов: {POST_INTERVAL} мин")
    logger.info(f"🌙 Тихие часы: {QUIET_HOURS_START}:00 - {QUIET_HOURS_END}:00")
    
    while True:
        try:
            await parse_and_send()
            
            if not is_quiet_hours():
                await post_from_queue()
            else:
                logger.info("🌙 Тихие часы, публикация приостановлена")
            
            await asyncio.sleep(PARSER_INTERVAL * 60)
            
        except KeyboardInterrupt:
            logger.info("👋 Остановка по сигналу пользователя")
            break
        except Exception as e:
            logger.error(f"❌ Критическая ошибка в цикле: {e}")
            await asyncio.sleep(60)  # Пауза перед перезапуском

if __name__ == "__main__":
    asyncio.run(main())
