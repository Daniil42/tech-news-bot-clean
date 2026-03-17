"""
Tech News Parser v2.5
Парсер новостей о технологиях: ИИ, мобильные устройства, гаджеты, техно-мероприятия
Скрыпинг статей + AI-суммаризация через Gemini
v2.5: MyMemory Translation API для fallback (бесплатный перевод без лимитов Gemini)
"""

import logging
import os
import json
import asyncio
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional

import requests
import feedparser
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pathlib import Path

# Загружаем .env из директории скрипта
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

# Токены берём из переменных окружения
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")  # Для AI-суммаризации

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Конфигурация
PARSER_INTERVAL = int(os.getenv("PARSER_INTERVAL", "30"))  # минут
POST_INTERVAL = int(os.getenv("POST_INTERVAL", "120"))  # минут между постами (2 часа по умолчанию)

# Ночной перерыв (часы в timezone сервера)
QUIET_HOURS_START = int(os.getenv("QUIET_HOURS_START", "1"))   # 1:00
QUIET_HOURS_END = int(os.getenv("QUIET_HOURS_END", "7"))       # 7:00


def is_quiet_hours() -> bool:
    """Проверка: сейчас ночное время (публикация приостановлена)"""
    hour = datetime.now().hour
    if QUIET_HOURS_START <= QUIET_HOURS_END:
        return QUIET_HOURS_START <= hour < QUIET_HOURS_END
    else:  # Переход через полночь (например, 23:00-6:00)
        return hour >= QUIET_HOURS_START or hour < QUIET_HOURS_END


def fetch_article_content(url: str) -> str:
    """
    Скрыпинг полной статьи через Jina AI Reader (бесплатно).
    Jina Reader извлекает чистый текст статьи без рекламы.
    """
    try:
        jina_url = f"https://r.jina.ai/{url}"
        headers = {
            "Accept": "text/plain"
        }
        
        response = requests.get(jina_url, headers=headers, timeout=15)
        response.raise_for_status()
        
        content = response.text
        
        # Проверяем, что получили реальный контент
        if len(content) < 200:
            logger.warning(f"Слишком короткий контент: {len(content)} символов")
            return ""
        
        # Ограничиваем размер для Gemini
        if len(content) > 8000:
            content = content[:8000]
        
        logger.info(f"✓ Получено {len(content)} символов статьи")
        return content
        
    except Exception as e:
        logger.warning(f"⚠ Ошибка скрыпинга {url}: {e}")
        return ""


def translate_with_mymemory(text: str) -> str:
    """
    Перевод через MyMemory Translation API (бесплатно, без ключа).
    Лимит: 5000 слов/день.
    """
    if not text:
        return ""
    
    try:
        # MyMemory API: https://mymemory.translated.net/doc/
        url = f"https://api.mymemory.translated.net/get?q={requests.utils.quote(text)}&langpair=en|ru"
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        result = response.json()
        
        if result.get("responseStatus") == 200:
            translation = result["responseData"]["translatedText"]
            logger.info(f"✓ MyMemory перевод: {translation[:50]}...")
            return translation
        
        logger.warning(f"⚠ MyMemory ошибка: {result.get('responseDetails', 'Unknown')}")
        return ""
        
    except Exception as e:
        logger.error(f"⚠ Ошибка MyMemory перевода: {e}")
        return ""


def translate_fallback(title: str, summary: str = "") -> str:
    """
    Fallback: перевод через MyMemory API когда Gemini недоступен.
    Переводит заголовок + summary если есть.
    """
    if not title:
        return ""
    
    # Пробуем MyMemory (бесплатный, без лимитов как у Gemini)
    translated_title = translate_with_mymemory(title)
    
    if translated_title:
        result = f"📰 {translated_title}"
        
        if summary:
            translated_summary = translate_with_mymemory(summary[:500])  # лимит 500 символов
            if translated_summary:
                result += f"\n\n{translated_summary}"
        
        result += "\n\n_Перевод автоматически. Оригинал по ссылке ниже._"
        logger.info(f"✓ Fallback перевод выполнен через MyMemory")
        return result
    
    # Если MyMemory не сработал, пробуем Gemini
    if GEMINI_API_KEY:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
            
            prompt = f"""Переведи этот заголовок техно-новости на русский язык. 
Только перевод, без комментариев.

Заголовок: {title}

Перевод:"""

            data = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 200
                }
            }
            
            response = requests.post(url, json=data, timeout=15)
            
            if response.status_code == 429:
                logger.warning("⚠ Gemini rate limit в fallback переводе")
                return f"📰 {title}\n\n_English article — translation unavailable_"
            
            response.raise_for_status()
            result = response.json()
            
            if "candidates" in result and len(result["candidates"]) > 0:
                translation = result["candidates"][0]["content"]["parts"][0]["text"].strip()
                logger.info(f"✓ Gemini fallback перевод: {translation[:50]}...")
                return f"📰 {translation}\n\n_Оригинальная статья на английском_"
            
        except Exception as e:
            logger.error(f"⚠ Ошибка Gemini fallback: {e}")
    
    return f"📰 {title}\n\n_English article_"


def summarize_with_gemini(title: str, content: str, retry_count: int = 0) -> str:
    """
    AI-суммаризация статьи через Gemini Flash (бесплатный).
    Возвращает структурированный пост на русском языке.
    При ошибке 429 (rate limit) — retry с задержкой или fallback перевод.
    """
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY не настроен, пропускаем суммаризацию")
        return ""
    
    if not content or len(content) < 100:
        return ""
    
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
        
        prompt = f"""Ты — техно-энтузиаст, который объясняет новости друзьям простым языком.

Заголовок: {title}

Статья:
{content}

Напиши краткое объяснение (150-200 слов) на русском:

1. ЧТО произошло? (1 предложение)
2. ПОЧЕМУ это интересно? (ключевые моменты, 2-3 предложения)
3. КАК это влияет на людей? (примеры, аналогии)

Пиши как человек объясняет другу за кофе — понятно, без терминов, с примерами из жизни. 
Не "переводи" статью — объясни её суть.

Эмодзи: 1-2 уместных.
Хэштеги и ссылку НЕ добавлять."""

        data = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 1000
            }
        }
        
        response = requests.post(url, json=data, timeout=30)
        
        # Обработка 429 (rate limit)
        if response.status_code == 429:
            if retry_count < 2:
                wait_time = (retry_count + 1) * 10  # 10s, 20s
                logger.warning(f"⚠ Gemini rate limit. Retry {retry_count + 1} через {wait_time}s...")
                time.sleep(wait_time)
                return summarize_with_gemini(title, content, retry_count + 1)
            else:
                logger.warning("⚠ Gemini rate limit (3 попытки). Используем fallback перевод...")
                return translate_fallback(title, content[:500])
        
        response.raise_for_status()
        
        result = response.json()
        
        if "candidates" in result and len(result["candidates"]) > 0:
            text = result["candidates"][0]["content"]["parts"][0]["text"]
            logger.info(f"✓ AI-суммаризация выполнена: {len(text)} символов")
            return text.strip()
        
        logger.warning("⚠ Пустой ответ от Gemini")
        return ""
        
    except Exception as e:
        logger.error(f"⚠ Ошибка Gemini: {e}")
        return translate_fallback(title, content[:500] if content else "")


def format_telegram_post(news: Dict, ai_summary: str = "") -> str:
    """Форматирование новости для Telegram (используем HTML вместо Markdown)"""
    title = news["title"].strip()
    
    # Экранируем HTML символы в заголовке
    title_escaped = title.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    
    # Если есть AI-суммаризация, используем её
    if ai_summary:
        # Экранируем AI summary тоже
        summary_escaped = ai_summary.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        post = f"📰 <b>{title_escaped}</b>\n\n{summary_escaped}\n\n"
    else:
        # Fallback на обычный summary
        post = f"📰 <b>{title_escaped}</b>\n\n"
        if news.get("summary"):
            summary = news["summary"].strip()
            if len(summary) > 400:
                summary = summary[:397] + "..."
            summary_escaped = summary.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            post += f"{summary_escaped}\n\n"
    
    source_escaped = news['source'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    post += f"📌 <b>Источник:</b> {source_escaped}\n"
    post += f"🔗 <a href=\"{news['link']}\">Читать оригинал</a>\n\n"
    
    # Хэштеги (не экранируем, это простые строки)
    hashtags = generate_hashtags(news["title"])
    post += hashtags
    
    return post


def generate_hashtags(title: str) -> str:
    """Генерация хэштегов на основе заголовка"""
    hashtags = []
    title_lower = title.lower()

    if any(word in title_lower for word in ["openai", "gpt", "chatgpt", "o1", "o3"]):
        hashtags.append("#OpenAI")
    if any(word in title_lower for word in ["anthropic", "claude"]):
        hashtags.append("#Claude")
    if any(word in title_lower for word in ["google", "gemini", "deepmind"]):
        hashtags.append("#Google")
    if any(word in title_lower for word in ["apple", "iphone", "ipad", "mac"]):
        hashtags.append("#Apple")
    if any(word in title_lower for word in ["samsung", "galaxy"]):
        hashtags.append("#Samsung")
    if any(word in title_lower for word in ["midjourney", "stable diffusion", "dall"]):
        hashtags.append("#GenerativeAI")
    if any(word in title_lower for word in ["llm", "language model"]):
        hashtags.append("#LLM")
    if any(word in title_lower for word in ["ai", "artificial intelligence", "ии", "нейросеть"]):
        hashtags.append("#ИИ")
    if any(word in title_lower for word in ["vr", "ar", "vision pro", "quest", "virtual reality"]):
        hashtags.append("#VR_AR")
    if any(word in title_lower for word in ["robot", "robotics", "робот"]):
        hashtags.append("#Роботы")
    if any(word in title_lower for word in ["startup", "стартап", "funding", "investment"]):
        hashtags.append("#Стартапы")

    hashtags.append("#Технологии")
    hashtags.append("#Новости")

    return " ".join(hashtags)


# Ключевые слова для фильтрации новостей
TECH_KEYWORDS = [
    # ИИ и машинное обучение
    "artificial intelligence", "AI", "machine learning", "ML",
    "deep learning", "neural network", "GPT", "LLM",
    "ChatGPT", "Claude", "Gemini", "Midjourney", "DALL-E",
    "OpenAI", "Anthropic", "DeepMind",
    "генеративный", "нейросеть", "ИИ", "машинное обучение",
    
    # Мобильные технологии и телефоны
    "smartphone", "iPhone", "Android", "Samsung", "Google Pixel",
    "OnePlus", "Xiaomi", "Huawei", "mobile", "5G", "6G",
    "телефон", "смартфон", "мобильный",
    
    # Техно-мероприятия
    "MWC", "Mobile World Congress", "CES", "WWDC", "Google I/O",
    "Samsung Unpacked", "Apple Event", "Microsoft Build",
    "техно-мероприятие", "конференция", "выставка технологий",
    
    # Гаджеты и устройства
    "gadget", "wearable", "smartwatch", "tablet", "laptop",
    "VR", "AR", "virtual reality", "augmented reality",
    "Metaverse", "Quest", "Vision Pro",
    "гаджет", "устройство", "носимый",
    
    # Инновации и стартапы
    "startup", "innovation", "tech startup", "funding", "venture capital",
    "инновация", "стартап", "инвестиции",
    
    # Роботы
    "robot", "robotics", "humanoid", "automation",
    "робот", "робототехника",
    
    # Общие технологии
    "technology", "tech", "software", "hardware", "app",
    "cloud", "cybersecurity", "blockchain", "crypto",
    "технология", "программное обеспечение", "приложение"
]

# Источники новостей
SOURCES = {
    "hackernews": {
        "name": "Hacker News",
        "url": "https://news.ycombinator.com/rss",
        "type": "rss"
    },
    "techcrunch": {
        "name": "TechCrunch",
        "url": "https://techcrunch.com/feed/",
        "type": "rss"
    },
    "theverge": {
        "name": "The Verge",
        "url": "https://www.theverge.com/rss/index.xml",
        "type": "rss"
    },
    "ars_technica": {
        "name": "Ars Technica",
        "url": "https://feeds.arstechnica.com/arstechnica/index",
        "type": "rss"
    },
    "mit_technology_review": {
        "name": "MIT Technology Review",
        "url": "https://www.technologyreview.com/feed/",
        "type": "rss"
    },
    "engadget": {
        "name": "Engadget",
        "url": "https://www.engadget.com/rss.xml",
        "type": "rss"
    },
    "gsmarena": {
        "name": "GSMArena",
        "url": "https://www.gsmarena.com/rss-news.php3",
        "type": "rss"
    },
    "android_authority": {
        "name": "Android Authority",
        "url": "https://www.androidauthority.com/feed/",
        "type": "rss"
    },
    "openai": {
        "name": "OpenAI Blog",
        "url": "https://openai.com/blog/rss/",
        "type": "rss"
    },
    "google_ai": {
        "name": "Google AI",
        "url": "https://ai.google/rss.xml",
        "type": "rss"
    }
}

# Хранилище
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
SEEN_FILE = DATA_DIR / "seen_news.json"
QUEUE_FILE = DATA_DIR / "news_queue.json"
LAST_POST_FILE = DATA_DIR / "last_post.json"
CATEGORY_FILE = DATA_DIR / "last_category.json"

# Категории для чередования
CATEGORIES = ["ai", "mobile", "gadget", "event", "other"]
CATEGORY_KEYWORDS = {
    "ai": ["AI", "artificial intelligence", "machine learning", "GPT", "LLM", "ChatGPT", "Claude", "Gemini", "нейросеть", "ИИ", "OpenAI", "Anthropic", "DeepMind"],
    "mobile": ["iPhone", "Android", "Samsung", "Google Pixel", "smartphone", "телефон", "смартфон", "5G", "6G", "mobile", "MWC", "Mobile World Congress"],
    "gadget": ["VR", "AR", "Vision Pro", "Quest", "smartwatch", "tablet", "laptop", "гаджет", "устройство", "wearable"],
    "event": ["CES", "WWDC", "Google I/O", "Apple Event", "Samsung Unpacked", "Microsoft Build", "конференция", "выставка"],
    "other": []
}


def load_seen_news() -> dict:
    """
    Загрузить уже виденные новости с временными метками.
    Возвращает dict: {news_id: timestamp}
    """
    if SEEN_FILE.exists():
        try:
            with open(SEEN_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Поддержка старого формата (просто список)
                if isinstance(data, list):
                    return {item: datetime.now().isoformat() for item in data}
                return data
        except Exception as e:
            logger.warning(f"⚠ Ошибка загрузки seen_news.json: {e}")
            return {}
    return {}


def save_seen_news(seen: dict):
    """
    Сохранить виденные новости с временными метками.
    Автоматически удаляет записи старше 48 часов.
    """
    # Очищаем старые записи (> 48 часов)
    cutoff = datetime.now() - timedelta(hours=48)
    cleaned = {
        news_id: ts 
        for news_id, ts in seen.items() 
        if datetime.fromisoformat(ts) > cutoff
    }
    
    with open(SEEN_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2)
    
    removed = len(seen) - len(cleaned)
    if removed > 0:
        logger.info(f"🧹 Удалено {removed} старых записей из seen_news.json")


def load_queue() -> List[Dict]:
    """Загрузить очередь новостей"""
    if QUEUE_FILE.exists():
        with open(QUEUE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_queue(queue: List[Dict]):
    """Сохранить очередь новостей"""
    with open(QUEUE_FILE, "w", encoding="utf-8") as f:
        json.dump(queue, f, indent=2, ensure_ascii=False)


def get_last_post_time() -> Optional[datetime]:
    """Получить время последней публикации"""
    if LAST_POST_FILE.exists():
        with open(LAST_POST_FILE, "r") as f:
            return datetime.fromisoformat(f.read().strip())
    return None


def save_last_post_time(dt: datetime):
    """Сохранить время последней публикации"""
    with open(LAST_POST_FILE, "w") as f:
        f.write(dt.isoformat())


def get_last_category() -> str:
    """Получить последнюю категорию"""
    if CATEGORY_FILE.exists():
        with open(CATEGORY_FILE, "r") as f:
            return f.read().strip()
    return "other"


def save_last_category(category: str):
    """Сохранить последнюю категорию"""
    with open(CATEGORY_FILE, "w") as f:
        f.write(category)


def detect_category(title: str, summary: str = "") -> str:
    """Определить категорию новости"""
    text = f"{title} {summary}".lower()
    
    for category, keywords in CATEGORY_KEYWORDS.items():
        if category == "other":
            continue
        for keyword in keywords:
            if keyword.lower() in text:
                return category
    
    return "other"


def get_next_category() -> str:
    """Получить следующую категорию для публикации (по кругу)"""
    last = get_last_category()
    try:
        idx = CATEGORIES.index(last)
        next_idx = (idx + 1) % len(CATEGORIES)
        return CATEGORIES[next_idx]
    except ValueError:
        return CATEGORIES[0]


def select_news_by_category(queue: List[Dict], preferred_category: str) -> Optional[Dict]:
    """Выбрать новость по категории"""
    # Сначала ищем новость нужной категории
    for i, news in enumerate(queue):
        if news.get("category") == preferred_category:
            return queue.pop(i)
    
    # Если нет, берём первую с категорией, отличной от последней
    for i, news in enumerate(queue):
        if news.get("category") != get_last_category():
            return queue.pop(i)
    
    # Если все одинаковые или очередь пуста, берём первую
    if queue:
        return queue.pop(0)
    
    return None


def contains_tech_keywords(title: str, summary: str = "") -> bool:
    """Проверить, содержит ли новость технологические ключевые слова"""
    text = f"{title} {summary}".lower()
    return any(keyword.lower() in text for keyword in TECH_KEYWORDS)


def parse_rss_feed(source_key: str, limit: int = 10) -> List[Dict]:
    """Парсинг RSS ленты"""
    source = SOURCES.get(source_key)
    if not source:
        logger.error(f"Источник {source_key} не найден")
        return []

    logger.info(f"Парсинг {source['name']}...")

    try:
        response = requests.get(source["url"], timeout=10)
        response.raise_for_status()

        feed = feedparser.parse(response.content)
        news_items = []

        for entry in feed.entries[:limit]:
            title = entry.get("title", "")
            link = entry.get("link", "")
            summary = entry.get("summary", entry.get("description", ""))
            published = entry.get("published", "")

            # Очищаем summary от HTML тегов
            if summary:
                soup = BeautifulSoup(summary, "html.parser")
                summary = soup.get_text()[:300]

            news_items.append({
                "source": source["name"],
                "source_key": source_key,
                "title": title,
                "link": link,
                "summary": summary,
                "published": published,
                "parsed_at": datetime.now().isoformat()
            })

        logger.info(f"Найдено {len(news_items)} новостей из {source['name']}")
        return news_items

    except Exception as e:
        logger.error(f"Ошибка парсинга {source['name']}: {e}")
        return []


def filter_tech_news(news_items: List[Dict]) -> List[Dict]:
    """Фильтрация новостей по технологической тематике"""
    filtered = []
    for item in news_items:
        if contains_tech_keywords(item["title"], item["summary"]):
            filtered.append(item)
    return filtered


async def send_to_telegram(post: str):
    """Отправка поста в Telegram"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHANNEL_ID:
        logger.warning("Telegram токен или ID канала не настроены")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    # Проверяем длину поста (лимит Telegram 4096 символов)
    if len(post) > 4096:
        logger.warning(f"Пост слишком длинный: {len(post)} символов, обрезаем до 4096")
        post = post[:4093] + "..."
    
    data = {
        "chat_id": TELEGRAM_CHANNEL_ID,
        "text": post,
        "parse_mode": "HTML",
        "disable_web_page_preview": False
    }

    try:
        response = requests.post(url, json=data, timeout=15)
        response.raise_for_status()
        result = response.json()
        logger.info(f"✓ Пост отправлен в Telegram: {result.get('result', {}).get('message_id', 'N/A')}")
        return True
    except requests.exceptions.HTTPError as e:
        logger.error(f"Ошибка Telegram API: {e}")
        logger.error(f"Текст ошибки: {e.response.text if e.response else 'N/A'}")
        return False
    except Exception as e:
        logger.error(f"Ошибка отправки в Telegram: {e}")
        return False


def save_news(news_items: List[Dict]):
    """Сохранение новостей в файл"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = DATA_DIR / f"news_{timestamp}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(news_items, f, indent=2, ensure_ascii=False)

    logger.info(f"Новости сохранены в {filename}")


async def parse_and_send(sources: List[str] = None):
    """Парсинг и добавление новостей в очередь"""
    if sources is None:
        sources = list(SOURCES.keys())

    seen_news = load_seen_news()  # Теперь dict: {news_id: timestamp}
    queue = load_queue()

    all_news = []
    new_news = []

    for source_key in sources:
        news_items = parse_rss_feed(source_key)
        all_news.extend(news_items)

    tech_news = filter_tech_news(all_news)
    logger.info(f"Найдено {len(tech_news)} техно-новостей")

    for news in tech_news:
        news_id = news["link"]
        if news_id not in seen_news:
            news["category"] = detect_category(news["title"], news["summary"])
            new_news.append(news)
            seen_news[news_id] = datetime.now().isoformat()  # Сохраняем с timestamp

    logger.info(f"Новых новостей: {len(new_news)}")

    save_seen_news(seen_news)  # Авто-очистка старых записей

    if new_news:
        queue.extend(new_news)
        save_queue(queue)
        save_news(new_news)
        logger.info(f"Очередь: {len(queue)} новостей")

    return queue


async def post_from_queue():
    """Опубликовать одну новость из очереди с AI-суммаризацией"""
    # Проверяем ночной перерыв
    if is_quiet_hours():
        logger.info(f"🌙 Ночное время ({QUIET_HOURS_START}:00-{QUIET_HOURS_END}:00), публикация приостановлена")
        return False
    
    queue = load_queue()
    
    if not queue:
        logger.info("Очередь пуста, нечего публиковать")
        return False
    
    # Проверяем время последней публикации
    last_post = get_last_post_time()
    if last_post:
        time_since_last = datetime.now() - last_post
        min_interval = timedelta(minutes=POST_INTERVAL)
        
        if time_since_last < min_interval:
            wait_seconds = (min_interval - time_since_last).total_seconds()
            logger.info(f"До следующей публикации: {int(wait_seconds)} сек")
            return False
    
    # Получаем следующую категорию
    next_category = get_next_category()
    logger.info(f"Ищем новость категории: {next_category}")
    
    # Выбираем новость
    news = select_news_by_category(queue, next_category)
    
    if not news:
        logger.info("Нет подходящих новостей")
        return False
    
    # AI-суммаризация
    ai_summary = ""
    content = ""
    
    if GEMINI_API_KEY:
        logger.info(f"🔄 Скрапинг статьи: {news['link']}")
        content = fetch_article_content(news["link"])
        
        if content:
            logger.info("🤖 AI-суммаризация...")
            ai_summary = summarize_with_gemini(news["title"], content)
        else:
            logger.warning("⚠️ Скрапинг не удался, используем fallback перевод")
    
    # Если нет AI-суммаризации (rate limit или ошибка скрапинга) — переводим заголовок
    if not ai_summary:
        logger.info("🔄 Fallback: перевод заголовка через MyMemory...")
        ai_summary = translate_fallback(news["title"], news.get("summary", "")[:500] if news.get("summary") else "")
    
    # Формируем и отправляем пост
    post = format_telegram_post(news, ai_summary)
    success = await send_to_telegram(post)
    
    if success:
        save_last_post_time(datetime.now())
        save_last_category(news.get("category", "other"))
        save_queue(queue)
        logger.info(f"✅ Опубликовано: {news['title'][:50]}...")
        return True
    
    return False


async def main():
    """Основная функция"""
    logger.info("🤖 Запуск Tech News Parser v2.6 (AI-powered + schedule)")
    logger.info(f"📅 Расписание: пост каждые {POST_INTERVAL} минут")
    logger.info(f"🌙 Ночной перерыв: {QUIET_HOURS_START}:00-{QUIET_HOURS_END}:00")
    if GEMINI_API_KEY:
        logger.info("🧠 AI-суммаризация: ВКЛЮЧЕНА")
        logger.info("🔄 Fallback перевод: ВКЛЮЧЁН (при rate limit)")
    else:
        logger.info("⚠️ AI-суммаризация: ОТКЛЮЧЕНА (добавь GEMINI_API_KEY)")

    while True:
        try:
            # Парсим новости в любое время
            await parse_and_send()
            
            # Публикуем только вне тихих часов
            if not is_quiet_hours():
                await post_from_queue()
            else:
                logger.info(f"🌙 Ночной перерыв, публикация пропущена")
        except Exception as e:
            logger.error(f"Ошибка в цикле: {e}")

        logger.info(f"⏳ Следующий цикл через {PARSER_INTERVAL} минут")
        await asyncio.sleep(PARSER_INTERVAL * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Парсер остановлен пользователем")
