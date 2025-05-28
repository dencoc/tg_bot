[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/fastapi-%5E0.110-green.svg)](https://fastapi.tiangolo.com/)
[![LanceDB](https://img.shields.io/badge/lancedb-%5E0.4.12-orange.svg)](https://lancedb.github.io/lancedb/)
[![Telegram Bot API](https://img.shields.io/badge/telegram-bot-lightblue.svg)](https://core.telegram.org/bots/api)

# CryptoSense Bot Backend

Серверная часть Telegram-бота с поддержкой RAG (Retrieval-Augmented Generation), предназначенного для ответов на вопросы в тематике криптовалют. Использует векторную базу LanceDB, генерацию через Ollama и поиск через Tavily API.

---

## 🚀 Основные технологии

- **Язык**: Python 3.11+
- **Фреймворк API**: FastAPI
- **Асинхронный Telegram-бот**: aiogram
- **Генерация текста**: Ollama API (LLM)
- **Поиск по Интернету**: Tavily API
- **Локальный векторный поиск**: LanceDB
- **Работа с эмбеддингами**: OpenAI или BGE модели
- **Логирование**: loguru

---

## ⚙️ Установка и запуск

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
