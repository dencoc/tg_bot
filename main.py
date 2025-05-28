import os
import logging
import requests
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import LanceDB as LangchainLanceDB
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import lancedb
from datetime import datetime, timezone
from hashlib import md5
import json
import pyarrow as pa

load_dotenv()

# ========== Константы ==========
TELEGRAM_API_KEY = os.getenv("TELEGRAM_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:latest")
MAX_DB_SIZE = 10000

# ========== Логирование ==========
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ========== Контексты ==========
user_contexts: dict[int, list[dict]] = {}

# ========== Подключение к LanceDB ==========
def get_lance_db(
    db_path: str = "lancedb",
    table_name: str = "pdf_docs",
    dim: int = 384,
    model_name: str = "all-MiniLM-L6-v2"
) -> LangchainLanceDB:

    embedding_fn = HuggingFaceEmbeddings(model_name=model_name)

    connection = lancedb.connect(db_path)

    try:
        table = connection.open_table(table_name)
    except Exception:
        schema = pa.schema([
            ("vector", lancedb.vector(dim)),
            ("text", pa.string()),
            ("id", pa.string()),
            ("source", pa.string()),
            ("title", pa.string()),
            ("query", pa.string()),
            ("timestamp", pa.string()),
            ("user_id", pa.string()),
        ])
        connection.create_table(table_name, schema=schema)
        table = connection.open_table(table_name)

        test_emb = embedding_fn.embed_query("инициализация LanceDB")
        table.add([{
            "vector": test_emb,
            "text": "Тестовый документ",
            "id": "init_doc",
            "source": "system",
            "title": "init",
            "query": "инициализация",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id": "system"
        }])

    vectorstore = LangchainLanceDB(
        connection=connection,
        embedding=embedding_fn,
        table_name=table_name
    )

    vectorstore.table = table
    return vectorstore

lance_db = get_lance_db()
embedding_fn = lance_db._embedding

# ========== Функции поиска и генерации ==========
def search_tavily(query: str, max_results: int = 3) -> list[dict]:
    url = "https://api.tavily.com/search"
    headers = {
        "Authorization": f"Bearer {TAVILY_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "query": query,
        "search_depth": "advanced",
        "max_results": max_results
    }
    resp = requests.post(url, json=payload, headers=headers)
    resp.raise_for_status()
    return resp.json().get("results", [])


def search_lance_db(query: str, k: int = 3) -> str:
    docs = lance_db.similarity_search(query, k=k)
    if not docs:
        return "❗ Нет релевантных документов в локальной базе."
    
    results = []
    for i, doc in enumerate(docs, 1):
        content = doc.page_content
        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
        source = metadata.get('source', 'неизвестный источник')
        results.append(f"🔹 Документ {i} ({source}):\n{content[:500]}{'...' if len(content) > 500 else ''}")
    
    return "\n\n".join(results)

def generate_with_ollama(messages: list[dict]) -> str:
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False
    }
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    return resp.json()["message"]["content"]

# ========== Агент ==========
def decide_strategy(query: str) -> dict:
    planning_prompt = [
        {"role": "system", "content": (
            "Ты агент, помогающий решить, где искать информацию. "
            "У тебя есть два источника: интернет и локальная база (документы). "
            "Ответь в формате JSON: {\"use_internet\": bool, \"use_local\": bool, \"reasoning\": \"...\"}. "
            "Если нужно искать в обоих — укажи оба флага как true. "
            "Отвечай только JSON, без пояснений."
        )},
        {"role": "user", "content": f"Запрос: {query}"}
    ]

    try:
        response = generate_with_ollama(planning_prompt)
        result = json.loads(response.strip())
        return {
            "internet": result.get("use_internet", True),
            "local": result.get("use_local", True)
        }
    except Exception as e:
        logger.error("Ошибка стратегии от Ollama: %s", e)
        return {"internet": True, "local": True}

# ========== Хендлеры ==========
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_contexts[user_id] = []
    await update.message.reply_text(
        "Привет! Я RAG-бот.\n"
        "Задай мне вопрос, и я найду информацию в интернете и/или в документах.\n"
        "Все найденные результаты автоматически сохраняются в моей базе знаний."
    )

async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    query = update.message.text

    history = user_contexts.get(user_id, [])
    user_contexts[user_id] = history

    strategy = decide_strategy(query)
    docs_combined = ""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    if strategy["internet"]:
        try:
            internet_results = search_tavily(query)
            logger.info("Tavily returned %d results for query=%s", len(internet_results), query)

            if internet_results:
                internet_output = "\n\n".join(
                    f"🔹 {r['title']}\n{r['url']}" for r in internet_results
                )
                docs_combined += f"[Интернет результаты]\n{internet_output}\n\n"

                documents = []
                existing_ids = lance_db.table.to_pandas()["id"].tolist()
                for res in internet_results:
                    content = f"{res['title']}\n{res['url']}\n{res.get('content','')}"
                    for chunk in text_splitter.split_text(content):
                        chunk_id = md5(chunk.encode()).hexdigest()

                        if chunk_id in existing_ids:
                            logger.debug("Chunk %s already exists, skipping", chunk_id)
                            continue

                        metadata = {
                            "source": res['url'],
                            "title": res['title'],
                            "query": query,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "user_id": str(user_id)
                        }
                        documents.append((chunk, metadata, chunk_id))
                        existing_ids.append(chunk_id)

                if documents:
                    texts, metadatas, ids = zip(*documents)
                    embeddings = [embedding_fn.embed_query(t) for t in texts]

                    rows = []
                    for emb, text, meta, id_ in zip(embeddings, texts, metadatas, ids):
                        rows.append({
                            "vector": emb,
                            "text": text,
                            "id": id_,
                            "source": meta["source"],
                            "title": meta["title"],
                            "query": meta["query"],
                            "timestamp": meta["timestamp"],
                            "user_id": meta["user_id"],
                        })

                    try:
                        lance_db.table.add(rows)
                        logger.info("✔ Added %d new chunks to LanceDB", len(rows))
                    except Exception as e:
                        logger.error("❌ Не удалось добавить строки напрямую: %s", e, exc_info=True)
        except Exception as e:
            logger.error("Tavily error: %s", e)
    if strategy["local"]:
        try:
            local_docs = search_lance_db(query)
            docs_combined += f"[Локальная база]\n{local_docs}"
        except Exception as e:
            logger.error("LanceDB error: %s", e)
            await update.message.reply_text("❗ Ошибка поиска в локальной базе.")

    messages = [{"role": "system", "content": "Отвечай на русском языке."}] + history
    messages.append({
        "role": "user",
        "content": f"Найдено:\n{docs_combined}\nВопрос: {query}"
    })

    try:
        answer = generate_with_ollama(messages)
    except Exception as e:
        logger.error("Ollama error: %s", e)
        await update.message.reply_text("❗ Ошибка генерации ответа.")
        return

    history.extend([{"role": "user", "content": query},
                    {"role": "assistant", "content": answer}])
    user_contexts[user_id] = history[-10:]
    await update.message.reply_text(answer)

# ========== Main ==========
def main():
    app = ApplicationBuilder().token(TELEGRAM_API_KEY).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()

if __name__ == "__main__":
    main()