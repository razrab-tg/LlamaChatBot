import os
import re
import subprocess
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.enums import ChatAction
from aiogram.methods import SendChatAction

logging.basicConfig(level=logging.INFO)

BOT_TOKEN = ""

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

user_sessions = {}
processing_flags = {}

def detect_query_type(prompt: str) -> str:
    code_keywords = ["напиши код", "python", "функция", "скрипт", "алгоритм", "программа"]
    return "code" if any(kw in prompt.lower() for kw in code_keywords) else "general"

def run_llama(prompt: str) -> str:
    query_type = detect_query_type(prompt)
    
    if query_type == "code":
        formatted_prompt = (
            "[INST] "
            "Ты — русскоязычный программист. Пиши чистый Python-код без лишних объяснений.\n"
            "Используй тройные кавычки для многострочного кода.\n\n"
            f"Задача: {prompt}\n"
            "Код:[/INST]"
        )
        cmd_params = {
            "n_predict": 1024,
            "temp": 0.5,
            "timeout": 120
        }
    else:
        formatted_prompt = (
            "[INST] "
            "Ты — русскоязычный ассистент. ОТВЕЧАЙ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ. "
            "Используй краткие ответы (1-3 предложения), разговорный стиль.\n\n"
            f"Вопрос: {prompt}\n"
            "Краткий ответ:[/INST]"
        )
        cmd_params = {
            "n_predict": 256,
            "temp": 0.7,
            "timeout": 60
        }

    cmd = [
        "llama",
        "-m", "/home/rich/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        "-p", formatted_prompt,
        "-t", "16",
        "--n-predict", str(cmd_params["n_predict"]),
        "--temp", str(cmd_params["temp"]),
        "--top-p", "0.9",
        "--repeat-penalty", "1.2",
        "--frequency-penalty", "0.1",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=cmd_params["timeout"],
            env={**os.environ, "LLAMA_VERBOSE": "0"}
        )

        logging.info(f"RAW OUTPUT: {result.stdout}")  # Логируем полный вывод

        output = result.stdout
        output = re.sub(r'</?s>|<s>', '', output)
        output = re.sub(r'\[INST\].*?\[/INST\]', '', output, flags=re.DOTALL)
        output = output.replace('[end of text]', '').strip()

        if query_type == "code":
            code_block = re.search(r'```python(.*?)```', output, re.DOTALL)
            response = code_block.group(1).strip() if code_block else output[:1024].strip()
        else:
            response = output.split("Краткий ответ:")[-1].split("Вопрос:")[0].strip()
            response = re.sub(r'\n+', ' ', response).strip()

        return response[:4000] if response else "Ошибка: пустой ответ"

    except subprocess.TimeoutExpired:
        return "Таймаут: модель не ответила вовремя"
    except Exception as e:
        logging.error(f"Ошибка: {str(e)}")
        return f"Критическая ошибка: {str(e)}"

@dp.message(Command("start"))
async def start(message: types.Message):
    await message.answer("Привет! Я отвечаю на русском языке. Задайте любой вопрос.")

@dp.message()
async def handle_message(message: types.Message):
    user_id = message.from_user.id

    if processing_flags.get(user_id, False):
        await message.answer("⏳ Обработка предыдущего запроса...")
        return

    processing_flags[user_id] = True

    try:
        await bot(SendChatAction(chat_id=message.chat.id, action=ChatAction.TYPING))
        response = await asyncio.to_thread(run_llama, message.text)
        response = response.strip()

        if detect_query_type(message.text) == "code":
            await message.answer(f"```python\n{response}\n```", parse_mode="Markdown")
        else:
            await message.answer(response)

    except Exception as e:
        logging.error(f"Ошибка обработки: {e}")
        await message.answer(f"Ошибка: {str(e)}")
    finally:
        processing_flags[user_id] = False

if __name__ == "__main__":
    import asyncio
    asyncio.run(dp.start_polling(bot))
