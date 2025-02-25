# 🤖 LlamaChatBot — Telegram-бот с ИИ-ассистентом
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

Telegram-бот, использующий модель **Mistral-7B** для:
- Генерации Python-кода
- Кратких ответов на вопросы
- Помощи в решении задач

![image](https://github.com/user-attachments/assets/a20309fd-60a6-4f99-a048-d754890b2867)
 <!-- Добавьте реальный скриншот -->

## 🚀 Особенности
✅ **Двухрежимный ответ**:  
- Код в формате Markdown  
- Краткие текстовые ответы на русском  

✅ **Адаптивность**:  
- Распознаёт запросы автоматически  
- Оптимизирован для скорости (16 потоков)  

✅ **Безопасность**:  
- Защита от двойных запросов  
- Логирование всех операций  

## 📋 Требования
- LLAMA собранная под ваш процессор (в нашем случае Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz)
- Python 3.10+
- Модель [Mistral-7B GGUF](https://huggingface.co/models)
- Библиотека: `aiogram`
- 8+ GB RAM 

## 🛠 Установка
1. Клонируйте репозиторий:
```bash
git clone https://github.com/yourname/LlamaChatBot.git
cd LlamaChatBot
```

2. Установите зависимости:
```bash
pip install aiogram
```

3. Загрузите модель:
```bash
mkdir models
wget -O models/mistral-7b.gguf https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf?download=true
```

4. Настройте токен в `bot.py`:
```python
bot = Bot(token="ВАШ_ТОКЕН_ЗДЕСЬ")
```

## 🎮 Использование
1. Запустите бота:
```bash
python bot.py
```

2. В Telegram:
- `/start` — приветствие
- "Напиши код на python для факториала" → получите Python-скрипт
- "Как сварить кофе?" → краткая инструкция

## 📚 Пример
**Запрос:**  
`напиши код на python скрипт для парсинга CSV и вывода первых 5 строк"`

**Ответ:**
```python
import csv
with open('file.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader[:5]:
        print(row)
```
## Средняя температура при тесте 500 промптов
![image](https://github.com/user-attachments/assets/08f70ef8-62a8-459e-b2d3-80ef8ccabbf7)
## Доп.информация
![image](https://github.com/user-attachments/assets/3e05ab57-655e-4d97-8906-48fc9c69df7e)
