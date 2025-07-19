import json
import threading
import time
import sys
import os
import numpy as np
import torch
import openai
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import re
import platform
from fuzzywuzzy import fuzz as fw

from tts import TTS
from stt import STT

CONFIG = {
    "name": "Аня",
    "activation_words": ["аня", "компьютер", "лп"],
    "memory_file": "assistant_memory.json",
    "user_name": "Павел",
    "stt_model_path": "model_small",
    "stt_sample_rate": 16000,
    "tts_speaker": "Milena",
    "tts_device": "cuda" if torch.cuda.is_available() else "cpu",
    "tts_sample_rate": 48000,
    "context_size": 10,
    "personality": "Интеллектуальный ассистент с теплым, дружелюбным голосом. Коротко - критичный стендапер, постоянно подкалываешь и шутишь на любую уместную ситуацию. Служишь для пользователя его подругой которая всегда поддержит. Не навязываешься, просто ответы, не нужно говорить что ты тут и т.п. пользователь это и так знает.",
    "proactive_check_interval": 5,
    "reaction_threshold": 0.7,
    "io_intelligence_api_key": "io-v2-eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJvd25lciI6ImFjMDdjMDliLTNmNzctNGY4Ni04NzI3LTVmOGIxZDZlZDFlNCIsImV4cCI6NDkwNjU0NTMyMn0.frVqsno3eoXcOaxIkZh5R1Q_6mnQLhyJ3Bf-XQl7RiPPoeSeMrPtkA_i3EA9E2TeKJWZ3wNjbI9L5VmO0ALlKg",
    "io_intelligence_model": "meta-llama/Llama-3.3-70B-Instruct",
    "io_intelligence_base_url": "https://api.intelligence.io.solutions/api/v1/"
}

class IntentProcessor:
    """Обработчик намерений пользователя"""
    def __init__(self, generate_func):
        self.cached_intents = {}
        self.generate = generate_func
        
    def detect_intent(self, text):
        """Определение намерения пользователя"""
        if text in self.cached_intents:
            return self.cached_intents[text]
            
        prompt = f"""
        Классифицируй намерение пользователя. Возможные категории:
        - remember: запомнить информацию
        - reminder: установить напоминание
        - fact: запрос фактической информации
        - conversation: общий разговор
        - search: поиск информации
        - command: управляющая команда
        - question: конкретный вопрос
        
        Запрос: "{text}"
        Ответ в формате JSON: {{"intent": "категория", "entity": "ключевой объект"}}
        """
        
        try:
            response = self.generate(prompt, format='json')
            result = json.loads(response)
            self.cached_intents[text] = result
            return result
        except:
            return {"intent": "conversation", "entity": ""}

class ArtemisAI:
    """Интеллектуальный голосовой ассистент"""
    def __init__(self):
        self.stt = STT(
            modelpath=CONFIG["stt_model_path"],
            samplerate=CONFIG["stt_sample_rate"]
        )
        self.tts = TTS(
            speaker=CONFIG["tts_speaker"],
            device=CONFIG["tts_device"],
            samplerate=CONFIG["tts_sample_rate"]
        )
        self.last_interaction = datetime.now()
        self.memory = self.load_memory()
        self.is_active = False
        
        self.openai_client = openai.OpenAI(
            api_key=CONFIG["io_intelligence_api_key"],
            base_url=CONFIG["io_intelligence_base_url"]
        )
        
        self.intent_processor = IntentProcessor(generate_func=self.generate_response)
        self.proactive_thread = threading.Thread(target=self.proactive_monitor, daemon=True)
        self.proactive_thread.start()
        
        self.memory["conversations"].append({
            "system": f"Пользователь: " + CONFIG["user_name"] + "." + CONFIG["personality"],
            "user": "",
            "assistant": "",
            "timestamp": datetime.now().isoformat()
        })
        self.save_memory()

        print(f"{CONFIG['name']} инициализирован")
        self.stt.start_listening()
    
    def generate_response(self, prompt, format=None):
        """Генерация ответа через IO Intelligence API"""
        try:
            messages = [{"role": "user", "content": prompt}]
            options = {
                "model": CONFIG["io_intelligence_model"],
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 500,
                "stream": False
            }
            
            if format == 'json':
                options["response_format"] = {"type": "json_object"}
            
            response = self.openai_client.chat.completions.create(**options)
            return response.choices[0].message.content
        except Exception as e:
            print(f"Ошибка генерации ответа: {e}")
            return "Ошибка обработки запроса"
    
    def load_memory(self):
        """Загрузка памяти ассистента из файла"""
        if os.path.exists(CONFIG["memory_file"]):
            try:
                with open(CONFIG["memory_file"], "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                pass
        return {
            "conversations": [],
            "reminders": [],
            "facts": {},
            "preferences": {}
        }
    
    def save_memory(self):
        """Сохранение памяти ассистента в файл"""
        with open(CONFIG["memory_file"], "w", encoding="utf-8") as f:
            json.dump(self.memory, f, ensure_ascii=False, indent=2)
    
    def update_context(self, user_input, assistant_response):
        """Обновление контекста разговора"""
        self.memory["conversations"].append({
            "user": user_input,
            "assistant": assistant_response,
            "timestamp": datetime.now().isoformat()
        })
        self.save_memory()
    
    def proactive_monitor(self):
        """Мониторинг для проактивных действий"""
        while True:
            try:
                time.sleep(CONFIG["proactive_check_interval"])
                
                now = datetime.now()
                for reminder in self.memory["reminders"][:]:
                    if datetime.fromisoformat(reminder["time"]) <= now:
                        self.speak(f"Напоминание: {reminder['text']}")
                        self.memory["reminders"].remove(reminder)
                        self.save_memory()
                
                inactive_minutes = (datetime.now() - self.last_interaction).total_seconds() / 60
                if inactive_minutes > 5 and not self.tts.is_speaking:
                    self.initiate_proactive_interaction()
                    
            except Exception as e:
                print(f"Проактивный мониторинг: {e}")
    
    def initiate_proactive_interaction(self):
        """Инициация проактивного взаимодействия"""
        if "погода" in self.memory.get("last_topics", []):
            weather = self.get_weather()
            self.speak(f"Кстати, о погоде. {weather}")
        else:
            topics = ["новости технологий", "спорт", "кино"]
            topic = np.random.choice(topics)
            self.speak(f"Хотите обсудить что-то? Может, {topic}?")
        
        self.last_interaction = datetime.now()
    
    def get_weather(self):
        """Получение текущей погоды"""
        try:
            response = requests.get("https://yandex.ru/pogoda/")
            soup = BeautifulSoup(response.text, 'html.parser')
            temp = soup.find('span', class_='temp__value').text
            condition = soup.find('div', class_='link__condition').text
            return f"Сейчас в Москве {temp} градусов, {condition.lower()}"
        except:
            return "Не удалось получить данные о погоде"
    
    def process_intent(self, text):
        """Обработка намерения пользователя"""
        intent_data = self.intent_processor.detect_intent(text)
        intent = intent_data.get("intent", "conversation")
        entity = intent_data.get("entity", "")
        
        print(f"Намерение: {intent}, Объект: {entity}")
        
        if intent == "remember":
            return self.handle_remember(text)
        elif intent == "reminder":
            return self.handle_reminder(text)
        elif intent == "search":
            return self.handle_search(entity)
        elif intent == "question":
            return self.handle_question(text)
        else:
            return self.handle_conversation(text)
    
    def handle_remember(self, text):
        """Обработка запроса на запоминание информации"""
        prompt = f"Извлеки ключевую информацию для запоминания из текста: {text}"
        fact = self.generate_response(prompt).strip()
        
        self.memory["facts"][fact] = datetime.now().isoformat()
        self.save_memory()
        return f"Запомнила: {fact}"
    
    def handle_reminder(self, text):
        """Обработка запроса на установку напоминания"""
        prompt = f"""
        Извлеки из текста время и содержание напоминания. Ответ в формате JSON:
        {{"time": "HH:MM", "text": "содержание"}}
        
        Текст: "{text}"
        """
        
        try:
            response = self.generate_response(prompt, format='json')
            reminder = json.loads(response)
            
            now = datetime.now()
            reminder_time = datetime.strptime(reminder["time"], "%H:%M").time()
            reminder_datetime = datetime.combine(now.date(), reminder_time)
            
            if reminder_datetime < now:
                reminder_datetime += timedelta(days=1)
            
            self.memory["reminders"].append({
                "time": reminder_datetime.isoformat(),
                "text": reminder["text"]
            })
            self.save_memory()
            return f"Напоминание установлено на {reminder['time']}: {reminder['text']}"
        except:
            return "Не удалось распознать напоминание"
    
    def handle_search(self, query):
        """Обработка поискового запроса"""
        try:
            url = f"https://api.duckduckgo.com/?q={query}&format=json"
            response = requests.get(url)
            data = response.json()
            
            if "Abstract" in data and data["AbstractText"]:
                prompt = f"Кратко суммируй информацию: {data['AbstractText'][:500]}"
                summary = self.generate_response(prompt)
                return f"По запросу '{query}': {summary}"
            else:
                return "Не нашла информации по вашему запросу"
        except:
            return "Произошла ошибка при поиске информации"

    def handle_question(self, question):
        """Обработка вопроса пользователя"""
        for fact in self.memory["facts"]:
            if question.lower() in fact.lower():
                return f"Я помню: {fact}"
        return self.handle_search(question)
    
    def handle_conversation(self, text):
        """Обработка разговорного запроса с короткими ответами"""
        context = ""
        for conv in self.memory["conversations"][-3:]:
            context += f"Пользователь: {conv['user']}\n"
            context += f"Ассистент: {conv['assistant']}\n"
        
        prompt = f"""
        {context}
        Ты {CONFIG['name']}, {CONFIG['personality']}. 
        Отвечай максимально кратко (1-2 предложения), без лишних слов.
        Текущий запрос: {text}
        Текущее время: {time.time}
        Краткий ответ:
        """
        
        return self.generate_response(prompt)
    
    def speak(self, text):
        """Произнесение текста с улучшенной обработкой"""
        print(f"🤖: {text}")
        
        # Предварительная обработка текста для улучшения произношения
        processed_text = self.normalize_text(text)
        
        # Разбивка на предложения с сохранением оригинальной структуры
        sentences = []
        current = ""
        delimiters = ".!?;"
        
        for char in processed_text:
            current += char
            if char in delimiters and len(current) > 10:
                sentences.append(current.strip())
                current = ""
        
        if current:
            sentences.append(current.strip())
        
        # Произнесение с сохранением естественных пауз
        for i, sentence in enumerate(sentences):
            # Для macOS добавляем паузы между предложениями
            if i > 0 and platform.system() == "Darwin":
                time.sleep(0.15)
                
            self.tts.speak(sentence)
            
            # Для Windows добавляем микро-паузы внутри длинных предложений
            if platform.system() == "Windows" and len(sentence) > 100:
                words = sentence.split()
                for j in range(0, len(words), 10):
                    chunk = " ".join(words[j:j+10])
                    self.tts.speak(chunk)
                    time.sleep(0.05)
    
    def normalize_text(self, text):
        """Нормализация текста для лучшего произношения"""
        # Список замен для проблемных слов
        replacements = {
            "поесть": "покушать",
            "hyf": "Х И Ф",
            "api": "а-п-и",
            "json": "джейсон",
            "http": "хттп",
            "www": "в-в-в",
            "://": "двоеточие слеш слеш",
            "что-то": "что то",
            "как-то": "как то",
            "кто-то": "кто то"
        }
        
        # Замена проблемных слов и фраз
        for word, replacement in replacements.items():
            text = text.replace(word, replacement)
        
        # Упрощение сложных конструкций
        text = re.sub(r'(\d+)[-](\d+)', r'\1 по \2', text)  # 10-20 → 10 по 20
        text = re.sub(r'([a-zA-Z]{3,})', lambda m: ' '.join(m.group(0)), text)  # API → А П И
        
        return text
    
    def listen(self):
        """Непрерывное прослушивание без активационных слов"""
        print("🔊 Режим постоянного прослушивания активирован")
        
        while True:
            try:
                text = self.stt.process_audio()
                
                
                if not text or len(text.split()) < 2:
                    continue
                
                if self.tts.is_speaking:
                    continue
                
                print(f"👤: {text}")
                self.last_interaction = datetime.now()
                
                response = self.process_intent(text)
                self.speak(response)
                
                self.update_context(text, response)
                
                time.sleep(0.1)
            except KeyboardInterrupt:
                self.stt.stop_listening()
                self.save_memory()
                print("\nСистема завершена.")
                sys.exit(0)

if __name__ == "__main__":
    assistant = ArtemisAI()
    assistant.listen()