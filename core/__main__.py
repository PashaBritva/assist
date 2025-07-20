import json
import threading
import time
import sys
import os
import numpy as np
import torch
import openai
import yaml
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import re
import logging
import platform

from tts import TTSServiceManager, VoiceProfile, PlaybackState
from stt import STTServiceManager, STTConfig


logger = logging.getLogger("AssistAI")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class AssistantConfig:
    """Класс для управления конфигурацией ассистента"""
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()
        self._configure_logging()
        
    def _load_config(self) -> dict:
        """Загрузка конфигурации из YAML файла"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                logger.info(f"Конфигурация загружена из {self.config_path}")
                return config
        except Exception as e:
            logger.error(f"Ошибка загрузки конфигурации: {e}")
            return self._default_config()
    
    def _default_config(self) -> dict:
        """Конфигурация по умолчанию"""
        return {
            "name": "Аня",
            "activation_words": ["аня", "компьютер", "лп"],
            "memory_file": "assistant_memory.json",
            "user_name": "",
            "tts": {
                "voice_profile": "ASSISTANT",
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "sample_rate": 48000,
                "default_volume": 0.8
            },
            "stt": {
                "model_path": "model_path",
                "sample_rate": 16000,
                "device_index": None,
                "block_size": 8000,
                "phrase_timeout": 1.5,
                "vad_threshold": 0.5
            },
            "context": {
                "size": 10,
                "personality": "Интеллектуальный ассистент с теплым, дружелюбным голосом. Коротко - критичный стендапер, постоянно подкалываешь и шутишь на любую уместную ситуацию. Служишь для пользователя его подругой которая всегда поддержит. Не навязываешься, просто ответы, не нужно говорить что ты тут и т.п. пользователь это и так знает."
            },
            "proactive": {
                "check_interval": 5,
                "reaction_threshold": 0.7
            },
            "intelligence": {
                "api_key": "ai.io.net-api-key",
                "model": "meta-llama/Llama-3.3-70B-Instruct",
                "base_url": "https://api.intelligence.io.solutions/api/v1/"
            }
        }
    
    def _validate_config(self) -> None:
        """Валидация конфигурационных параметров"""
        required_keys = [
            "name", "activation_words", "memory_file", "user_name",
            "stt", "tts", "context", "proactive", "intelligence"
        ]
        
        for key in required_keys:
            if key not in self.config:
                logger.warning(f"Отсутствует ключ конфигурации: {key}")
                self.config[key] = self._default_config().get(key, "")
    
    def _configure_logging(self) -> None:
        """Настройка системы логгирования"""
        log_level = self.config.get("logging", {}).get("level", "INFO")
        logger.setLevel(getattr(logging, log_level.upper()))
    
    def __getattr__(self, name):
        """Доступ к конфигурационным параметрам как к атрибутам"""
        if name in self.config:
            return self.config[name]
        raise AttributeError(f"Конфигурационный параметр '{name}' не найден")
    
    def get_voice_profile(self) -> VoiceProfile:
        """Получение голосового профиля из конфига"""
        try:
            return VoiceProfile[self.config['tts']['voice_profile']]
        except KeyError:
            logger.warning("Неверный голосовой профиль, используется ASSISTANT")
            return VoiceProfile.ASSISTANT
    
    def save_config(self) -> None:
        """Сохранение конфигурации в файл"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, allow_unicode=True)
            logger.info(f"Конфигурация сохранена в {self.config_path}")
        except Exception as e:
            logger.error(f"Ошибка сохранения конфигурации: {e}")


class IntentProcessor:
    """Обработчик намерений пользователя"""
    def __init__(self, generate_func):
        self.cached_intents = {}
        self.generate = generate_func
        
    def detect_intent(self, text: str) -> dict:
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
        except Exception as e:
            logger.error(f"Ошибка определения намерения: {e}")
            return {"intent": "conversation", "entity": ""}


class AssistAICore:
    """Ядро интеллектуального голосового ассистента"""
    def __init__(self, config: AssistantConfig):
        self.config = config
        self._init_stt_service()
        self._init_tts_service()
        
        self.last_interaction = datetime.now()
        self.memory = self.load_memory()
        self.is_active = False
        
        self.openai_client = openai.OpenAI(
            api_key=config.intelligence["api_key"],
            base_url=config.intelligence["base_url"]
        )
        
        self.intent_processor = IntentProcessor(generate_func=self.generate_response)
        self.proactive_thread = threading.Thread(
            target=self.proactive_monitor, 
            daemon=True,
            name="ProactiveMonitor"
        )
        self.proactive_thread.start()
        
        self.memory["conversations"].append({
            "system": f"Пользователь: {config.user_name}. {config.context['personality']}",
            "user": "",
            "assistant": "",
            "timestamp": datetime.now().isoformat()
        })
        self.save_memory()

        logger.info(f"{config.name} инициализирован")
        self.start_listening()
    
    def _init_stt_service(self):
        """Инициализация службы распознавания речи"""
        stt_config = STTConfig(
            model_path=self.config.stt["model_path"],
            sample_rate=self.config.stt["sample_rate"],
            device_index=self.config.stt.get("device_index"),
            block_size=self.config.stt.get("block_size", 8000),
            phrase_timeout=self.config.stt.get("phrase_timeout", 1.5),
            vad_threshold=self.config.stt.get("vad_threshold", 0.5)
        )
        
        self.stt_manager = STTServiceManager(
            config=stt_config,
            callback=self.process_audio_input
        )
        self.stt_service = self.stt_manager.get_service()
    
    def _init_tts_service(self):
        """Инициализация службы синтеза речи"""
        tts_config = {
            "voice_profile": self.config.get_voice_profile(),
            "sample_rate": self.config.tts["sample_rate"],
            "device": self.config.tts["device"],
            "default_volume": self.config.tts["default_volume"]
        }
        self.tts_manager = TTSServiceManager(config=tts_config)
        self.tts_service = self.tts_manager.get_service()
    
    def start_listening(self):
        """Запуск прослушивания"""
        self.stt_manager.start()
        logger.info("Служба распознавания речи запущена")
    
    def stop_listening(self):
        """Остановка прослушивания"""
        self.stt_manager.stop()
        logger.info("Служба распознавания речи остановлена")
    
    def process_audio_input(self, text: str):
        """Обработка распознанного текста"""
        if not text or len(text.split()) < 2:
            return
            
        if self.tts_service.get_playback_state() != PlaybackState.IDLE:
            return
            
        logger.info(f"👤: {text}")
        self.last_interaction = datetime.now()
        
        response = self.process_intent(text)
        self.speak(response)
        
        self.update_context(text, response)
    
    def generate_response(self, prompt: str, format: str = None) -> str:
        """Генерация ответа через IO Intelligence API"""
        try:
            messages = [{"role": "user", "content": prompt}]
            options = {
                "model": self.config.intelligence["model"],
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
            logger.error(f"Ошибка генерации ответа: {e}")
            return "Ошибка обработки запроса"
    
    def load_memory(self) -> dict:
        """Загрузка памяти ассистента из файла"""
        memory_file = self.config.memory_file
        if os.path.exists(memory_file):
            try:
                with open(memory_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Ошибка загрузки памяти: {e}")
        return {
            "conversations": [],
            "reminders": [],
            "facts": {},
            "preferences": {}
        }
    
    def save_memory(self) -> None:
        """Сохранение памяти ассистента в файл"""
        try:
            with open(self.config.memory_file, "w", encoding="utf-8") as f:
                json.dump(self.memory, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения памяти: {e}")
    
    def update_context(self, user_input: str, assistant_response: str) -> None:
        """Обновление контекста разговора"""
        self.memory["conversations"].append({
            "user": user_input,
            "assistant": assistant_response,
            "timestamp": datetime.now().isoformat()
        })
        if len(self.memory["conversations"]) > self.config.context["size"]:
            self.memory["conversations"] = self.memory["conversations"][-self.config.context["size"]:]
        self.save_memory()
    
    def proactive_monitor(self) -> None:
        """Мониторинг для проактивных действий"""
        while True:
            try:
                time.sleep(self.config.proactive["check_interval"])
                
                now = datetime.now()
                for reminder in self.memory["reminders"][:]:
                    if datetime.fromisoformat(reminder["time"]) <= now:
                        self.speak(f"Напоминание: {reminder['text']}")
                        self.memory["reminders"].remove(reminder)
                        self.save_memory()
                
                inactive_minutes = (datetime.now() - self.last_interaction).total_seconds() / 60
                if inactive_minutes > 5 and self.tts_service.get_playback_state() == PlaybackState.IDLE:
                    self.initiate_proactive_interaction()
                    
            except Exception as e:
                logger.error(f"Ошибка проактивного мониторинга: {e}")
    
    def initiate_proactive_interaction(self) -> None:
        """Инициация проактивного взаимодействия"""
        if "погода" in self.memory.get("last_topics", []):
            weather = self.get_weather()
            self.speak(f"Кстати, о погоде. {weather}")
        else:
            topics = ["новости технологий", "спорт", "кино"]
            topic = np.random.choice(topics)
            self.speak(f"Хотите обсудить что-то? Может, {topic}?")
        
        self.last_interaction = datetime.now()
    
    def get_weather(self) -> str:
        """Получение текущей погоды"""
        try:
            response = requests.get("https://yandex.ru/pogoda/", timeout=5)
            soup = BeautifulSoup(response.text, 'html.parser')
            temp = soup.find('span', class_='temp__value').text
            condition = soup.find('div', class_='link__condition').text
            return f"Сейчас в Москве {temp} градусов, {condition.lower()}"
        except Exception as e:
            logger.error(f"Ошибка получения погоды: {e}")
            return "Не удалось получить данные о погоде"
    
    def process_intent(self, text: str) -> str:
        """Обработка намерения пользователя"""
        intent_data = self.intent_processor.detect_intent(text)
        intent = intent_data.get("intent", "conversation")
        entity = intent_data.get("entity", "")
        
        logger.info(f"Намерение: {intent}, Объект: {entity}")
        
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
    
    def handle_remember(self, text: str) -> str:
        """Обработка запроса на запоминание информации"""
        prompt = f"Извлеки ключевую информацию для запоминания из текста: {text}"
        fact = self.generate_response(prompt).strip()
        
        self.memory["facts"][fact] = datetime.now().isoformat()
        self.save_memory()
        return f"Запомнила: {fact}"
    
    def handle_reminder(self, text: str) -> str:
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
        except Exception as e:
            logger.error(f"Ошибка установки напоминания: {e}")
            return "Не удалось распознать напоминание"
    
    def handle_search(self, query: str) -> str:
        """Обработка поискового запроса"""
        try:
            url = f"https://api.duckduckgo.com/?q={query}&format=json"
            response = requests.get(url, timeout=5)
            data = response.json()
            
            if "Abstract" in data and data["AbstractText"]:
                prompt = f"Кратко суммируй информацию: {data['AbstractText'][:500]}"
                summary = self.generate_response(prompt)
                return f"По запросу '{query}': {summary}"
            else:
                return "Не нашла информации по вашему запросу"
        except Exception as e:
            logger.error(f"Ошибка поиска информации: {e}")
            return "Произошла ошибка при поиске информации"

    def handle_question(self, question: str) -> str:
        """Обработка вопроса пользователя"""
        for fact in self.memory["facts"]:
            if question.lower() in fact.lower():
                return f"Я помню: {fact}"
        return self.handle_search(question)
    
    def handle_conversation(self, text: str) -> str:
        """Обработка разговорного запроса с короткими ответами"""
        context = ""
        for conv in self.memory["conversations"][-3:]:
            context += f"Пользователь: {conv['user']}\n"
            context += f"Ассистент: {conv['assistant']}\n"
        
        prompt = f"""
        {context}
        Ты {self.config.name}, {self.config.context['personality']}. 
        Отвечай максимально кратко (1-2 предложения), без лишних слов.
        Текущий запрос: {text}
        Текущее время: {datetime.now().strftime("%H:%M")}
        Краткий ответ:
        """
        
        return self.generate_response(prompt)
    
    def speak(self, text: str) -> None:
        """Произнесение текста с улучшенной обработкой"""
        logger.info(f"🤖: {text}")
        
        processed_text = self.normalize_text(text)
        
        sentences = re.split(r'(?<=[.!?;]) +', processed_text)
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            if len(sentence) > 100:
                words = sentence.split()
                chunks = [' '.join(words[i:i+10]) for i in range(0, len(words), 10)]
                for chunk in chunks:
                    self.tts_service.enqueue_speech(chunk)
                    time.sleep(0.05)
            else:
                self.tts_service.enqueue_speech(sentence)
    
    def normalize_text(self, text: str) -> str:
        """Нормализация текста для лучшего произношения"""
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
        
        for word, replacement in replacements.items():
            text = text.replace(word, replacement)
        
        text = re.sub(r'(\d+)[-](\d+)', r'\1 по \2', text)
        text = re.sub(r'([a-zA-Z]{3,})', lambda m: ' '.join(m.group(0)), text)
        
        return text
    
    def shutdown(self) -> None:
        """Корректное завершение работы ассистента"""
        logger.info("Завершение работы ассистента...")
        self.stop_listening()
        self.save_memory()
        self.tts_service.shutdown()
        logger.info("Ассистент успешно остановлен")


class AssistAIManager:
    """Менеджер для управления ассистентом"""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, config_path: str = "config.yaml"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.config = AssistantConfig(config_path)
                cls._instance.assistant = AssistAICore(cls._instance.config)
            return cls._instance

    def start(self) -> None:
        """Запуск ассистента"""
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
        except Exception as e:
            logger.error(f"Ошибка в основном цикле: {e}")
            self.stop()
    
    def stop(self) -> None:
        """Остановка ассистента"""
        self.assistant.shutdown()
    
    def get_assistant(self) -> AssistAICore:
        """Получение экземпляра ассистента"""
        return self.assistant
    
    @classmethod
    def get_instance(cls) -> 'AssistAIManager':
        """Получение экземпляра менеджера"""
        if cls._instance is None:
            cls()
        return cls._instance


if __name__ == "__main__":
    manager = AssistAIManager(config_path="config.yaml")
    try:
        manager.start()
    except KeyboardInterrupt:
        manager.stop()
        sys.exit(0)