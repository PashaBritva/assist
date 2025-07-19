import platform
import queue
import threading
import pyttsx3
import subprocess
import time

class TTS:
    """Кроссплатформенный синтезатор речи с короткими ответами и непрерывным прослушиванием"""
    def __init__(self, speaker="Yuri", device="cpu", samplerate=48000):
        self.system = platform.system()
        self.speaker = speaker
        self.samplerate = samplerate
        self.is_speaking = False
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._process_queue, daemon=True)
        self.thread.start()
        self.last_speech_time = 0
        
        if self.system == "Windows":
            self.engine = pyttsx3.init()
            self._configure_windows_voice()
    
    def _configure_windows_voice(self):
        """Настройка русского голоса для Windows"""
        voices = self.engine.getProperty('voices')
        windows_voices = {
            "Irina": "Microsoft Irina Desktop",
            "Pavel": "Microsoft Pavel Mobile",
            "Yuri": "Microsoft Yuri Mobile"
        }
        
        for voice in voices:
            if self.speaker in voice.name and "russian" in voice.languages[0].decode():
                self.engine.setProperty('voice', voice.id)
                return
            elif windows_voices.get(self.speaker) == voice.name:
                self.engine.setProperty('voice', voice.id)
                return
        
        for voice in voices:
            if "russian" in voice.languages[0].decode():
                self.engine.setProperty('voice', voice.id)
                self.speaker = voice.name
                return
        
        self.engine.setProperty('voice', voices[0].id)
        self.speaker = voices[0].name
        self.engine.setProperty('rate', 300)  # Более быстрая скорость по умолчанию
        self.engine.setProperty('volume', 0.9)
    
    def _speak_macos(self, text):
        """Произнесение текста на macOS с ускоренной речью"""
        try:
            self.is_speaking = True
            self.last_speech_time = time.time()
            voice_mapping = {
                "Milena": "Milena",
                "Yuri": "Yuri",
                "Anna": "Anna",
                "Samantha": "Samantha"
            }
            voice = voice_mapping.get(self.speaker, self.speaker)
            
            # Ускоренная речь для macOS
            cmd = ['say', '-v', voice, '-r', '250', text]  # Высокая скорость (400 слов/мин)
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            process.communicate()
        except Exception as e:
            print(f"Ошибка синтеза речи на macOS: {e}")
        finally:
            self.is_speaking = False
    
    def _speak_windows(self, text):
        """Произнесение текста на Windows с ускоренной речью"""
        try:
            self.is_speaking = True
            self.last_speech_time = time.time()
            self.engine.setProperty('rate', 200)  # Ускоренная речь
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"Ошибка синтеза речи на Windows: {e}")
        finally:
            self.is_speaking = False
    
    def _process_queue(self):
        """Обработка очереди сообщений с короткими ответами"""
        while True:
            text = self.queue.get()
            
            # Сокращение ответов
            short_text = self._shorten_response(text)
            
            if self.system == "Darwin":
                self._speak_macos(short_text)
            elif self.system == "Windows":
                self._speak_windows(short_text)
            else:
                print(f"TTS: {short_text}")
                time.sleep(len(short_text) * 0.03)  # Более короткие паузы
            
            self.queue.task_done()
    
    def _shorten_response(self, text):
        """Сокращение ответов для большей краткости"""
        # Удаление приветствий и излишних фраз
        removals = [
            "пожалуйста", "спасибо", "извините", 
            "я думаю", "возможно", "наверное",
            "артемис здесь", "слушаю вас", "чем могу помочь"
        ]
        
        # Сокращение длинных фраз
        replacements = {
            "похоже, что": "",
            "я считаю, что": "",
            "мне кажется, ": "",
            "должен сказать, что": "",
            "хотел бы отметить, что": ""
        }
        
        # Применяем сокращения
        short_text = text
        for phrase in removals:
            short_text = short_text.replace(phrase, "")
        
        for pattern, replacement in replacements.items():
            short_text = short_text.replace(pattern, replacement)
        
        # Удаление двойных пробелов
        short_text = " ".join(short_text.split())
        
        # Ограничение длины
        if len(short_text.split()) > 10:
            short_text = " ".join(short_text.split()[:10]) + "."
        
        return short_text
    
    def speak(self, text):
        """Добавление текста в очередь на произнесение с короткими ответами"""
        if text.strip():
            # Пропускаем ответ, если не прошло 2 секунды с последней речи
            if time.time() - self.last_speech_time < 2.0:
                return
                
            self.queue.put(text)
    
    def wait(self):
        """Ожидание завершения всех задач"""
        self.queue.join()