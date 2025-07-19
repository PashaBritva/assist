import platform
import queue
import threading
import time
import logging
import torch
import pyaudio
import numpy as np
from enum import Enum
from silero import silero_tts
from typing import Optional, Tuple, Dict, Any, Callable

logger = logging.getLogger("TTSEnterprise")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class VoiceProfile(Enum):
    """Профили голосов с предустановленными параметрами"""
    STANDARD_FEMALE = ("baya", 1.7, 0.95)
    STANDARD_MALE = ("aidar", 1.6, 1.0)
    NEWS_ANNOUNCER = ("xenia", 1.5, 0.9)
    GAME_CHARACTER = ("ragnaros", 1.8, 0.8)
    ASSISTANT = ("anna", 1.9, 1.0)

    def __init__(self, voice_id: str, speed: float, pitch: float):
        self.voice_id = voice_id
        self.speed = speed
        self.pitch = pitch


class PlaybackState(Enum):
    """Состояния воспроизведения"""
    IDLE = 0
    PLAYING = 1
    PAUSED = 2
    STOPPED = 3


class TTSEngineConfig:
    """Конфигурация движка синтеза речи"""
    def __init__(
        self,
        voice_profile: VoiceProfile = VoiceProfile.ASSISTANT,
        sample_rate: int = 48000,
        device: Optional[str] = None,
        max_queue_size: int = 15,
        min_speech_interval: float = 1.0,
        default_volume: float = 0.8
    ):
        """
        :param voice_profile: Профиль голоса
        :param sample_rate: Частота дискретизации
        :param device: Устройство обработки (cuda/cpu)
        :param max_queue_size: Макс размер очереди
        :param min_speech_interval: Мин интервал между фразами
        :param default_volume: Громкость по умолчанию (0.0-1.0)
        """
        self.voice_profile = voice_profile
        self.sample_rate = sample_rate
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_queue_size = max_queue_size
        self.min_speech_interval = min_speech_interval
        self.default_volume = default_volume


class SpeechSynthesizer:
    """Ядро синтеза речи с использованием Silero TTS"""
    def __init__(self, config: TTSEngineConfig):
        self.config = config
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Инициализация нейросетевой модели TTS"""
        try:
            self.model, self.symbols, self.sample_rate, self.speakers, self.apply_tts = silero_tts(
                model_id='v3_1_ru',
                language='ru',
                speaker=self.config.voice_profile.voice_id,
                device=self.config.device,
                stress_flags=True
            )
            logger.info(f"TTS модель инициализирована: {self.config.voice_profile.name}, "
                       f"устройство: {self.config.device}")
        except Exception as e:
            logger.error(f"Ошибка инициализации модели: {e}")
            raise RuntimeError("Не удалось загрузить модель TTS") from e

    def synthesize(self, text: str) -> Optional[bytes]:
        """Синтез аудио из текста"""
        try:
            audio = self.apply_tts(
                text=text,
                model=self.model,
                symbols=self.symbols,
                sample_rate=self.sample_rate,
                speaker=self.config.voice_profile.voice_id,
                stress_flags=True,
                speed=self.config.voice_profile.speed,
                pitch=self.config.voice_profile.pitch,
                pauses_duration=0.15
            )
            return audio.cpu().numpy().tobytes()
        except Exception as e:
            logger.error(f"Ошибка синтеза речи: {e}")
            return None


class AudioPlayer:
    """Управление воспроизведением аудио с поддержкой паузы и громкости"""
    def __init__(self, sample_rate: int, default_volume: float = 0.8):
        self.sample_rate = sample_rate
        self.volume = default_volume
        self.state = PlaybackState.IDLE
        self.pyaudio = pyaudio.PyAudio()
        self.stream = None
        self.playback_thread = None
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.lock = threading.RLock()
        self.current_audio = None
        self.playback_position = 0

    def _apply_volume(self, audio_data: bytes) -> bytes:
        """Применение уровня громкости к аудиоданным"""
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            adjusted_audio = (audio_array * self.volume).astype(np.int16)
            return adjusted_audio.tobytes()
        except Exception as e:
            logger.error(f"Ошибка применения громкости: {e}")
            return audio_data

    def _playback_worker(self, audio_data: bytes) -> None:
        """Поток для управления воспроизведением с поддержкой паузы"""
        try:
            with self.lock:
                self.state = PlaybackState.PLAYING
                self.current_audio = audio_data
                self.playback_position = 0
                
            chunk_size = 1024
            audio_length = len(audio_data)
            
            while self.playback_position < audio_length and not self.stop_event.is_set():
                if self.pause_event.is_set():
                    with self.lock:
                        self.state = PlaybackState.PAUSED
                    self.pause_event.wait()
                    with self.lock:
                        if self.state == PlaybackState.PAUSED:
                            self.state = PlaybackState.PLAYING
                
                with self.lock:
                    chunk = audio_data[self.playback_position:self.playback_position + chunk_size]
                    self.playback_position += chunk_size
                
                if chunk:
                    self.stream.write(chunk)
                
                time.sleep(0.01)
            
            with self.lock:
                if self.stop_event.is_set():
                    self.state = PlaybackState.STOPPED
                else:
                    self.state = PlaybackState.IDLE
                self.current_audio = None
                self.playback_position = 0
                
        except Exception as e:
            logger.error(f"Ошибка в потоке воспроизведения: {e}")
            with self.lock:
                self.state = PlaybackState.IDLE
        finally:
            self.stop_event.clear()

    def play(self, audio_data: bytes) -> bool:
        """Начать воспроизведение аудио"""
        if self.state in [PlaybackState.PLAYING, PlaybackState.PAUSED]:
            self.stop()
            
        try:
            audio_data = self._apply_volume(audio_data)
            
            self.stream = self.pyaudio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=1024
            )
            
            self.playback_thread = threading.Thread(
                target=self._playback_worker, 
                args=(audio_data,),
                daemon=True
            )
            self.playback_thread.start()
            return True
        except Exception as e:
            logger.error(f"Ошибка запуска воспроизведения: {e}")
            self.state = PlaybackState.IDLE
            return False

    def pause(self) -> bool:
        """Приостановить воспроизведение"""
        if self.state == PlaybackState.PLAYING:
            self.pause_event.set()
            with self.lock:
                self.state = PlaybackState.PAUSED
            return True
        return False

    def resume(self) -> bool:
        """Возобновить воспроизведение"""
        if self.state == PlaybackState.PAUSED:
            self.pause_event.clear()
            with self.lock:
                self.state = PlaybackState.PLAYING
            return True
        return False

    def stop(self) -> bool:
        """Остановить воспроизведение"""
        if self.state in [PlaybackState.PLAYING, PlaybackState.PAUSED]:
            self.stop_event.set()
            self.pause_event.clear()
            
            if self.playback_thread and self.playback_thread.is_alive():
                self.playback_thread.join(timeout=0.5)
            
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
                
            with self.lock:
                self.state = PlaybackState.STOPPED
            return True
        return False

    def set_volume(self, volume: float) -> None:
        """Установка уровня громкости (0.0 - 1.0)"""
        with self.lock:
            self.volume = max(0.0, min(1.0, volume))
            if self.current_audio and self.state in [PlaybackState.PLAYING, PlaybackState.PAUSED]:
                current_pos = self.playback_position
                self.stop()
                self.play(self.current_audio[current_pos:])

    def get_state(self) -> PlaybackState:
        """Текущее состояние воспроизведения"""
        return self.state

    def release_resources(self) -> None:
        """Освобождение аудиоресурсов"""
        self.stop()
        if self.pyaudio:
            self.pyaudio.terminate()


class TextToSpeechService:
    """Сервис преобразования текста в речь с расширенным управлением"""
    def __init__(self, config: TTSEngineConfig):
        self.config = config
        self.playback_state = PlaybackState.IDLE
        self.last_speech_time = 0
        self.queue = queue.Queue(maxsize=config.max_queue_size)
        self.audio_queue = queue.Queue()
        
        self.synthesizer = SpeechSynthesizer(config)
        self.audio_player = AudioPlayer(
            sample_rate=config.sample_rate,
            default_volume=config.default_volume
        )
        
        self.synthesis_thread = threading.Thread(
            target=self._synthesis_worker, 
            daemon=True,
            name="TTS-SynthesisWorker"
        )
        self.playback_thread = threading.Thread(
            target=self._playback_worker, 
            daemon=True,
            name="TTS-PlaybackWorker"
        )
        
        self.synthesis_thread.start()
        self.playback_thread.start()
        
        logger.info("TTS сервис запущен")

    def _synthesis_worker(self) -> None:
        """Поток для синтеза текста в аудио"""
        while True:
            try:
                text = self.queue.get()
                if not text:
                    continue
                
                short_text = self._optimize_text(text)
                logger.debug(f"Синтез текста: '{short_text}'")
                
                audio_data = self.synthesizer.synthesize(short_text)
                if audio_data:
                    self.audio_queue.put(audio_data)
                
                self.queue.task_done()
            except Exception as e:
                logger.exception(f"Ошибка в потоке синтеза: {e}")

    def _playback_worker(self) -> None:
        """Поток для воспроизведения синтезированного аудио"""
        while True:
            try:
                audio_data = self.audio_queue.get()
                if not audio_data:
                    continue
                
                self.last_speech_time = time.time()
                self.playback_state = PlaybackState.PLAYING
                self.audio_player.play(audio_data)
                
                while self.audio_player.get_state() in [PlaybackState.PLAYING, PlaybackState.PAUSED]:
                    time.sleep(0.1)
                
                self.audio_queue.task_done()
                self.playback_state = PlaybackState.IDLE
            except Exception as e:
                logger.exception(f"Ошибка в потоке воспроизведения: {e}")

    def _optimize_text(self, text: str) -> str:
        """Оптимизация текста для TTS"""
        optimization_rules = {
            "пожалуйста": "", "спасибо": "", "извините": "",
            "я думаю": "", "возможно": "", "наверное": "",
            "артемис здесь": "", "слушаю вас": "", "чем могу помочь": "",
            "похоже, что": "", "я считаю, что": "", "мне кажется, ": "",
            "должен сказать, что": "", "хотел бы отметить, что": ""
        }
        
        optimized_text = text
        for pattern, replacement in optimization_rules.items():
            optimized_text = optimized_text.replace(pattern, replacement)
        
        optimized_text = " ".join(optimized_text.split())
        
        words = optimized_text.split()
        if len(words) > 20:
            optimized_text = " ".join(words[:15]) + "..."
            logger.info(f"Текст сокращен: {text[:50]}... -> {optimized_text}")
        
        return optimized_text

    def enqueue_speech(self, text: str, priority: bool = False) -> bool:
        """
        Добавление текста в очередь на озвучивание
        
        :param text: Текст для озвучивания
        :param priority: Приоритетное добавление в начало очереди
        :return: Успешность добавления в очередь
        """
        if not text.strip():
            return False
            
        current_time = time.time()
        if (current_time - self.last_speech_time < self.config.min_speech_interval or
            self.queue.qsize() >= self.config.max_queue_size):
            logger.warning("Запрос отклонён: превышен лимит очереди или временной интервал")
            return False
            
        try:
            if priority:
                temp_queue = queue.Queue()
                temp_queue.put(text)
                while not self.queue.empty():
                    temp_queue.put(self.queue.get())
                self.queue = temp_queue
            else:
                self.queue.put(text)
                
            logger.debug(f"Добавлено в очередь: '{text[:30]}...'")
            return True
        except queue.Full:
            logger.warning("Очередь TTS переполнена, запрос отклонён")
            return False

    def pause_playback(self) -> bool:
        """Приостановка текущего воспроизведения"""
        if self.playback_state == PlaybackState.PLAYING:
            return self.audio_player.pause()
        return False

    def resume_playback(self) -> bool:
        """Возобновление воспроизведения"""
        if self.playback_state == PlaybackState.PAUSED:
            return self.audio_player.resume()
        return False

    def stop_playback(self) -> bool:
        """Остановка текущего воспроизведения"""
        if self.playback_state in [PlaybackState.PLAYING, PlaybackState.PAUSED]:
            return self.audio_player.stop()
        return False

    def set_volume(self, volume: float) -> None:
        """Установка уровня громкости (0.0 - 1.0)"""
        self.audio_player.set_volume(volume)
        logger.info(f"Установлена громкость: {volume*100:.0f}%")

    def clear_queue(self) -> None:
        """Очистка всех очередей"""
        with self.queue.mutex:
            self.queue.queue.clear()
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()
        self.stop_playback()
        logger.info("Очереди TTS очищены")

    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """
        Ожидание завершения всех задач в очереди
        
        :param timeout: Максимальное время ожидания
        :return: Все ли задачи выполнены
        """
        try:
            start_time = time.time()
            while not (self.queue.empty() and self.audio_queue.empty() 
                      and self.playback_state == PlaybackState.IDLE):
                if timeout and (time.time() - start_time) > timeout:
                    return False
                time.sleep(0.1)
            return True
        except Exception as e:
            logger.error(f"Ошибка ожидания завершения: {e}")
            return False

    def get_playback_state(self) -> PlaybackState:
        """Текущее состояние воспроизведения"""
        return self.playback_state

    def get_queue_size(self) -> Tuple[int, int]:
        """Размеры очередей (синтез, воспроизведение)"""
        return self.queue.qsize(), self.audio_queue.qsize()

    def shutdown(self) -> None:
        """Корректное завершение работы сервиса"""
        self.clear_queue()
        self.wait_for_completion(timeout=2.0)
        self.audio_player.release_resources()
        logger.info("TTS сервис остановлен")


class TTSServiceManager:
    """Менеджер для управления TTS сервисом"""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, config: Optional[TTSEngineConfig] = None):
        if self._initialized:
            return
            
        self.config = config or TTSEngineConfig()
        self.service = TextToSpeechService(self.config)
        self._initialized = True
        logger.info("Менеджер TTS инициализирован")

    def get_service(self) -> TextToSpeechService:
        return self.service

    @classmethod
    def get_instance(cls) -> 'TTSServiceManager':
        if cls._instance is None:
            cls()
        return cls._instance


if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor
    
    tts_manager = TTSServiceManager()
    tts_service = tts_manager.get_service()
    
    def async_speak(text: str, priority: bool = False):
        tts_service.enqueue_speech(text, priority)
    
    executor = ThreadPoolExecutor(max_workers=4)
    
    try:
        executor.submit(async_speak, "Начинаем тестовое проигрывание", True)
        time.sleep(1)
        
        executor.submit(async_speak, "Это обычное сообщение с низким приоритетом")
        executor.submit(async_speak, "Срочное сообщение", True)
        
        executor.submit(tts_service.pause_playback)
        time.sleep(0.5)
        executor.submit(tts_service.resume_playback)
        
        executor.submit(tts_service.set_volume, 0.6)
        
        executor.submit(tts_service.wait_for_completion)
        
        executor.submit(async_speak, "Тестирование завершено успешно")
        
    finally:
        executor.shutdown(wait=True)
        tts_service.shutdown()