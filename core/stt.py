import queue
import json
import logging
import threading
import time
import sounddevice as sd
import vosk
import numpy as np
from typing import Optional, Callable, Dict, Any

logger = logging.getLogger("STTService")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class STTConfig:
    """Конфигурация сервиса распознавания речи"""
    def __init__(
        self,
        model_path: str = "model_small",
        sample_rate: int = 16000,
        device_index: Optional[int] = None,
        block_size: int = 8000,
        phrase_timeout: float = 1.5,
        vad_threshold: float = 0.5
    ):
        """
        :param model_path: Путь к модели Vosk
        :param sample_rate: Частота дискретизации аудио
        :param device_index: Индекс аудиоустройства
        :param block_size: Размер блока аудиоданных
        :param phrase_timeout: Таймаут завершения фразы (сек)
        :param vad_threshold: Порог активации голоса (0.0-1.0)
        """
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.device_index = device_index
        self.block_size = block_size
        self.phrase_timeout = phrase_timeout
        self.vad_threshold = vad_threshold


class VADDetector:
    """Детектор голосовой активности"""
    def __init__(self, threshold: float = 0.5, sample_rate: int = 16000):
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.energy_history = []
        self.history_size = int(0.5 * sample_rate / 1000)
        self.is_speech = False
        
    def update(self, audio_frame: np.ndarray) -> bool:
        """Обновление состояния детектора"""
        energy = np.sqrt(np.mean(audio_frame**2))
        self.energy_history.append(energy)
        
        if len(self.energy_history) > self.history_size:
            self.energy_history.pop(0)
            
        if len(self.energy_history) > 10:
            background = np.percentile(self.energy_history, 30)
            threshold = background + self.threshold * (np.max(self.energy_history) - background)
            
            self.is_speech = energy > threshold
            return self.is_speech
        return False


class SpeechToTextService:
    """Сервис распознавания речи с поддержкой VAD и потоковой обработки"""
    def __init__(self, config: STTConfig, text_callback: Callable[[str], None]):
        self.config = config
        self.text_callback = text_callback
        self.is_listening = False
        self.is_processing = False
        
        try:
            self.model = vosk.Model(config.model_path)
            self.recognizer = vosk.KaldiRecognizer(self.model, config.sample_rate)
            logger.info(f"Модель распознавания загружена: {config.model_path}")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            raise RuntimeError("Не удалось инициализировать модель распознавания") from e
        
        self.vad = VADDetector(
            threshold=config.vad_threshold,
            sample_rate=config.sample_rate
        )
        
        self.audio_queue = queue.Queue()
        self.text_queue = queue.Queue()
        
        self.capture_thread = None
        self.process_thread = None
        
    def _audio_callback(self, indata: np.ndarray, frames: int, time: Any, status: Any) -> None:
        """Callback для захвата аудио"""
        if status:
            logger.warning(f"Аудиостатус: {status}")
        
        if self.is_listening:
          
            audio_bytes = indata.tobytes()
            self.audio_queue.put(audio_bytes)
            
          
            self.vad.update(indata.flatten())
    
    def _capture_audio(self) -> None:
        """Поток для захвата аудио"""
        try:
            with sd.RawInputStream(
                samplerate=self.config.sample_rate,
                blocksize=self.config.block_size,
                device=self.config.device_index,
                dtype='int16',
                channels=1,
                callback=self._audio_callback
            ) as stream:
                logger.info("Захват аудио запущен")
                while self.is_listening:
                    sd.sleep(100)
        except Exception as e:
            logger.error(f"Ошибка захвата аудио: {e}")
    
    def _process_audio(self) -> None:
        """Поток для обработки аудио и распознавания речи"""
        logger.info("Обработка аудио запущена")
        current_phrase = []
        last_audio_time = time.time()
        
        while self.is_processing:
            try:
                audio_data = self.audio_queue.get(timeout=0.5)
                
                if not self.vad.is_speech:
                    if current_phrase and (time.time() - last_audio_time) > self.config.phrase_timeout:
                      
                        self._finalize_phrase(current_phrase)
                        current_phrase = []
                    continue
                
                last_audio_time = time.time()
                
                current_phrase.append(audio_data)
                
                if self.recognizer.AcceptWaveform(audio_data):
                    partial_result = json.loads(self.recognizer.PartialResult())
                    text = partial_result.get('partial', '').strip()
                    if text:
                        self.text_queue.put(text)
                
            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"Ошибка обработки аудио: {e}")
        
        if current_phrase:
            self._finalize_phrase(current_phrase)
    
    def _finalize_phrase(self, phrase_chunks: list) -> None:
        """Финализация и распознавание полной фразы"""
        for chunk in phrase_chunks:
            self.recognizer.AcceptWaveform(chunk)
        
        result = json.loads(self.recognizer.Result())
        text = result.get('text', '').strip()
        if text:
            self.text_queue.put(text)
    
    def _dispatch_results(self) -> None:
        """Поток для отправки результатов"""
        logger.info("Диспетчеризация результатов запущена")
        while self.is_processing:
            try:
                text = self.text_queue.get(timeout=1.0)
                if text and self.text_callback:
                    self.text_callback(text)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Ошибка диспетчеризации: {e}")
    
    def start_listening(self) -> None:
        """Запуск службы распознавания речи"""
        if self.is_listening:
            logger.warning("Служба распознавания уже запущена")
            return
        
        logger.info("Запуск службы распознавания речи")
        self.is_listening = True
        self.is_processing = True
        
        self.capture_thread = threading.Thread(
            target=self._capture_audio,
            name="STT-CaptureThread",
            daemon=True
        )
        
        self.process_thread = threading.Thread(
            target=self._process_audio,
            name="STT-ProcessThread",
            daemon=True
        )
        
        self.dispatch_thread = threading.Thread(
            target=self._dispatch_results,
            name="STT-DispatchThread",
            daemon=True
        )
        
        self.capture_thread.start()
        self.process_thread.start()
        self.dispatch_thread.start()
    
    def stop_listening(self) -> None:
        """Остановка службы распознавания речи"""
        if not self.is_listening:
            return
            
        logger.info("Остановка службы распознавания речи")
        self.is_listening = False
        self.is_processing = False
        
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()
        with self.text_queue.mutex:
            self.text_queue.queue.clear()
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        if self.process_thread and self.process_thread.is_alive():
            self.process_thread.join(timeout=1.0)
        if self.dispatch_thread and self.dispatch_thread.is_alive():
            self.dispatch_thread.join(timeout=1.0)
        
        logger.info("Служба распознавания остановлена")


class STTServiceManager:
    """Менеджер для управления службой распознавания речи"""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, config: STTConfig, callback: Callable[[str], None]):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.service = SpeechToTextService(config, callback)
            return cls._instance

    def start(self) -> None:
        """Запуск службы распознавания"""
        self.service.start_listening()
    
    def stop(self) -> None:
        """Остановка службы распознавания"""
        self.service.stop_listening()
    
    def get_service(self) -> SpeechToTextService:
        """Получение экземпляра службы"""
        return self.service
    
    @classmethod
    def get_instance(cls) -> 'STTServiceManager':
        """Получение экземпляра менеджера"""
        if cls._instance is None:
            raise RuntimeError("Менеджер STT не инициализирован")
        return cls._instance
