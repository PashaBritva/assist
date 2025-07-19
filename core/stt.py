import sounddevice as sd
import vosk
import queue
import sys
import json

class STT:
    def __init__(self, modelpath: str = "model_small", samplerate: int = 16000):
        self.model = vosk.Model(modelpath)
        self.recognizer = vosk.KaldiRecognizer(self.model, samplerate)
        self.audio_queue = queue.Queue()
        self.sample_rate = samplerate
        self.is_listening = False

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        if self.is_listening:
            self.audio_queue.put(bytes(indata))

    def start_listening(self):
        self.is_listening = True
        self.stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=8000,
            device=0,
            dtype='int16',
            channels=1,
            callback=self.audio_callback
        )
        self.stream.start()

    def stop_listening(self):
        self.is_listening = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()

    def process_audio(self):
        if self.audio_queue.empty():
            return None
            
        data = self.audio_queue.get()
        if self.recognizer.AcceptWaveform(data):
            result = json.loads(self.recognizer.Result())
            return result.get('text', '')
        return None
