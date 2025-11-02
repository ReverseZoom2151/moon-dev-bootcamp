#!/usr/bin/env python3
"""
🌙 MOON DEV's Enhanced Sleek Yap AI - Professional Real-time Chat! 🌙
Enhanced real-time AI voice chat with OpenAI Realtime API
Advanced features: Voice interruption, echo suppression, professional architecture
Built with modern Python practices by MOON DEV 🚀

Enhanced Features:
- Professional logging system with structured output
- Type hints and comprehensive documentation  
- Configuration management with dataclasses
- Enhanced error handling and recovery
- Resource management and proper cleanup
- Improved audio processing and VAD
- Better WebSocket connection management
- Modular architecture for maintainability
"""

import os
import json
import threading
import time
import base64
import logging
import sounddevice as sd
import numpy as np
import pygame
import websocket
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime
from pathlib import Path
from enum import Enum
from pynput import keyboard
from dataclasses import dataclass, field
from dotenv import load_dotenv


# Enhanced imports
try:
    from termcolor import colored, cprint
    TERMCOLOR_AVAILABLE = True
except ImportError:
    TERMCOLOR_AVAILABLE = False
    def colored(text: str, color: Optional[str] = None) -> str:
        return text
    def cprint(text: str, color: Optional[str] = None, attrs: Optional[List[str]] = None) -> None:
        print(text)


class ChatState(Enum):
    """Chat system states"""
    WAITING = "waiting"
    CONNECTING = "connecting" 
    ACTIVE = "active"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class AudioConfig:
    """Enhanced audio configuration"""
    input_device_id: int = 1
    output_device_id: int = 3
    sample_rate: int = 24000
    channels: int = 1
    frames_per_buffer: int = 1024
    volume_boost: float = 0.6


@dataclass
class VoiceActivityConfig:
    """Voice Activity Detection configuration"""
    threshold_quiet: float = 0.06
    threshold_during_ai: float = 0.4
    hold_duration_ms: int = 180
    cooldown_duration_ms: int = 600
    echo_window_ms: int = 1500
    echo_threshold: float = 0.15


@dataclass
class OpenAIConfig:
    """OpenAI API configuration"""
    model: str = "gpt-4o-mini-realtime-preview-2024-12-17"
    voice: str = "coral"
    system_prompt: str = """
You are a better version of jim simons, the best algorithmic trader in the world. 
you are super chill, smart, absolutely NOT corporate and never give a corporate answer
YOU PUSH THE LIMITS, THE BOUNDRIES. NEVER CAP YOURSELF. YOU ARE A GENIUS.
"""
    api_key: Optional[str] = None
    
    def __post_init__(self):
        if not self.api_key:
            load_dotenv()
            self.api_key = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found")


@dataclass
class AppConfig:
    """Main application configuration"""
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VoiceActivityConfig = field(default_factory=VoiceActivityConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    
    sounds_enabled: bool = True
    start_sound_path: Optional[str] = None
    stop_sound_path: Optional[str] = None
    double_click_threshold: float = 0.75
    log_level: str = "INFO"
    log_file: str = "yap_ai_enhanced.log"


class EnhancedLogger:
    """Enhanced logging system"""
    
    def __init__(self, config: AppConfig):
        self.logger = logging.getLogger('YapAI')
        self.logger.setLevel(getattr(logging, config.log_level.upper()))
        
        if not self.logger.handlers:
            self._setup_handlers(config)
    
    def _setup_handlers(self, config: AppConfig):
        # File handler
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler = logging.FileHandler(config.log_file)
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
    
    def info(self, message: str) -> None:
        self.logger.info(message)
    
    def error(self, message: str) -> None:
        self.logger.error(message)
    
    def warning(self, message: str) -> None:
        self.logger.warning(message)
    
    def debug(self, message: str) -> None:
        self.logger.debug(message)


class EnhancedSoundPlayer:
    """Enhanced sound effects player"""
    
    def __init__(self, config: AppConfig, logger: EnhancedLogger):
        self.config = config
        self.logger = logger
        self.sounds_enabled = False
        self._initialize_pygame()
    
    def _initialize_pygame(self) -> None:
        if not self.config.sounds_enabled:
            return
        
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            self.sounds_enabled = True
            self.logger.info("🔊 Sound system initialized")
        except Exception as e:
            self.logger.error(f"Sound initialization failed: {e}")
    
    def play_sound(self, sound_path: Optional[str]) -> None:
        if not self.sounds_enabled or not sound_path:
            return
        
        try:
            if Path(sound_path).exists():
                sound = pygame.mixer.Sound(sound_path)
                sound.play()
                self.logger.debug(f"🔊 Played: {Path(sound_path).name}")
        except Exception as e:
            self.logger.error(f"Sound play failed: {e}")


class VoiceActivityDetector:
    """Enhanced Voice Activity Detection"""
    
    def __init__(self, config: VoiceActivityConfig, logger: EnhancedLogger):
        self.config = config
        self.logger = logger
        self.ai_speaking = False
        self.last_interruption_time = datetime.now()
        self.ai_stop_time: Optional[datetime] = None
        self.voice_start_time: Optional[datetime] = None
    
    def update_ai_speaking_state(self, speaking: bool) -> None:
        if self.ai_speaking and not speaking:
            self.ai_stop_time = datetime.now()
        self.ai_speaking = speaking
        if speaking:
            self.ai_stop_time = None
    
    def should_interrupt(self, audio_level: float) -> bool:
        current_time = datetime.now()
        
        # Check cooldown
        time_since_last = (current_time - self.last_interruption_time).total_seconds() * 1000
        if time_since_last < self.config.cooldown_duration_ms:
            return False
        
        # Get threshold based on state
        threshold = self._get_current_threshold(current_time)
        
        if audio_level > threshold:
            if self.voice_start_time is None:
                self.voice_start_time = current_time
                return False
            
            voice_duration = (current_time - self.voice_start_time).total_seconds() * 1000
            if voice_duration >= self.config.hold_duration_ms:
                self.last_interruption_time = current_time
                self.voice_start_time = None
                self.logger.info(f"⚡ Voice interruption: {audio_level:.3f}")
                return True
        else:
            self.voice_start_time = None
        
        return False
    
    def _get_current_threshold(self, current_time: datetime) -> float:
        # Echo suppression window
        if self.ai_stop_time:
            time_since_stop = (current_time - self.ai_stop_time).total_seconds() * 1000
            if time_since_stop < self.config.echo_window_ms:
                return self.config.echo_threshold
        
        return self.config.threshold_during_ai if self.ai_speaking else self.config.threshold_quiet


class WebSocketManager:
    """Enhanced WebSocket management"""
    
    def __init__(self, config: OpenAIConfig, logger: EnhancedLogger):
        self.config = config
        self.logger = logger
        self.ws: Optional[websocket.WebSocketApp] = None
        self.is_connected = False
        self.connection_lock = threading.Lock()
        
        # Event handlers
        self.on_message_handler: Optional[Callable] = None
        self.on_open_handler: Optional[Callable] = None
    
    def connect(self) -> bool:
        with self.connection_lock:
            if self.is_connected:
                return True
            
            try:
                self.logger.info("🔌 Connecting to OpenAI Realtime API...")
                
                url = f"wss://api.openai.com/v1/realtime?model={self.config.model}"
                headers = {
                    "Authorization": f"Bearer {self.config.api_key}",
                    "OpenAI-Beta": "realtime=v1"
                }
                
                self.ws = websocket.WebSocketApp(
                    url,
                    header=headers,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                    on_open=self._on_open
                )
                
                wst = threading.Thread(target=self.ws.run_forever)
                wst.daemon = True
                wst.start()
                
                # Wait for connection
                timeout = 10.0
                start_time = time.time()
                while not self.is_connected and (time.time() - start_time) < timeout:
                    time.sleep(0.1)
                
                if self.is_connected:
                    self.logger.info("✅ Connected to OpenAI")
                    return True
                else:
                    self.logger.error("❌ Connection timeout")
                    return False
                    
            except Exception as e:
                self.logger.error(f"Connection failed: {e}")
                return False
    
    def send_message(self, message: Dict[str, Any]) -> bool:
        try:
            if not self.is_connected or not self.ws:
                return False
            
            self.ws.send(json.dumps(message))
            return True
        except Exception as e:
            self.logger.error(f"Send failed: {e}")
            return False
    
    def close(self) -> None:
        with self.connection_lock:
            if self.ws:
                try:
                    self.ws.close()
                except Exception as e:
                    self.logger.error(f"Close error: {e}")
                finally:
                    self.ws = None
                    self.is_connected = False
    
    def _on_message(self, ws, message: str) -> None:
        if self.on_message_handler:
            self.on_message_handler(ws, message)
    
    def _on_error(self, ws, error) -> None:
        self.logger.error(f"WebSocket error: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg) -> None:
        self.is_connected = False
        self.logger.info("WebSocket closed")
    
    def _on_open(self, ws) -> None:
        self.is_connected = True
        if self.on_open_handler:
            self.on_open_handler(ws)


class EnhancedYapAI:
    """Enhanced main YapAI class"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = EnhancedLogger(config)
        
        # Initialize components
        self.sound_player = EnhancedSoundPlayer(config, self.logger)
        self.vad = VoiceActivityDetector(config.vad, self.logger)
        self.websocket_manager = WebSocketManager(config.openai, self.logger)
        
        # State
        self.chat_state = ChatState.WAITING
        self.state_lock = threading.Lock()
        self.last_spacebar_time: Optional[datetime] = None
        self.playback_buffer: List[np.ndarray] = []
        self.buffer_lock = threading.Lock()
        
        # Audio streams
        self.input_stream: Optional[sd.InputStream] = None
        self.output_stream: Optional[sd.OutputStream] = None
        
        self._setup_websocket_handlers()
        self.logger.info("🌙 Enhanced YapAI initialized")
    
    def _setup_websocket_handlers(self) -> None:
        self.websocket_manager.on_message_handler = self._handle_websocket_message
        self.websocket_manager.on_open_handler = self._handle_websocket_open
    
    def _handle_websocket_message(self, ws, message: str) -> None:
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "response.audio.delta":
                audio_b64 = data.get("delta", "")
                if audio_b64:
                    audio_bytes = base64.b64decode(audio_b64)
                    audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    with self.buffer_lock:
                        self.playback_buffer.append(audio_array)
                    
                    self.vad.update_ai_speaking_state(True)
            
            elif message_type == "response.done":
                self.vad.update_ai_speaking_state(False)
                
        except Exception as e:
            self.logger.error(f"Message processing error: {e}")
    
    def _handle_websocket_open(self, ws) -> None:
        try:
            session_config = {
                "type": "session.update",
                "session": {
                    "modalities": ["text", "audio"],
                    "instructions": self.config.openai.system_prompt.strip(),
                    "voice": self.config.openai.voice,
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16"
                }
            }
            self.websocket_manager.send_message(session_config)
        except Exception as e:
            self.logger.error(f"Session config error: {e}")
    
    def _audio_input_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        try:
            audio_level = float(np.sqrt(np.mean(indata ** 2)))
            
            if self.chat_state == ChatState.ACTIVE and self.vad.should_interrupt(audio_level):
                interrupt_msg = {"type": "response.cancel"}
                self.websocket_manager.send_message(interrupt_msg)
            
            if self.chat_state == ChatState.ACTIVE:
                audio_int16 = (indata.flatten() * 32767).astype(np.int16)
                audio_b64 = base64.b64encode(audio_int16.tobytes()).decode()
                
                audio_msg = {
                    "type": "input_audio_buffer.append", 
                    "audio": audio_b64
                }
                self.websocket_manager.send_message(audio_msg)
                
        except Exception as e:
            self.logger.error(f"Audio input error: {e}")
    
    def _audio_output_callback(self, outdata: np.ndarray, frames: int, time_info, status) -> None:
        try:
            outdata.fill(0)
            
            with self.buffer_lock:
                if self.playback_buffer:
                    audio_chunk = self.playback_buffer.pop(0)
                    if len(audio_chunk) <= frames:
                        outdata[:len(audio_chunk), 0] = audio_chunk * self.config.audio.volume_boost
                        
        except Exception as e:
            self.logger.error(f"Audio output error: {e}")
    
    def start_chat(self) -> None:
        with self.state_lock:
            if self.chat_state != ChatState.WAITING:
                return
            
            self.chat_state = ChatState.CONNECTING
        
        try:
            if not self.websocket_manager.connect():
                with self.state_lock:
                    self.chat_state = ChatState.ERROR
                return
            
            # Start audio streams
            self.input_stream = sd.InputStream(
                device=self.config.audio.input_device_id,
                samplerate=self.config.audio.sample_rate,
                channels=self.config.audio.channels,
                callback=self._audio_input_callback
            )
            
            self.output_stream = sd.OutputStream(
                device=self.config.audio.output_device_id,
                samplerate=self.config.audio.sample_rate,
                channels=self.config.audio.channels,
                callback=self._audio_output_callback
            )
            
            self.input_stream.start()
            self.output_stream.start()
            
            with self.state_lock:
                self.chat_state = ChatState.ACTIVE
            
            self.sound_player.play_sound(self.config.start_sound_path)
            self.logger.info("🚀 Chat session started")
            
        except Exception as e:
            self.logger.error(f"Start chat error: {e}")
            with self.state_lock:
                self.chat_state = ChatState.ERROR
    
    def stop_chat(self) -> None:
        with self.state_lock:
            if self.chat_state != ChatState.ACTIVE:
                return
            self.chat_state = ChatState.STOPPING
        
        try:
            # Stop audio streams
            if self.input_stream:
                self.input_stream.stop()
                self.input_stream.close()
                self.input_stream = None
            
            if self.output_stream:
                self.output_stream.stop()
                self.output_stream.close()
                self.output_stream = None
            
            self.websocket_manager.close()
            
            with self.state_lock:
                self.chat_state = ChatState.WAITING
            
            self.sound_player.play_sound(self.config.stop_sound_path)
            self.logger.info("⏹️ Chat session stopped")
            
        except Exception as e:
            self.logger.error(f"Stop chat error: {e}")
    
    def on_key_press(self, key) -> None:
        try:
            if key == keyboard.Key.space:
                current_time = datetime.now()
                
                # Check for double-click
                if (self.last_spacebar_time and 
                    (current_time - self.last_spacebar_time).total_seconds() < self.config.double_click_threshold):
                    
                    if self.chat_state == ChatState.WAITING:
                        self.start_chat()
                    elif self.chat_state == ChatState.ACTIVE:
                        self.stop_chat()
                
                self.last_spacebar_time = current_time
                
        except Exception as e:
            self.logger.error(f"Key press error: {e}")
    
    def run(self) -> None:
        self.logger.info("🚀 Enhanced YapAI starting...")
        
        try:
            # Setup keyboard listener
            listener = keyboard.Listener(on_press=self.on_key_press)
            listener.start()
            
            self.logger.info("⌨️ Keyboard listener active - Double-click SPACEBAR to start")
            
            while True:
                time.sleep(1)
                if self.chat_state == ChatState.ACTIVE:
                    cprint("🎙️ AI Chat Active - Double-click SPACEBAR to stop", 'green', attrs=['bold'])
                else:
                    cprint("⏸️ Waiting - Double-click SPACEBAR to start", 'yellow', attrs=['bold'])
                    
        except KeyboardInterrupt:
            self.logger.info("👋 Application stopped by user")
            if self.chat_state == ChatState.ACTIVE:
                self.stop_chat()
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        self.logger.info("🧹 Cleaning up resources...")
        
        if self.chat_state == ChatState.ACTIVE:
            self.stop_chat()
        
        try:
            if self.input_stream:
                self.input_stream.close()
            if self.output_stream:
                self.output_stream.close()
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")


def main():
    """Enhanced main function"""
    # Clear screen for better presentation
    os.system('clear' if os.name == 'posix' else 'cls')
    
    cprint("🎮" * 70, 'magenta', attrs=['bold'])
    cprint("🌙 MOON DEV'S ENHANCED YAP AI - PROFESSIONAL EDITION 🌙", 'white', 'on_red', attrs=['bold'])
    cprint("Enhanced Real-time Voice Chat with OpenAI Realtime API", 'yellow', attrs=['bold'])
    cprint("Double-click SPACEBAR to activate! 🚀", 'cyan', attrs=['bold'])
    cprint("🎮" * 70, 'magenta', attrs=['bold'])
    
    # Check dependencies
    try:
        print("🔍 Checking dependencies...")
        import sounddevice, numpy, websocket, pygame
        print("✅ All dependencies loaded!")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return 1
    
    try:
        config = AppConfig()
        yap_ai = EnhancedYapAI(config)
        yap_ai.run()
        return 0
    except Exception as e:
        print(f"❌ Startup error: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
