#!/usr/bin/env python3
"""
🌙 MOON DEV's Enhanced Yap Text - Professional Whisper Edition 🌙
Right arrow push-to-talk with pure Whisper AI transcription
Enhanced with modern Python practices, better error handling, and improved performance
Cross-platform: Works on Windows, Mac, and Linux! 💻

Enhanced Features:
- Professional logging system
- Type hints for better code documentation
- Configuration management with dataclasses
- Resource management and cleanup
- Performance optimizations
- Enhanced error handling
- Better cross-platform compatibility
- Improved audio stream management
"""

import pygame
import platform
import subprocess
import threading
import time
import wave
import logging
import pyperclip
import sounddevice as sd
import numpy as np
from datetime import datetime
from typing import Optional, Union, List
from dataclasses import dataclass, field
from contextlib import contextmanager
from pathlib import Path
from enum import Enum
from pynput import keyboard
from faster_whisper import WhisperModel


class OSType(Enum):
    """Supported operating system types"""
    WINDOWS = "windows"
    MACOS = "darwin"
    LINUX = "linux"
    UNKNOWN = "unknown"


@dataclass
class AudioConfig:
    """Enhanced audio configuration with validation"""
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = 'float32'
    chunk_size: int = 1024
    buffer_seconds: float = 10.0
    
    def __post_init__(self):
        """Validate audio configuration"""
        if self.sample_rate not in [16000, 22050, 44100, 48000]:
            raise ValueError(f"Unsupported sample rate: {self.sample_rate}")
        if self.channels not in [1, 2]:
            raise ValueError(f"Unsupported channel count: {self.channels}")


@dataclass
class WhisperConfig:
    """Enhanced Whisper model configuration"""
    model_size: str = "base.en"
    device: str = "auto"
    compute_type: str = "auto"
    beam_size: int = 5
    best_of: int = 5
    temperature: float = 0.0
    condition_on_previous_text: bool = False
    language: str = "en"
    
    def __post_init__(self):
        """Validate Whisper configuration"""
        valid_models = ["tiny.en", "base.en", "small.en", "medium.en", "large-v2", "large-v3"]
        if self.model_size not in valid_models:
            raise ValueError(f"Unsupported model: {self.model_size}. Choose from {valid_models}")


@dataclass
class AppConfig:
    """Main application configuration"""
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    
    # Sound settings
    sounds_enabled: bool = True
    start_sound_path: Optional[str] = None
    stop_sound_path: Optional[str] = None
    
    # Hotkey settings
    hotkey: str = "right_shift"
    
    # Performance settings
    auto_paste: bool = True
    clipboard_copy: bool = True
    
    # Logging settings
    log_level: str = "INFO"
    log_file: str = "yap_text_enhanced.log"
    
    def __post_init__(self):
        """Initialize sound paths if sounds are enabled"""
        if self.sounds_enabled:
            sound_dir = Path(__file__).parent
            self.start_sound_path = str(sound_dir / "crack5.wav")
            self.stop_sound_path = str(sound_dir / "crack1.wav")


class EnhancedLogger:
    """Enhanced logging system with file and console output"""
    
    def __init__(self, config: AppConfig):
        self.logger = logging.getLogger('YapText')
        self.logger.setLevel(getattr(logging, config.log_level.upper()))
        
        # Avoid duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers(config)
    
    def _setup_handlers(self, config: AppConfig):
        """Setup logging handlers"""
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        # File handler
        file_handler = logging.FileHandler(config.log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
    
    def info(self, message: str, *args) -> None:
        self.logger.info(message, *args)
    
    def error(self, message: str, *args) -> None:
        self.logger.error(message, *args)
    
    def warning(self, message: str, *args) -> None:
        self.logger.warning(message, *args)
    
    def debug(self, message: str, *args) -> None:
        self.logger.debug(message, *args)


class EnhancedSoundPlayer:
    """Enhanced sound effects player with better error handling"""
    
    def __init__(self, config: AppConfig, logger: EnhancedLogger):
        self.config = config
        self.logger = logger
        self.sounds_enabled = False
        self._initialize_pygame()
    
    def _initialize_pygame(self) -> None:
        """Initialize pygame mixer with error handling"""
        if not self.config.sounds_enabled:
            self.logger.info("🔇 Sound effects disabled by configuration")
            return
            
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            self.sounds_enabled = True
            self.logger.info("🔊 Cross-platform sound system initialized successfully")
        except Exception as e:
            self.logger.error(f"Sound initialization failed: {e}")
            self.sounds_enabled = False
    
    def play_sound(self, sound_path: Optional[str]) -> None:
        """Play a sound file with enhanced error handling"""
        if not self.sounds_enabled or not sound_path:
            return
            
        try:
            if Path(sound_path).exists():
                sound = pygame.mixer.Sound(sound_path)
                sound.play()
                self.logger.debug(f"🔊 Played sound: {Path(sound_path).name}")
            else:
                self.logger.warning(f"🔇 Sound file not found: {Path(sound_path).name}")
        except Exception as e:
            self.logger.error(f"Failed to play sound {sound_path}: {e}")
    
    def cleanup(self) -> None:
        """Cleanup pygame mixer resources"""
        if self.sounds_enabled:
            try:
                pygame.mixer.quit()
                self.logger.debug("🔊 Sound system cleaned up")
            except Exception as e:
                self.logger.error(f"Error cleaning up sound system: {e}")


class EnhancedWhisperTranscriber:
    """Enhanced Whisper transcriber with better resource management"""
    
    def __init__(self, config: WhisperConfig, logger: EnhancedLogger):
        self.config = config
        self.logger = logger
        self.model: Optional[WhisperModel] = None
        self._model_lock = threading.Lock()
        
    def load_model(self) -> bool:
        """Thread-safe model loading with enhanced error handling"""
        with self._model_lock:
            if self.model is not None:
                return True
                
            try:
                self.logger.info(f"🤖 Loading Whisper model: {self.config.model_size}")
                self.model = WhisperModel(
                    self.config.model_size,
                    device=self.config.device,
                    compute_type=self.config.compute_type
                )
                self.logger.info(f"✅ Whisper model {self.config.model_size} loaded successfully")
                return True
            except Exception as e:
                self.logger.error(f"Failed to load Whisper model: {e}")
                return False
    
    def transcribe(self, audio_file: Union[str, Path]) -> Optional[str]:
        """Enhanced transcription with better error handling"""
        if not self.load_model():
            return None
        
        try:
            self.logger.debug(f"🎯 Transcribing audio file: {audio_file}")
            
            segments, info = self.model.transcribe(
                str(audio_file),
                language=self.config.language,
                beam_size=self.config.beam_size,
                best_of=self.config.best_of,
                temperature=self.config.temperature,
                condition_on_previous_text=self.config.condition_on_previous_text
            )
            
            # Combine all segments
            transcription = " ".join(segment.text.strip() for segment in segments)
            
            if transcription.strip():
                self.logger.info(f"✅ Transcription successful: {len(transcription)} characters")
                return transcription.strip()
            else:
                self.logger.warning("⚠️ Transcription returned empty result")
                return None
                
        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            return None
    
    def cleanup(self) -> None:
        """Cleanup model resources"""
        with self._model_lock:
            if self.model is not None:
                try:
                    # The WhisperModel doesn't have an explicit cleanup method
                    # but we can remove our reference
                    self.model = None
                    self.logger.debug("🤖 Whisper model resources cleaned up")
                except Exception as e:
                    self.logger.error(f"Error cleaning up Whisper model: {e}")


class EnhancedCrossPlatformPaster:
    """Enhanced cross-platform auto-paste with better OS detection"""
    
    def __init__(self, config: AppConfig, logger: EnhancedLogger):
        self.config = config
        self.logger = logger
        self.os_type = self._detect_os()
        self.keyboard_available = self._check_keyboard_library()
        
    def _detect_os(self) -> OSType:
        """Detect the operating system type"""
        system = platform.system().lower()
        if system == "windows":
            return OSType.WINDOWS
        elif system == "darwin":
            return OSType.MACOS
        elif system == "linux":
            return OSType.LINUX
        else:
            return OSType.UNKNOWN
    
    def _check_keyboard_library(self) -> bool:
        """Check if the keyboard library is available"""
        try:
            import keyboard
            return True
        except (ImportError, OSError):
            return False
    
    def copy_to_clipboard(self, text: str) -> bool:
        """Copy text to clipboard with error handling"""
        if not self.config.clipboard_copy:
            return True
            
        try:
            pyperclip.copy(text)
            self.logger.debug(f"📋 Copied {len(text)} characters to clipboard")
            return True
        except Exception as e:
            self.logger.error(f"Failed to copy to clipboard: {e}")
            return False
    
    def paste_text(self) -> bool:
        """Enhanced cross-platform auto-paste"""
        if not self.config.auto_paste:
            return True
            
        try:
            if self.os_type == OSType.WINDOWS and self.keyboard_available:
                import keyboard
                keyboard.send('ctrl+v')
                self.logger.debug("📨 Windows: Pasted using keyboard library")
                
            elif self.os_type == OSType.LINUX and self.keyboard_available:
                import keyboard
                keyboard.send('ctrl+v')
                self.logger.debug("📨 Linux: Pasted using keyboard library")
                
            elif self.os_type == OSType.MACOS:
                subprocess.run(['osascript', '-e', 'tell application "System Events" to keystroke "v" using command down'], 
                             check=True, capture_output=True)
                self.logger.debug("📨 macOS: Pasted using AppleScript")
                
            else:
                self.logger.warning(f"⚠️ Auto-paste not supported on {self.os_type.value}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Auto-paste failed: {e}")
            return False


class EnhancedYapText:
    """Enhanced main YapText class with improved architecture"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = EnhancedLogger(config)
        
        # Initialize components
        self.sound_player = EnhancedSoundPlayer(config, self.logger)
        self.transcriber = EnhancedWhisperTranscriber(config.whisper, self.logger)
        self.paster = EnhancedCrossPlatformPaster(config, self.logger)
        
        # Audio recording state
        self.is_recording = False
        self.audio_data: List[np.ndarray] = []
        self.audio_stream: Optional[sd.InputStream] = None
        self.recording_lock = threading.Lock()
        
        # Temporary files
        self.temp_dir = Path("temp_audio")
        self.temp_dir.mkdir(exist_ok=True)
        
        self.logger.info("🌙 Enhanced YapText initialized successfully")
    
    @contextmanager
    def audio_stream_context(self):
        """Context manager for audio stream"""
        try:
            self.audio_stream = sd.InputStream(
                samplerate=self.config.audio.sample_rate,
                channels=self.config.audio.channels,
                dtype=self.config.audio.dtype,
                callback=self.audio_callback,
                blocksize=self.config.audio.chunk_size
            )
            self.audio_stream.start()
            self.logger.debug("🎤 Audio stream started")
            yield self.audio_stream
        except Exception as e:
            self.logger.error(f"Audio stream error: {e}")
            raise
        finally:
            if self.audio_stream:
                try:
                    self.audio_stream.stop()
                    self.audio_stream.close()
                    self.logger.debug("🎤 Audio stream stopped and closed")
                except Exception as e:
                    self.logger.error(f"Error closing audio stream: {e}")
                finally:
                    self.audio_stream = None
    
    def audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """Enhanced audio callback with error handling"""
        if status:
            self.logger.warning(f"Audio callback status: {status}")
        
        with self.recording_lock:
            if self.is_recording:
                self.audio_data.append(indata.copy())
    
    def save_audio_file(self, filename: Union[str, Path]) -> bool:
        """Save recorded audio to WAV file with enhanced error handling"""
        try:
            if not self.audio_data:
                self.logger.warning("No audio data to save")
                return False
            
            audio_array = np.concatenate(self.audio_data, axis=0)
            
            # Normalize audio to prevent clipping
            if np.max(np.abs(audio_array)) > 0:
                audio_array = audio_array / np.max(np.abs(audio_array)) * 0.95
            
            with wave.open(str(filename), 'wb') as wf:
                wf.setnchannels(self.config.audio.channels)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.config.audio.sample_rate)
                
                # Convert to 16-bit integers
                audio_int16 = (audio_array * 32767).astype(np.int16)
                wf.writeframes(audio_int16.tobytes())
            
            self.logger.debug(f"💾 Audio saved: {filename} ({len(audio_array)} samples)")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save audio file {filename}: {e}")
            return False
    
    def start_recording(self) -> None:
        """Start audio recording with enhanced logging"""
        with self.recording_lock:
            if self.is_recording:
                return
                
            self.is_recording = True
            self.audio_data.clear()
            
        self.sound_player.play_sound(self.config.start_sound_path)
        self.logger.info("🎙️ Recording started")
    
    def stop_recording_and_process(self) -> None:
        """Stop recording and process audio with enhanced error handling"""
        with self.recording_lock:
            if not self.is_recording:
                return
            self.is_recording = False
        
        self.sound_player.play_sound(self.config.stop_sound_path)
        self.logger.info("⏹️ Recording stopped")
        
        if not self.audio_data:
            self.logger.warning("No audio data recorded")
            return
        
        # Process in a separate thread to avoid blocking the UI
        processing_thread = threading.Thread(target=self._process_audio, daemon=True)
        processing_thread.start()
    
    def _process_audio(self) -> None:
        """Process recorded audio in a separate thread"""
        try:
            # Create temporary audio file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_audio_file = self.temp_dir / f"recording_{timestamp}.wav"
            
            if not self.save_audio_file(temp_audio_file):
                return
            
            # Transcribe audio
            transcription = self.transcriber.transcribe(temp_audio_file)
            
            if transcription:
                self.logger.info(f"📝 Transcription: {transcription}")
                
                # Copy to clipboard and paste
                if self.paster.copy_to_clipboard(transcription):
                    time.sleep(0.1)  # Brief delay before pasting
                    self.paster.paste_text()
                    
            else:
                self.logger.warning("❌ No transcription result")
            
            # Cleanup temporary file
            try:
                temp_audio_file.unlink()
                self.logger.debug(f"🗑️ Cleaned up temporary file: {temp_audio_file}")
            except Exception as e:
                self.logger.warning(f"Failed to cleanup temporary file: {e}")
                
        except Exception as e:
            self.logger.error(f"Audio processing error: {e}")
    
    def on_key_press(self, key) -> None:
        """Handle key press events with enhanced logging"""
        try:
            if key == keyboard.Key.shift_r:  # Right shift key
                self.start_recording()
        except Exception as e:
            self.logger.error(f"Key press handler error: {e}")
    
    def on_key_release(self, key) -> None:
        """Handle key release events with enhanced logging"""
        try:
            if key == keyboard.Key.shift_r:  # Right shift key
                self.stop_recording_and_process()
        except Exception as e:
            self.logger.error(f"Key release handler error: {e}")
    
    def run(self) -> None:
        """Enhanced main run loop with better error handling"""
        self.logger.info("🚀 Enhanced YapText starting...")
        self.logger.info(f"🎯 Hotkey: {self.config.hotkey}")
        self.logger.info(f"🤖 Whisper model: {self.config.whisper.model_size}")
        self.logger.info(f"💻 OS: {self.paster.os_type.value}")
        
        try:
            with self.audio_stream_context():
                # Setup keyboard listener
                with keyboard.Listener(
                    on_press=self.on_key_press,
                    on_release=self.on_key_release
                ) as listener:
                    
                    self.logger.info("⌨️ Keyboard listener active - Press Right Shift to record")
                    self.logger.info("🔄 Press Ctrl+C to stop")
                    
                    # Keep the application running
                    listener.join()
                    
        except KeyboardInterrupt:
            self.logger.info("👋 Application stopped by user (Ctrl+C)")
        except Exception as e:
            self.logger.error(f"Runtime error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Cleanup all resources"""
        self.logger.info("🧹 Cleaning up resources...")
        
        # Cleanup components
        self.sound_player.cleanup()
        self.transcriber.cleanup()
        
        # Cleanup temporary directory
        try:
            if self.temp_dir.exists():
                for file in self.temp_dir.glob("*.wav"):
                    file.unlink()
                self.temp_dir.rmdir()
                self.logger.debug("🗑️ Temporary directory cleaned up")
        except Exception as e:
            self.logger.warning(f"Failed to cleanup temporary directory: {e}")
        
        self.logger.info("✅ Cleanup completed")


def check_dependencies() -> bool:
    """Check if all required dependencies are available"""
    required_packages = [
        ('sounddevice', 'sounddevice'),
        ('pyperclip', 'pyperclip'), 
        ('numpy', 'numpy'),
        ('pynput', 'pynput'),
        ('faster_whisper', 'faster-whisper'),
        ('pygame', 'pygame')
    ]
    
    missing_packages = []
    
    for package_name, pip_name in required_packages:
        try:
            __import__(package_name)
        except ImportError:
            missing_packages.append(pip_name)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print(f"📦 Install with: pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All required dependencies are available")
    return True


def main():
    """Enhanced main function with comprehensive setup"""
    print("=" * 70)
    print("🌙 MOON DEV ENHANCED YAP TEXT - PROFESSIONAL WHISPER EDITION 🌙")
    print("Right Shift Push-to-Talk + Enhanced Whisper AI Transcription")
    print("Cross-Platform: Windows, Mac, Linux! 💻")
    print("Enhanced with modern Python practices! 🚀")
    print("=" * 70)
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    try:
        # Initialize configuration
        config = AppConfig()
        
        # Create and run the application
        yap_text = EnhancedYapText(config)
        yap_text.run()
        
        return 0
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
