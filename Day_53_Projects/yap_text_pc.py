#!/usr/bin/env python3
"""
🌙 MOON DEV's Yap Text - Pure Whisper Edition (PC Compatible) 🌙
Right arrow push-to-talk with pure Whisper AI transcription
Super simple, super fast, built with love by MOON DEV 🚀
Cross-platform: Works on Windows, Mac, and Linux! 💻
"""

import os
import platform

# 🎯 MOON DEV Configuration - Cross Platform!
WHISPER_MODEL = "base.en"  # Options: tiny.en, base.en, small.en (matches Wispr Flow!)

# Cross-platform sound file paths (OPTIONAL - set to None or "" to disable sounds)
SOUND_DIR = os.path.join(os.path.dirname(__file__))
START_SOUND = os.path.join(SOUND_DIR, "crack5.wav")  # Recording start sound
STOP_SOUND = os.path.join(SOUND_DIR, "crack1.wav")   # Recording stop sound

# 🔊 MOON DEV Sound Options:
# To disable sounds, comment out or set to None:
START_SOUND = None
STOP_SOUND = None


import subprocess
import threading
import time
import wave
from datetime import datetime

import pyperclip
import sounddevice as sd
import numpy as np
from pynput import keyboard
from faster_whisper import WhisperModel
import pygame


class SoundPlayer:
    """MOON DEV's Sound Effects Player 🔊 - Cross Platform!"""
    
    def __init__(self):
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            print("🔊 MOON DEV: Cross-platform sound system ready! 🌙")
            self.sounds_enabled = True
        except Exception as e:
            print(f"⚠️ MOON DEV: Sound init failed: {e}")
            self.sounds_enabled = False
    
    def play_sound(self, sound_file):
        """Play a sound file - works on Windows, Mac, Linux! (Optional sounds)"""
        # Skip if sounds disabled or no sound system
        if not self.sounds_enabled:
            return
            
        # Skip if no sound file specified (None, empty string, etc.)
        if not sound_file or sound_file.strip() == "":
            print("🔇 MOON DEV: No sound file specified (sounds disabled) 🌙")
            return
            
        try:
            if os.path.exists(sound_file):
                sound = pygame.mixer.Sound(sound_file)
                sound.play()
                print(f"🔊 MOON DEV played: {os.path.basename(sound_file)} 🌙")
            else:
                print(f"🔇 MOON DEV: Sound file not found, continuing silently: {os.path.basename(sound_file)} 🌙")
        except Exception as e:
            print(f"🔇 MOON DEV: Sound disabled due to error: {e} 🌙")


class WhisperTranscriber:
    """MOON DEV's Pure Whisper transcription - simple and fast! 🌙"""
    
    def __init__(self, model_size="base.en"):
        self.model_size = model_size
        self.model = None
        print(f"🎤 MOON DEV initializing Whisper {model_size}... 🌙")
        
    def load_model(self):
        """Load Whisper model on first use"""
        if self.model is None:
            print("🤖 MOON DEV loading Whisper AI model... 🌙")
            try:
                self.model = WhisperModel(
                    self.model_size, 
                    device="auto",
                    compute_type="auto"
                )
                print(f"✅ MOON DEV Whisper {self.model_size} loaded! 🚀")
                return True
            except Exception as e:
                print(f"❌ MOON DEV Whisper load failed: {e}")
                return False
        return True
    
    def transcribe(self, audio_file):
        """Transcribe audio file with Whisper"""
        if not self.load_model():
            return None
            
        try:
            print("🎯 MOON DEV transcribing with Whisper AI... 🌙")
            
            segments, info = self.model.transcribe(
                audio_file,
                language="en",
                beam_size=5,
                best_of=5,
                temperature=0.0,
                condition_on_previous_text=False
            )
            
            # Combine all segments
            transcription = ""
            for segment in segments:
                transcription += segment.text + " "
            
            transcription = transcription.strip()
            
            if transcription:
                print(f"🌙 MOON DEV Whisper transcribed: '{transcription}' ✨")
                return transcription
            else:
                print("🤷 MOON DEV: Whisper found no speech")
                return None
                
        except Exception as e:
            print(f"❌ MOON DEV Whisper transcription error: {e}")
            return None


class CrossPlatformPaster:
    """MOON DEV's Cross-Platform Auto-Paste! Works on Windows, Mac, Linux! 🌙"""
    
    def __init__(self):
        self.system = platform.system().lower()
        print(f"💻 MOON DEV detected OS: {self.system} 🌙")
        
        # Try to import keyboard library for better cross-platform support
        self.has_keyboard = False
        self.keyboard = None
        
        # Only try keyboard library on Windows/Linux - can cause bus errors on Mac
        if self.system in ['windows', 'linux']:
            try:
                import keyboard as kb
                self.keyboard = kb
                self.has_keyboard = True
                print("⌨️ MOON DEV: Advanced keyboard control loaded! 🚀")
            except (ImportError, OSError, Exception) as e:
                print(f"⚠️ MOON DEV: Keyboard library not available: {e}")
                print("📦 For better paste: pip install keyboard")
        else:
            print("🍎 MOON DEV: Using native Mac paste method! 🌙")
    
    def paste_text(self):
        """Cross-platform auto-paste function"""
        try:
            if self.has_keyboard:
                # Use keyboard library - works great on Windows!
                print("⌨️ MOON DEV: Using advanced keyboard paste! 🌙")
                self.keyboard.send('ctrl+v')
                print("✅ MOON DEV: Cross-platform paste successful! 🚀")
                return True
            
            elif self.system == "darwin":  # macOS
                print("🍎 MOON DEV: Using macOS AppleScript paste! 🌙")
                subprocess.run([
                    'osascript', '-e',
                    'tell application "System Events" to keystroke "v" using command down'
                ], check=True)
                print("✅ MOON DEV: macOS paste successful! 🚀")
                return True
                
            elif self.system == "windows":
                print("🪟 MOON DEV: Using Windows paste fallback! 🌙")
                # Windows fallback using pynput
                from pynput.keyboard import Key, Controller
                kb_controller = Controller()
                kb_controller.press(Key.ctrl)
                kb_controller.press('v')
                kb_controller.release('v')
                kb_controller.release(Key.ctrl)
                print("✅ MOON DEV: Windows paste successful! 🚀")
                return True
                
            elif self.system == "linux":
                print("🐧 MOON DEV: Using Linux paste fallback! 🌙")
                # Linux fallback
                from pynput.keyboard import Key, Controller
                kb_controller = Controller()
                kb_controller.press(Key.ctrl)
                kb_controller.press('v')
                kb_controller.release('v')
                kb_controller.release(Key.ctrl)
                print("✅ MOON DEV: Linux paste successful! 🚀")
                return True
            
            else:
                print(f"⚠️ MOON DEV: Unknown OS {self.system} - paste may not work!")
                return False
                
        except Exception as e:
            print(f"❌ MOON DEV paste error: {e}")
            return False


class YapText:
    def __init__(self, whisper_model="base.en"):
        print("🌙 MOON DEV Yap Text Pure Whisper (PC Compatible) initializing... 🚀")
        
        # Audio settings
        self.sample_rate = 16000
        self.channels = 1
        
        # Recording state
        self.is_recording = False
        self.audio_data = []
        self.recording_start_time = 0  # Track recording duration
        
        # Create temp directory
        self.temp_dir = "temp_audio"
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Hotkey settings
        self.hotkey = keyboard.Key.shift_r  # Right shift key instead of right arrow
        self.is_hotkey_pressed = False
        self.should_quit = False  # Manual quit flag For example, I'm speaking right here.
        
        # Whisper transcription
        self.whisper = WhisperTranscriber(whisper_model)
        
        # Sound effects
        self.sound_player = SoundPlayer()
        
        # Cross-platform pasting
        self.paster = CrossPlatformPaster()
        
        print("🌙 MOON DEV Yap Text Pure Whisper (PC Compatible) ready! ⚡")
        print("💡 MOON DEV Tips for better transcription:")
        print("   🔊 Speak clearly and loudly")
        print("   ⏱️ Hold for at least 1 second")
        print("   🎤 Make sure your microphone works")
        print("   💻 Works on Windows, Mac, and Linux!")
    
    def audio_callback(self, indata, frames, time, status):
        """Audio stream callback"""
        if status:
            print(f"⚠️ MOON DEV audio status: {status}")
        
        if self.is_recording:
            self.audio_data.extend(indata[:, 0])
    
    def save_audio_file(self, filename):
        """Save recorded audio to WAV"""
        try:
            print(f"💾 MOON DEV saving audio... 🌙")
            
            audio_array = np.array(self.audio_data, dtype=np.float32)
            
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                
                audio_int16 = (audio_array * 32767).astype(np.int16)
                wav_file.writeframes(audio_int16.tobytes())
            
            print(f"✅ MOON DEV audio saved! 🌟")
            return True
            
        except Exception as e:
            print(f"❌ MOON DEV save error: {e}")
            return False
    
    def copy_and_paste(self, text):
        """Copy to clipboard and auto-paste - Cross Platform!"""
        if not text or not text.strip():
            print("🤷 MOON DEV: Nothing to paste! 🌙")
            return
        
        try:
            pyperclip.copy(text.strip())
            print(f"📋 MOON DEV copied: '{text.strip()}' ✨")
            
            time.sleep(0.1)
            
            # Use cross-platform paster
            if self.paster.paste_text():
                print("⌨️ MOON DEV Pure Whisper cross-platform auto-pasted! 🎯")
            else:
                print("⚠️ MOON DEV: Auto-paste failed, but text is in clipboard! 📋")
                print("💡 MOON DEV: Try Ctrl+V manually or install 'keyboard' library!")
            
        except Exception as e:
            print(f"❌ MOON DEV paste error: {e}")
    
    def start_recording(self):
        """Start recording"""
        if not self.is_recording:
            print("🎤 MOON DEV: Recording with RIGHT SHIFT! 🌙")
            # Play start sound effect
            self.sound_player.play_sound(START_SOUND)
            self.is_recording = True
            self.audio_data = []
            self.recording_start_time = time.time()  # Track recording duration
    
    def stop_recording(self):
        """Stop recording and process"""
        if self.is_recording:
            print("⏹️ MOON DEV: Processing with pure Whisper... 🌙")
            # Play stop sound effect
            self.sound_player.play_sound(STOP_SOUND)
            self.is_recording = False
            
            # Check recording duration
            recording_duration = time.time() - self.recording_start_time
            print(f"⏱️ MOON DEV: Recording duration: {recording_duration:.2f} seconds")
            
            if len(self.audio_data) == 0:
                print("🤷 MOON DEV: No audio recorded! 🌙")
                return
            
            # Check if recording is too short
            if recording_duration < 0.5:
                print("⚠️ MOON DEV: Recording too short! Hold longer for better results! 🌙")
                return
            
            # Check audio levels
            audio_array = np.array(self.audio_data, dtype=np.float32)
            max_amplitude = np.max(np.abs(audio_array))
            avg_amplitude = np.mean(np.abs(audio_array))
            
            print(f"🔊 MOON DEV: Audio max level: {max_amplitude:.3f}, avg level: {avg_amplitude:.3f}")
            
            if max_amplitude < 0.01:
                print("⚠️ MOON DEV: Audio too quiet! Speak louder or check microphone! 🌙")
                return
            
            if avg_amplitude < 0.005:
                print("⚠️ MOON DEV: Average audio too low! Try speaking more clearly! 🌙")
                return
            
            # Save audio
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_file = os.path.join(self.temp_dir, f"yap_{timestamp}.wav")
            
            if self.save_audio_file(audio_file):
                print(f"📊 MOON DEV: Audio file size: {os.path.getsize(audio_file)} bytes")
                
                # Transcribe with pure Whisper
                transcribed_text = self.whisper.transcribe(audio_file)
                
                if transcribed_text and transcribed_text.strip():
                    # Check for common Whisper failure patterns
                    text_lower = transcribed_text.lower().strip()
                    
                    if text_lower in ['you', 'thank you', 'thanks']:
                        print("⚠️ MOON DEV: Whisper returned generic response - try speaking longer/louder! 🌙")
                        print(f"🔍 MOON DEV: Got: '{transcribed_text}' (probably not what you said)")
                        # Still copy it in case it's actually correct
                        self.copy_and_paste(transcribed_text)
                    else:
                        print("⚡ MOON DEV: Using pure Whisper text (no cleanup needed)!")
                        # Copy and paste directly
                        self.copy_and_paste(transcribed_text)
                else:
                    print("❌ MOON DEV: Whisper transcription failed! 🌙")
                
                # Cleanup temp file
                try:
                    os.remove(audio_file)
                    print("🧹 MOON DEV: Cleaned up temp file")
                except:
                    pass
    
    def on_key_press(self, key):
        """Handle key press"""
        try:
            if key == self.hotkey and not self.is_hotkey_pressed:
                self.is_hotkey_pressed = True
                self.start_recording()
        except Exception as e:
            print(f"⚠️ MOON DEV key press error: {e}")
    
    def on_key_release(self, key):
        """Handle key release"""
        try:
            if key == self.hotkey and self.is_hotkey_pressed:
                self.is_hotkey_pressed = False
                threading.Thread(target=self.stop_recording, daemon=True).start()
        except Exception as e:
            print(f"⚠️ MOON DEV key release error: {e}")
    
    def run(self):
        """Main run loop"""
        print("🚀 MOON DEV Pure Whisper Yap Text (PC Compatible) launching! 🌙")
        print("⬆️ Hold RIGHT SHIFT to record, release for pure Whisper transcription!")
        print("🛑 Press Ctrl+C to quit (NO automatic shutdown!)")
        print("⚡ MOON DEV: Pure Whisper mode - no AI cleanup, just raw accuracy!")
        print("🔊 MOON DEV: Sound effects enabled for that premium feel! 🌟")
        print("💻 MOON DEV: Cross-platform compatible - Windows, Mac, Linux! 🌍")
        print("♾️ MOON DEV: App will run FOREVER until you manually stop it! 🌙")
        
        # Start audio stream
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self.audio_callback,
            blocksize=1024
        ):
            print("🎤 MOON DEV audio stream active! 🌟")
            
            # Start keyboard listener - simplified version
            try:
                listener = keyboard.Listener(
                    on_press=self.on_key_press,
                    on_release=self.on_key_release
                )
                listener.start()
                print("⌨️ MOON DEV hotkey listener active! ⬆️")
                print("🔒 MOON DEV: NO automatic shutdown - only manual Ctrl+C!")
                
            except Exception as e:
                print(f"⚠️ MOON DEV listener setup error: {e}")
                print("🔄 MOON DEV: Continuing without hotkeys...")
            
            # Super simple infinite loop - NO quit conditions!
            print("♾️ MOON DEV: Entering INFINITE loop... Only Ctrl+C can stop this!")
            
            try:
                while True:  # Infinite loop with NO quit conditions
                    time.sleep(1)  # Sleep 1 second at a time
                    #print("💖 MOON DEV: Still alive and recording! Right shift to use! 🌙")
                    
            except KeyboardInterrupt:
                print("\n👋 MOON DEV: Manual Ctrl+C shutdown detected!")
                
            except Exception as e:
                print(f"⚠️ MOON DEV unexpected error: {e}")
                print("🔄 MOON DEV: But we keep going anyway!")
                # Even on error, restart the loop!
                try:
                    while True:
                        time.sleep(1)
                        print("🛡️ MOON DEV: Error recovery mode - still alive! 🌙")
                except KeyboardInterrupt:
                    print("\n👋 MOON DEV: Finally stopping via Ctrl+C!")
                    
        print("👋 MOON DEV: Audio stream closed, goodbye! 🚀")


def main():
    """MOON DEV Pure Whisper Yap Text Entry Point (PC Compatible)"""
    print("=" * 60)
    print("🌙 MOON DEV YAP TEXT - PURE WHISPER (PC COMPATIBLE) 🌙")
    print("Right Shift Push-to-Talk + Pure Whisper AI")
    print("Cross-Platform: Windows, Mac, Linux! 💻")
    print("Super simple, super fast! 🚀")
    print("=" * 60)
    
    # Check dependencies
    try:
        import sounddevice
        import pyperclip
        import numpy
        from pynput import keyboard
        from faster_whisper import WhisperModel
        import pygame
        print("✅ MOON DEV: All core dependencies loaded! 🚀")
        
        # Check for optional keyboard library (safer check)
        if platform.system().lower() in ['windows', 'linux']:
            try:
                import keyboard as kb
                print("✅ MOON DEV: Advanced keyboard library loaded! 💯")
            except (ImportError, OSError, Exception):
                print("⚠️ MOON DEV: Optional 'keyboard' library not found")
                print("📦 For better paste support: pip install keyboard")
        else:
            print("🍎 MOON DEV: Mac native paste mode (no keyboard lib needed)! 🌙")
            
    except ImportError as e:
        print(f"❌ MOON DEV missing dependency: {e}")
        print("📦 Install: pip install sounddevice pyperclip numpy pynput faster-whisper pygame")
        print("📦 Optional: pip install keyboard")
        return
    
    print(f"⚡ MOON DEV using pure Whisper: {WHISPER_MODEL}")
    
    # Show sound status
    if START_SOUND and os.path.exists(START_SOUND):
        print(f"🔊 MOON DEV start sound: {os.path.basename(START_SOUND)}")
    else:
        print("🔇 MOON DEV start sound: DISABLED (no file or set to None)")
        
    if STOP_SOUND and os.path.exists(STOP_SOUND):
        print(f"🔊 MOON DEV stop sound: {os.path.basename(STOP_SOUND)}")
    else:
        print("🔇 MOON DEV stop sound: DISABLED (no file or set to None)")
        
    print(f"💻 MOON DEV detected OS: {platform.system()}")
    
    try:
        yap_text = YapText(whisper_model=WHISPER_MODEL)
        yap_text.run()
    except Exception as e:
        print(f"❌ MOON DEV startup error: {e}")


if __name__ == "__main__":
    main() 