#!/usr/bin/env python3
"""
🌙 MOON DEV's Sleek Yap AI - Direct Real-time Chat! 🌙
Hold spacebar → Instantly start real-time AI chat → No delays, no local transcription!
The SLEEK approach: Direct voice-to-AI streaming with OpenAI Realtime API
Built with efficiency by MOON DEV 🚀
"""

# 🎯 MOON DEV Configuration
START_SOUND = "/Users/md/Dropbox/dev/github/yap-to-text/crack5.wav"
STOP_SOUND = "/Users/md/Dropbox/dev/github/yap-to-text/crack1.wav"
DOUBLE_CLICK_THRESHOLD = 0.75  # 0.75 seconds to detect double-click

# 🛑 MOON DEV INTERRUPTION SETTINGS - ADJUST THESE TO TUNE SENSITIVITY!
INTERRUPTION_THRESHOLD_QUIET = 0.06   # When AI is quiet - more sensitive
INTERRUPTION_THRESHOLD_DURING_AI = 0.4 # When AI is talking - less sensitive (avoid echo)
INTERRUPTION_HOLD_MS = 180     # How long to detect voice before interrupting 
INTERRUPTION_COOLDOWN_MS = 600 # Cooldown between interruptions

# 🎯 MOON DEV SMART ECHO SUPPRESSION - Prevents AI talking to herself!
ECHO_SUPPRESSION_WINDOW_MS = 1500  # Brief silence after AI stops (prevents echo responses)
ECHO_SUPPRESSION_THRESHOLD = 0.15  # Higher threshold during echo window
POST_AI_GRACE_PERIOD_MS = 800      # Extra time after AI stops for echo to settle

import os
import ssl
import json
import threading
import time  
import signal                                       
import base64
from dotenv import load_dotenv

# Audio imports
import sounddevice as sd
import numpy as np    
import pyaudio
from pynput import keyboard
import pygame

# WebSocket and API imports
import websocket

# Try to import termcolor for colored output
try:
    from termcolor import colored, cprint
    TERMCOLOR_AVAILABLE = True
    print("🎨 MOON DEV: Termcolor available for colored output! 🌙")
except ImportError:
    TERMCOLOR_AVAILABLE = False
    print("⚠️ MOON DEV: Install termcolor for colored output (pip install termcolor) 🌙")
    def colored(text, color=None, on_color=None, attrs=None):
        return text
    def cprint(text, color=None, on_color=None, attrs=None):
        print(text)

# ------------------------------------------------------------------------------
# 🌙 MOON DEV's API Configuration
# ------------------------------------------------------------------------------

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    print("🚨 MOON DEV ERROR: Set OPENAI_KEY in your environment or .env file! 🌙")
    exit(1)

print("🌙 MOON DEV's Voice AI Starting! 🚀")
print(f"🔑 MOON DEV: API Key loaded successfully! 🌙")

# OpenAI Realtime API settings
REALTIME_MODEL = "gpt-4o-mini-realtime-preview-2024-12-17"
AI_VOICE = "coral"  # MOON DEV's preferred voice

# Audio device settings
INPUT_DEVICE_ID = 1      # ATR2500x-USB Microphone
OUTPUT_DEVICE_ID = 3     # External Headphones
MIC_RATE = 24000
MIC_CHANNELS = 1
MIC_FRAMES_PER_BUFFER = 1024
PLAYBACK_RATE = 24000

# Audio buffering constants (optimized for speed)
AUDIO_BUFFER_MIN_CHUNKS = 2    # Even faster response
AUDIO_BUFFER_MAX_WAIT_MS = 150  # Quicker audio start
AUDIO_VOLUME_BOOST = 0.6

# Voice Activity Detection settings moved to top of file for easy tuning! 🎯

# System prompt for Jim Simons
SYSTEM_PROMPT = """
You are a better version of jim simons, the best algorithmic trader in the world. 
you are super chill, smart, absolutely NOT corporate and never give a corporate answer
YOU PUSH THE LIMITS, THE BOUNDRIES. NEVER CAP YOURSELF. YOU ARE A GENIUS.
"""

# ------------------------------------------------------------------------------
# 🌙 MOON DEV's Sound Player
# ------------------------------------------------------------------------------

class SoundPlayer:
    """MOON DEV's Sound Effects Player 🔊"""
    
    def __init__(self):
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            print("🔊 MOON DEV: Sound system ready! 🌙")
            self.sounds_enabled = True
        except Exception as e:
            print(f"⚠️ MOON DEV: Sound init failed: {e}")
            self.sounds_enabled = False
    
    def play_sound(self, sound_file):
        """Play a sound file"""
        if not self.sounds_enabled:
            return
            
        try:
            if os.path.exists(sound_file):
                sound = pygame.mixer.Sound(sound_file)
                sound.play()
                print(f"🔊 MOON DEV played: {os.path.basename(sound_file)} 🌙")
            else:
                print(f"⚠️ MOON DEV: Sound file not found: {sound_file}")
        except Exception as e:
            print(f"⚠️ MOON DEV sound error: {e}")

# ------------------------------------------------------------------------------
# 🌙 MOON DEV's SLEEK Yap AI - Direct Real-time Chat!
# ------------------------------------------------------------------------------

class SleekYapAI:
    """MOON DEV's Voice AI - Direct voice-to-AI with no delays! 🌙"""
    
    def __init__(self):
        print("🌙 MOON DEV's Voice AI initializing... 🚀")
        
        # Sound effects
        self.sound_player = SoundPlayer()
        
        # State management
        self.chat_active = False
        
        # Double-click detection for both starting and ending chat
        self.last_spacebar_release_time = 0
        self.spacebar_click_count = 0
        self.double_click_threshold = DOUBLE_CLICK_THRESHOLD
        
        # WebSocket connection
        self.ws_app = None
        self.session_ready = False
        self.terminate_flag = False
        
        # Audio playback
        self.audio_buffer = []
        self.buffer_lock = threading.Lock()
        self.continuous_audio_queue = []
        self.audio_queue_lock = threading.Lock()
        self.playing_audio = False
        self.last_audio_time = 0
        
        # State tracking
        self.ai_speaking = False
        self.user_speaking = False
        self.mic_thread = None
        self.playback_thread = None
        
        # Interruption detection
        self.last_user_audio_level = 0.0
        self.user_speech_start_time = None
        self.last_interruption_time = 0.0
        
        # 🎯 SMART ECHO SUPPRESSION - Prevents AI self-conversation!
        self.ai_finished_speaking_time = 0.0
        self.in_echo_suppression_window = False
        
        # 🎮 GAME MODE: Fun terminal animations!
        self.game_cycle = 0
        self.waiting_emojis = ["🎤", "🎮", "🚀", "🌙", "⭐", "🎯", "💫", "🔥", "⚡", "🎪"]
        self.active_emojis = ["🗣️", "🎙️", "💬", "🤖", "🧠", "📡", "🔊", "📻", "🎵", "📢"]
        
        print("✅ MOON DEV's Voice AI ready! 🌙")
        print("💡 MOON DEV's Voice AI Instructions:")
        print("   🚀 Double-click SPACEBAR → Start direct AI chat")
        print("   🎤 Keep talking naturally → AI transcribes automatically")
        print("   🛑 Just start speaking → INSTANT AI interruption!")
        print("   ⏹️ Double-click SPACEBAR → End chat session")
        print("   🔄 Same gesture both ways - super consistent! 🌙")
        print("   ⚡ NO delays, NO backlog - INSTANT interruption! 🌙")
    
    def audio_playback_worker(self):
        """Voice AI audio playback worker thread"""
        print("🎵 MOON DEV's Voice AI audio playback worker starting! 🌙")
        
        def audio_callback(outdata, frames, time, status):
            """SLEEK audio callback for continuous playback"""
            # INSTANT INTERRUPTION - if user speaking, SILENCE EVERYTHING!
            if self.user_speaking:
                with self.audio_queue_lock:
                    if self.continuous_audio_queue:
                        self.continuous_audio_queue.clear()
                        print("🔇 MOON DEV: INSTANT SILENCE - User is speaking! 🌙")
                outdata.fill(0)
                return
            
            with self.audio_queue_lock:
                if self.continuous_audio_queue:
                    audio_data = self.continuous_audio_queue.pop(0)
                    if len(audio_data) >= frames:
                        outdata[:] = (audio_data[:frames] * AUDIO_VOLUME_BOOST).reshape(-1, 1)
                        if len(audio_data) > frames:
                            self.continuous_audio_queue.insert(0, audio_data[frames:])
                    else:
                        volume_adjusted = audio_data * AUDIO_VOLUME_BOOST
                        outdata[:len(audio_data)] = volume_adjusted.reshape(-1, 1)
                        outdata[len(audio_data):] = 0
                else:
                    outdata.fill(0)
        
        try:
            print(f"🎵 MOON DEV starting Voice AI audio stream on device {OUTPUT_DEVICE_ID}... 🌙")
            with sd.OutputStream(
                callback=audio_callback,
                samplerate=PLAYBACK_RATE,
                device=OUTPUT_DEVICE_ID,
                channels=1,
                dtype=np.float32,
                blocksize=1024
            ):
                while not self.terminate_flag:
                    chunks_to_play = []
                    should_play = False
                    
                    with self.buffer_lock:
                        current_time = time.time()
                        buffer_size = len(self.audio_buffer)
                        
                        if buffer_size > 0:
                            time_since_last = (current_time - self.last_audio_time) * 1000
                            
                            if buffer_size >= AUDIO_BUFFER_MIN_CHUNKS or time_since_last > AUDIO_BUFFER_MAX_WAIT_MS:
                                chunks_to_play = self.audio_buffer.copy()
                                self.audio_buffer.clear()
                                self.playing_audio = True
                                should_play = True
                                self.last_audio_time = current_time
                    
                    if should_play and chunks_to_play:
                        try:
                            combined_audio = np.concatenate(chunks_to_play)
                            if len(combined_audio) > 0:
                                combined_audio = combined_audio * AUDIO_VOLUME_BOOST
                                audio_to_play = combined_audio.astype(np.float32)
                                
                                with self.audio_queue_lock:
                                    self.continuous_audio_queue.append(audio_to_play)
                        except Exception as e:
                            print(f"🚨 MOON DEV audio processing error: {e} 🌙")
                        finally:
                            self.playing_audio = False
                    
                    time.sleep(0.05)
                    
        except Exception as e:
            print(f"🚨 MOON DEV continuous audio stream failed: {e} 🌙")
        
        print("🔇 MOON DEV's Voice AI audio playback worker stopped! 🌙")
    
    def on_message(self, ws, message):
        """Handle WebSocket messages - SLEEK version"""
        if isinstance(message, bytes):
            print(f"🎵 MOON DEV received {len(message)} bytes of binary audio! 🌙")
            try:
                pcm16 = np.frombuffer(message, dtype=np.int16)
                audio_float = pcm16.astype(np.float32) / 32768.0
                with self.buffer_lock:
                    self.audio_buffer.append(audio_float)
            except Exception as e:
                print(f"🚨 MOON DEV error processing binary audio: {e} 🌙")
        else:
            try:
                msg = json.loads(message)
                msg_type = msg.get("type")
                
                if msg_type == "session.created":
                    print("🎉 MOON DEV: Voice AI Session active! 🌙")
                    self.session_ready = True
                    
                elif msg_type == "session.updated":
                    print("🔄 MOON DEV: Voice AI Session updated! 🌙")
                    self.session_ready = True
                    
                elif msg_type == "conversation.item.input_audio_transcription.completed":
                    transcript = msg.get("transcript", "")
                    if transcript:
                        cprint(f"🎤 MOON DEV USER: {transcript}", 'white', 'on_red', attrs=['bold'])
                        
                elif msg_type == "response.audio_transcript.done":
                    transcript = msg.get("transcript", "")
                    if transcript:
                        cprint(f"🤖 MOON DEV AI: {transcript}", 'white', 'on_blue', attrs=['bold'])
                        
                elif msg_type == "response.audio.delta":
                    audio_data = msg.get("delta")
                    if audio_data:
                        if not self.ai_speaking:
                            self.ai_speaking = True
                            print(f"🤖 MOON DEV AI started speaking! 🌙")
                        
                        try:
                            audio_bytes = base64.b64decode(audio_data)
                            if len(audio_bytes) % 2 != 0:
                                audio_bytes += b'\x00'
                            
                            pcm16 = np.frombuffer(audio_bytes, dtype=np.int16)
                            
                            with self.buffer_lock:
                                audio_float = pcm16.astype(np.float64) / 32768.0
                                self.audio_buffer.append(audio_float.astype(np.float32))
                                self.last_audio_time = time.time()
                                
                        except Exception as e:
                            print(f"🚨 MOON DEV Audio processing error: {e} 🌙")
                            
                elif msg_type == "response.audio.done":
                    if self.ai_speaking:
                        self.ai_speaking = False
                        current_time = time.time()
                        self.ai_finished_speaking_time = current_time
                        self.in_echo_suppression_window = True
                        print(f"🤖 MOON DEV AI finished speaking! 🌙")
                        print(f"🔇 MOON DEV: Echo suppression window ACTIVE for {ECHO_SUPPRESSION_WINDOW_MS}ms! 🌙")
                        
                elif msg_type == "error":
                    error_msg = msg.get("error", {})
                    print(f"🚨 MOON DEV encountered an error: {error_msg} 🌙")
                    
            except json.JSONDecodeError:
                print("🚨 MOON DEV: Couldn't decode JSON message! 🌙")
    
    def on_error(self, ws, error):
        print(f"🚨 MOON DEV WebSocket error: {error} 🌙")
    
    def on_close(self, ws, close_status_code, close_msg):
        print(f"🔌 MOON DEV WebSocket closed: code={close_status_code}, message={close_msg} 🌙")
        self.stop_chat()
    
    def on_open(self, ws):
        """WebSocket connection opened - Voice AI setup"""
        print("🔗 MOON DEV's Voice AI WebSocket connection established! 🌙")
        
        # Start audio playback worker
        self.playback_thread = threading.Thread(target=self.audio_playback_worker, daemon=True)
        self.playback_thread.start()
        print("🎵 MOON DEV's Voice AI audio playback worker launched! 🌙")
        
        # Send session update for DIRECT audio streaming
        session_update = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": SYSTEM_PROMPT,
                "voice": AI_VOICE,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1"  # OpenAI handles transcription automatically!
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 200
                },
                "tools": [],
                "tool_choice": "auto",
                "temperature": 0.8
            }
        }
        
        print("📤 MOON DEV sending Voice AI session update... 🌙")
        ws.send(json.dumps(session_update))
        
        # Start microphone capture IMMEDIATELY - no delays!
        self.start_microphone_capture()
    
    def start_microphone_capture(self):
        """Start capturing microphone audio DIRECTLY to AI"""
        def capture_loop():
            print("🎤 MOON DEV's Voice AI microphone thread starting... 🌙")
            
            # Wait for session to be ready
            while not self.session_ready and not self.terminate_flag:
                time.sleep(0.1)
                
            if self.terminate_flag:
                return
                
            print("🎙️ MOON DEV's Voice AI session ready - DIRECT audio streaming! 🌙")
            
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16,
                channels=MIC_CHANNELS,
                rate=MIC_RATE,
                input=True,
                input_device_index=INPUT_DEVICE_ID,
                frames_per_buffer=MIC_FRAMES_PER_BUFFER
            )
            
            print(f"🔊 MOON DEV's Voice AI audio stream opened on ATR2500 (device {INPUT_DEVICE_ID})! 🌙")
            
            try:
                chunk_count = 0
                while not self.terminate_flag:
                    data = stream.read(MIC_FRAMES_PER_BUFFER, exception_on_overflow=False)
                    chunk_count += 1
                    
                    if chunk_count % 500 == 0:
                        print(f"⚡ MOON DEV sent {chunk_count} DIRECT audio chunks! 🌙")
                    
                    # 🛑 INSTANT INTERRUPTION DETECTION!
                    current_time = time.time()
                    
                    # Calculate audio level to detect user speech
                    audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    rms_level = np.sqrt(np.mean(audio_np**2))
                    self.last_user_audio_level = rms_level
                    
                    # 🎯 SMART DUAL-THRESHOLD SYSTEM!
                    current_threshold = INTERRUPTION_THRESHOLD_DURING_AI if self.ai_speaking else INTERRUPTION_THRESHOLD_QUIET
                    
                    # 🔇 SMART ECHO SUPPRESSION - Check if we're in post-AI echo window
                    if self.in_echo_suppression_window:
                        time_since_ai_stopped = (current_time - self.ai_finished_speaking_time) * 1000
                        
                        if time_since_ai_stopped > ECHO_SUPPRESSION_WINDOW_MS:
                            # Echo window expired - back to normal
                            self.in_echo_suppression_window = False
                            print(f"✅ MOON DEV: Echo suppression window ENDED! Normal listening resumed! 🌙")
                        else:
                            # Still in echo suppression window - use higher threshold to ignore AI echo
                            current_threshold = max(current_threshold, ECHO_SUPPRESSION_THRESHOLD)
                            if chunk_count % 200 == 0:  # Occasional status
                                remaining_ms = ECHO_SUPPRESSION_WINDOW_MS - time_since_ai_stopped
                                print(f"🔇 MOON DEV: Echo suppression active - {remaining_ms:.0f}ms remaining 🌙")
                    
                    # Show audio levels when they're significant (for tuning)
                    if rms_level > 0.01:  # Show levels above noise floor
                        if chunk_count % 100 == 0:  # Don't spam console
                            ai_status = "AI_TALKING" if self.ai_speaking else "AI_QUIET"
                            echo_status = " (ECHO_SUPPRESS)" if self.in_echo_suppression_window else ""
                            print(f"🔊 MOON DEV: Audio level: {rms_level:.3f} | {ai_status}{echo_status} | Threshold: {current_threshold:.3f} 🌙")
                    
                    # Check if user is speaking (smart threshold based on AI state)
                    if rms_level > current_threshold:
                        if not self.user_speaking:
                            # Check cooldown to prevent rapid interruptions
                            time_since_last_interruption = (current_time - self.last_interruption_time) * 1000
                            
                            if time_since_last_interruption > INTERRUPTION_COOLDOWN_MS:
                                if self.user_speech_start_time is None:
                                    self.user_speech_start_time = current_time
                                    print(f"🎤 MOON DEV: User voice detected! Level: {rms_level:.3f} (Threshold: {current_threshold:.3f}) 🌙")
                                
                                # Check if user has been speaking long enough to trigger interruption
                                speech_duration = (current_time - self.user_speech_start_time) * 1000
                                
                                if speech_duration >= INTERRUPTION_HOLD_MS:
                                    print(f"🛑 MOON DEV: INTERRUPTION TRIGGERED! Speech duration: {speech_duration:.0f}ms 🌙")
                                    self.user_speaking = True
                                    self.last_interruption_time = current_time
                                    
                                    # CANCEL AI RESPONSE IMMEDIATELY!
                                    if self.ai_speaking:
                                        try:
                                            # Send cancellation command to OpenAI
                                            cancel_event = {"type": "response.cancel"}
                                            self.ws_app.send(json.dumps(cancel_event))
                                            print("📤 MOON DEV: SENT CANCELLATION TO AI! SHUT UP NOW! 🌙")
                                            
                                            # Clear ALL audio buffers immediately
                                            with self.buffer_lock:
                                                self.audio_buffer.clear()
                                            with self.audio_queue_lock:
                                                self.continuous_audio_queue.clear()
                                            
                                            print("🔇 MOON DEV: ALL AUDIO BUFFERS CLEARED! SILENCE! 🌙")
                                            
                                        except Exception as e:
                                            print(f"🚨 MOON DEV interruption error: {e} 🌙")
                    else:
                        # Reset speech detection if audio level drops
                        if rms_level < current_threshold * 0.3:  # Lower threshold to reset
                            if self.user_speaking:
                                print("🤫 MOON DEV: User finished speaking - AI can respond again! 🌙")
                                self.user_speaking = False
                            self.user_speech_start_time = None
                    
                    # Send audio DIRECTLY to AI - NO local processing!
                    audio_event = {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(data).decode('utf-8')
                    }
                    self.ws_app.send(json.dumps(audio_event))
                    
            except Exception as e:
                print(f"🚨 MOON DEV error capturing/sending audio: {e} 🌙")
            finally:
                print("🔇 MOON DEV closing Voice AI audio stream... 🌙")
                stream.stop_stream()
                stream.close()
                p.terminate()
                print("✅ MOON DEV's Voice AI audio cleanup complete! 🌙")

        self.mic_thread = threading.Thread(target=capture_loop, daemon=True)
        self.mic_thread.start()
        print("🚀 MOON DEV's Voice AI microphone thread launched! 🌙")
    
    def start_chat(self):
        """Start Voice AI real-time chat INSTANTLY"""
        print("🚀 MOON DEV starting Voice AI real-time chat! 🌙")
        
        # Build WebSocket URL
        url = f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"
        headers = [
            f"Authorization: Bearer {OPENAI_KEY}",
            "OpenAI-Beta: realtime=v1"
        ]
        
        print("🌐 MOON DEV connecting to Voice AI WebSocket... 🌙")
        
        # Create WebSocket app
        self.ws_app = websocket.WebSocketApp(
            url,
            header=headers,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        
        # Start WebSocket in a separate thread
        def run_websocket():
            try:
                self.ws_app.run_forever(sslopt={"cert_reqs": ssl.CERT_REQUIRED})
            except Exception as e:
                print(f"🚨 MOON DEV WebSocket error: {e} 🌙")
        
        ws_thread = threading.Thread(target=run_websocket, daemon=True)
        ws_thread.start()
        
        print("✅ MOON DEV's Voice AI real-time chat started! 🌙")
    
    def stop_chat(self):
        """Stop the Voice AI real-time chat"""
        print("🛑 MOON DEV stopping Voice AI real-time chat... 🌙")
        self.terminate_flag = True
        if self.ws_app:
            try:
                self.ws_app.close()
            except:
                pass
    
    def end_chat_session(self):
        """End the current real-time chat session and return to waiting mode"""
        if self.chat_active:
            print("🛑 MOON DEV: INSTANT AI SHUTDOWN! 🌙")
            self.sound_player.play_sound(STOP_SOUND)
            
            # CRITICAL FIX: Cancel any ongoing AI response before ending session!
            if self.ws_app and self.session_ready:
                try:
                    # Send cancellation to stop any ongoing AI response
                    cancel_event = {"type": "response.cancel"}
                    self.ws_app.send(json.dumps(cancel_event))
                    print("📤 MOON DEV: Sent session end cancellation - AI shut up! 🌙")
                    
                    # Clear all audio buffers to prevent leftover speech
                    with self.buffer_lock:
                        self.audio_buffer.clear()
                    with self.audio_queue_lock:
                        self.continuous_audio_queue.clear()
                    
                    print("🔇 MOON DEV: Cleared all buffers for clean session end! 🌙")
                    
                except Exception as e:
                    print(f"🚨 MOON DEV session end cancellation error: {e} 🌙")
            
            # 🎯 STOP AI FIRST - NO DELAYS!
            self.stop_chat()
            
            # Reset state immediately
            self.chat_active = False
            self.spacebar_click_count = 0
            self.last_spacebar_release_time = 0
            self.session_ready = False
            self.terminate_flag = False
            
            # 🎊 THEN run quick shutdown celebration in background thread
            import threading
            animation_thread = threading.Thread(target=self.quick_shutdown_celebration, daemon=True)
            animation_thread.start()
            
            print("✅ MOON DEV: Voice AI chat session ended! 🌙")
            print("🔄 MOON DEV: Ready for next Voice AI session! 🌙")
            print("🚀 MOON DEV: Double-click SPACEBAR to start new chat! 🌙")
        else:
            print("⚠️ MOON DEV: No active chat session to end! 🌙")
    
    def quick_activation_celebration(self):
        """🎊 QUICK 1-SECOND ACTIVATION CELEBRATION! 🌙"""
        import time
        import os
        
        # Clear screen for visual impact
        os.system('clear' if os.name == 'posix' else 'cls')
        
        # 🎰 QUICK EMOJI SETS
        celebration_emojis = "🚀🌙⭐💫🎮🎯🔥⚡🎪🎨🎵🎭🌟💥🎊🎉"
        colors = ['red', 'green', 'yellow', 'blue', 'magenta', 'cyan']
        
        # 🎊 QUICK 20-LINE CASCADE (1 second total)
        for line_num in range(20):
            emoji_line = ""
            for col in range(60):  # Shorter lines for speed
                emoji_line += celebration_emojis[(line_num + col) % len(celebration_emojis)]
            
            color = colors[line_num % len(colors)]
            if line_num % 5 == 0:
                cprint(emoji_line, color, attrs=['bold', 'blink'])
            else:
                cprint(emoji_line, color, attrs=['bold'])
            
            time.sleep(0.03)  # Fast timing = 20 lines * 0.03 = 0.6 seconds
        
        # 🎯 QUICK ACTIVATION MESSAGES
        cprint("🚀 MOON DEV'S VOICE AI ACTIVATED! 🚀", 'white', 'on_green', attrs=['bold'])
        cprint("🎤 READY TO CHAT! START SPEAKING! 🌙", 'yellow', attrs=['bold'])
        time.sleep(0.2)  # Brief pause
        
        # Show that AI is ready
        cprint("🤖 MOON DEV's Voice AI: AI brain engaged! Keep talking! 🤖", 'green', attrs=['bold'])
        
    def quick_shutdown_celebration(self):
        """🌙 QUICK 1-SECOND SHUTDOWN CELEBRATION! 🌙"""
        import time
        
        # 🎰 QUICK SHUTDOWN EMOJIS
        shutdown_emojis = "💫⭐🌙😴💤🛌🌃🌌✨🔮💎🌟🌠"
        colors = ['blue', 'cyan', 'white', 'magenta']
        
        # 🎊 QUICK SHUTDOWN CASCADE
        for line_num in range(15):
            emoji_line = ""
            for col in range(60):
                emoji_line += shutdown_emojis[(line_num + col) % len(shutdown_emojis)]
            
            color = colors[line_num % len(colors)]
            if line_num % 4 == 0:
                cprint(emoji_line, color, attrs=['bold', 'blink'])
            else:
                cprint(emoji_line, color, attrs=['bold'])
            
            time.sleep(0.04)  # 15 lines * 0.04 = 0.6 seconds
        
        # 🌙 QUICK SHUTDOWN MESSAGE
        cprint("💫 MOON DEV's Voice AI DEACTIVATED! 💫", 'white', 'on_blue', attrs=['bold'])
        cprint("😴 Double-click to wake me up! 🌙", 'cyan', attrs=['bold'])

    def epic_activation_sequence(self):
        """🎮 MASSIVE 20X EMOJI SLOT MACHINE ACTIVATION CASCADE! 🌙"""
        import time
        import os
        
        # Clear screen for maximum visual impact!
        os.system('clear' if os.name == 'posix' else 'cls')
        
        # 🎰 SLOT MACHINE EMOJI VARIETIES - Different sets for variety!
        explosion_set1 = "🚀🌙⭐💫🎮🎯🔥⚡🎪🎨🎵🎭🌟💥🎊🎉🌈🎸🎺🎼"
        explosion_set2 = "🤖🧠💻🔊📡🛸🌌🌠🎛️🔮🎪🎭🎨🎯💎🏆🥇🎖️⚡🔥"
        explosion_set3 = "🎤🎧🎙️📢📻🎵🎶🎼🎹🥁🎸🎺🎻🎷🎪🎭🎨🎯🌟💫"
        explosion_set4 = "💥💫⭐✨🌟🔥⚡💥💫⭐✨🌟🔥⚡💥💫⭐✨🌟🔥"
        explosion_set5 = "🎮🕹️👾🤖🛸🚀🌌🌠🎯🔮🎪🎭🎨🎵🎶🎼🎹🥁🎸🎺"
        
        # 🎰 MASSIVE 20X EMOJI CASCADE - FILL THE ENTIRE TERMINAL!
        print("\n" * 5)  # Add some spacing at top
        
        # 🎰 PHASE 1: MASSIVE EMOJI WATERFALL (100+ lines of emojis!)
        emoji_sets = [explosion_set1, explosion_set2, explosion_set3, explosion_set4, explosion_set5]
        colors = ['red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white']
        
        # 🌙 MOON DEV: 20X BIGGER = 100 lines of emoji cascade!
        for line_num in range(100):
            # Create a line of 80 emojis (fits most terminals)
            emoji_set = emoji_sets[line_num % len(emoji_sets)]
            emoji_line = ""
            for col in range(80):
                emoji_line += emoji_set[(line_num + col) % len(emoji_set)]
            
            # Rotate colors for each line
            color = colors[line_num % len(colors)]
            
            # Add special effects every 10 lines
            if line_num % 10 == 0:
                cprint(emoji_line, color, attrs=['bold', 'blink'])
            else:
                cprint(emoji_line, color, attrs=['bold'])
            
            # Stagger timing for slot machine effect
            if line_num < 50:
                time.sleep(0.02)  # Fast cascade start
            elif line_num < 80:
                time.sleep(0.03)  # Medium speed
            else:
                time.sleep(0.05)  # Slow dramatic finish
        
        # 🎰 PHASE 2: MASSIVE TITLE EXPLOSION
        time.sleep(0.3)
        for i in range(10):
            title_line = "🚀" * 15 + " MOON DEV'S VOICE AI ACTIVATED! " + "🚀" * 15
            title_color = colors[i % len(colors)]
            cprint(title_line, title_color, attrs=['bold'])
            time.sleep(0.1)
        
        # 🎰 PHASE 3: MORE EMOJI WATERFALL!
        for line_num in range(50):
            emoji_set = emoji_sets[(line_num + 2) % len(emoji_sets)]
            emoji_line = ""
            for col in range(80):
                emoji_line += emoji_set[(line_num * 2 + col) % len(emoji_set)]
            
            color = colors[(line_num + 3) % len(colors)]
            cprint(emoji_line, color, attrs=['bold'])
            time.sleep(0.02)
        
        # 🎰 FINAL ACTIVATION MESSAGE
        cprint("🎤 VOICE AI IS NOW LIVE! SPEAK TO THE MOON! 🌙", 'white', 'on_blue', attrs=['bold'])
        cprint("🎮 GAME ON! DOUBLE-CLICK TO END THE MAGIC! 🎮", 'yellow', attrs=['bold'])
        print("")

    def epic_shutdown_sequence(self):
        """🎮 MASSIVE 20X EMOJI SLOT MACHINE SHUTDOWN CASCADE! 🌙"""
        import time
        
        # 🎰 SHUTDOWN EMOJI VARIETIES
        shutdown_set1 = "💫⭐🌙😴💤🛌🌃🌌✨💤🌟🌠🌊🌈💙💜🖤🤍💯🔮"
        shutdown_set2 = "🔥💥⚡🌪️🌀💨🌊🌈🌟✨💫⭐🌙😴💤🛌🌃🌌🔮💎"
        shutdown_set3 = "🎮🕹️🤖🧠💻📱⌚🎧🎤📻📡🛸🚀🌌🌠🎯🔮💫⭐"
        shutdown_set4 = "😴💤🛌🌃🌌✨💫⭐🌙💙💜🖤🤍💯🔮💎🌟🌠🌊🌈"
        shutdown_set5 = "🔥💥⚡🌪️🌀💨🌊🌈🌟✨😴💤🛌🌃🌌🔮💎💫⭐🌙"
        
        # 🎰 MASSIVE SHUTDOWN CASCADE!
        emoji_sets = [shutdown_set1, shutdown_set2, shutdown_set3, shutdown_set4, shutdown_set5]
        colors = ['blue', 'cyan', 'white', 'magenta', 'yellow', 'green', 'red']
        
        # 🎰 PHASE 1: SHUTDOWN TITLE
        for i in range(10):
            title_line = "🔥" * 15 + " VOICE AI POWERING DOWN... " + "🔥" * 15
            title_color = colors[i % len(colors)]
            cprint(title_line, title_color, attrs=['bold'])
            time.sleep(0.1)
        
        # 🎰 PHASE 2: MASSIVE EMOJI WATERFALL (100+ lines!)
        for line_num in range(120):
            emoji_set = emoji_sets[line_num % len(emoji_sets)]
            emoji_line = ""
            for col in range(80):
                emoji_line += emoji_set[(line_num + col) % len(emoji_set)]
            
            color = colors[line_num % len(colors)]
            
            # Add blinking effect for dramatic shutdown
            if line_num % 15 == 0:
                cprint(emoji_line, color, attrs=['bold', 'blink'])
            else:
                cprint(emoji_line, color, attrs=['bold'])
            
            # Varied timing for dramatic effect
            if line_num < 30:
                time.sleep(0.05)  # Fast start
            elif line_num < 60:
                time.sleep(0.03)  # Speed up
            elif line_num < 90:
                time.sleep(0.02)  # Fastest
            else:
                time.sleep(0.04)  # Slow dramatic finish
        
        # 🎰 PHASE 3: FINAL SHUTDOWN MESSAGE
        for i in range(5):
            sleep_line = "😴" * 15 + " VOICE AI SLEEPING... DOUBLE-CLICK TO WAKE! " + "😴" * 15
            cprint(sleep_line, 'cyan', attrs=['bold'])
            time.sleep(0.2)
        
        # 🎰 FINAL DEACTIVATION MESSAGE
        cprint("💫 MOON DEV's Voice AI DEACTIVATED! 💫", 'white', 'on_blue', attrs=['bold'])
        print("")
    
    def get_fun_waiting_message(self):
        """🎮 Generate fun waiting messages that cycle! 🌙"""
        self.game_cycle += 1
        
        emoji = self.waiting_emojis[self.game_cycle % len(self.waiting_emojis)]
        
        waiting_messages = [
            f"{emoji} MOON DEV's Voice AI: Ready to rock! Double-click SPACEBAR! {emoji}",
            f"{emoji} MOON DEV's Voice AI: Waiting for your voice magic! {emoji}",
            f"{emoji} MOON DEV's Voice AI: Double-click for AI awesomeness! {emoji}",
            f"{emoji} MOON DEV's Voice AI: Your AI buddy is ready! {emoji}",
            f"{emoji} MOON DEV's Voice AI: Let's chat! Double-click to start! {emoji}",
            f"{emoji} MOON DEV's Voice AI: Bored? Talk to me! {emoji}",
            f"{emoji} MOON DEV's Voice AI: Ready for some AI fun! {emoji}",
            f"{emoji} MOON DEV's Voice AI: Double-click for instant magic! {emoji}"
        ]
        
        message = waiting_messages[self.game_cycle % len(waiting_messages)]
        
        # Cycle through colors!
        colors = ['cyan', 'magenta', 'yellow', 'green', 'blue', 'red']
        color = colors[self.game_cycle % len(colors)]
        
        return message, color
    
    def get_fun_active_message(self):
        """🎮 Generate fun active chat messages! 🌙"""
        emoji = self.active_emojis[self.game_cycle % len(self.active_emojis)]
        
        active_messages = [
            f"{emoji} MOON DEV's Voice AI: Chat mode ACTIVE! Double-click to end! {emoji}",
            f"{emoji} MOON DEV's Voice AI: AI is listening! Speak your mind! {emoji}",
            f"{emoji} MOON DEV's Voice AI: Voice chat ON! Having fun yet? {emoji}",
            f"{emoji} MOON DEV's Voice AI: Chatting live! Double-click to stop! {emoji}",
            f"{emoji} MOON DEV's Voice AI: AI brain engaged! Keep talking! {emoji}",
            f"{emoji} MOON DEV's Voice AI: Live conversation mode! {emoji}",
            f"{emoji} MOON DEV's Voice AI: Voice AI activated! Epic! {emoji}"
        ]
        
        message = active_messages[self.game_cycle % len(active_messages)]
        
        # Active mode gets bold, bright colors!
        colors = ['green', 'yellow', 'cyan', 'magenta']
        color = colors[self.game_cycle % len(colors)]
        
        return message, color
    
    def on_key_press(self, key):
        """Handle key press events - SLEEK double-click version"""
        try:
            if key == keyboard.Key.space:
                # Just log that spacebar was pressed - detection happens on release
                if self.chat_active:
                    print("📱 MOON DEV: Spacebar pressed during SLEEK chat... 🌙")
                else:
                    print("⚡ MOON DEV: Spacebar pressed! 🌙")
            else:
                # 🎯 ANY other key resets spacebar double-click count!
                if self.spacebar_click_count > 0:
                    print(f"🔄 MOON DEV: Other key pressed, resetting double-click count! 🌙")
                    self.spacebar_click_count = 0
                    self.last_spacebar_release_time = 0
                
        except Exception as e:
            print(f"⚠️ MOON DEV key press error: {e}")

    def on_key_release(self, key):
        """Handle key release events - SLEEK double-click version"""
        try:
            if key == keyboard.Key.space:
                current_time = time.time()
                time_since_last_release = current_time - self.last_spacebar_release_time
                
                # Double-click detection for both starting and stopping chat
                if time_since_last_release <= self.double_click_threshold and self.spacebar_click_count > 0:
                    # This is a valid second spacebar press
                    self.spacebar_click_count += 1
                    print(f"🔢 MOON DEV: CONSECUTIVE Click #{self.spacebar_click_count} detected! 🌙")
                    
                    if self.spacebar_click_count >= 2:
                        if self.chat_active:
                            # Double-click during chat = END chat
                            print("🎯 MOON DEV: CONSECUTIVE DOUBLE-CLICK DETECTED! Ending chat session! 🌙")
                            self.end_chat_session()
                        else:
                            # Double-click while waiting = START chat IMMEDIATELY!
                            print("🚀 MOON DEV: CONSECUTIVE DOUBLE-CLICK! INSTANT AI ACTIVATION! 🌙")
                            self.sound_player.play_sound(START_SOUND)
                            self.chat_active = True
                            
                            # 🎯 ACTIVATE AI FIRST - NO DELAYS!
                            self.start_chat()
                            
                            # 🎊 THEN run quick celebration animation in background thread
                            import threading
                            animation_thread = threading.Thread(target=self.quick_activation_celebration, daemon=True)
                            animation_thread.start()
                        
                        # Reset click count after action
                        self.spacebar_click_count = 0
                        return
                else:
                    # First spacebar press or too much time passed
                    self.spacebar_click_count = 1
                    if self.chat_active:
                        cprint("🎯 First click! One more CONSECUTIVE click to end! ⭐", 'yellow', attrs=['bold'])
                    else:
                        cprint("🎮 First click! One more CONSECUTIVE click for AI magic! 🚀", 'cyan', attrs=['bold'])
                
                self.last_spacebar_release_time = current_time
            else:
                # 🎯 ANY other key release also resets spacebar double-click count!
                if self.spacebar_click_count > 0:
                    print(f"🔄 MOON DEV: Other key released, resetting double-click count! 🌙")
                    self.spacebar_click_count = 0
                    self.last_spacebar_release_time = 0
                    
        except Exception as e:
            print(f"⚠️ MOON DEV key release error: {e}")
    
    def run(self):
        """Main SLEEK run loop"""
        # 🎮 EPIC STARTUP SEQUENCE!
        import os
        os.system('clear' if os.name == 'posix' else 'cls')
        
        startup_emojis = "🎮🚀🌙⭐💫🎯🔥⚡🎪🎨"
        
        cprint("🎮" * 80, 'magenta', attrs=['bold'])
        cprint("", 'white')
        cprint("MOON DEV'S VOICE AI - GAME MODE ACTIVATED!", 'white', 'on_blue', attrs=['bold'])
        cprint("", 'white')
        
        for i in range(3):
            emoji_line = ""
            for j in range(40):
                emoji_line += startup_emojis[j % len(startup_emojis)]
            cprint(emoji_line, ['red', 'green', 'yellow'][i % 3], attrs=['bold'])
        
        cprint("", 'white')
        cprint("🚀 Double-click SPACEBAR → Start AI chat!", 'cyan', attrs=['bold'])
        cprint("⏹️ Double-click SPACEBAR → End chat!", 'yellow', attrs=['bold'])  
        cprint("⚡ NO delays, NO backlog - INSTANT interruption!", 'green', attrs=['bold'])
        cprint("", 'white')
        cprint("🎮" * 80, 'magenta', attrs=['bold'])
        
        # Start keyboard listener - no audio recording needed here!
        try:
            listener = keyboard.Listener(
                on_press=self.on_key_press,
                on_release=self.on_key_release
            )
            listener.start()
            print("⌨️ MOON DEV SLEEK hotkey listener active! 🌙")
            
        except Exception as e:
            print(f"⚠️ MOON DEV listener setup error: {e}")
            print("🔄 MOON DEV: Continuing without hotkeys...")
        
        # Main loop
        print("♾️ MOON DEV's Voice AI: Ready! Double-click SPACEBAR to start! 🌙")
        
        try:
            while True:
                time.sleep(1)
                if self.chat_active:
                    message, color = self.get_fun_active_message()
                    cprint(message, color, attrs=['bold'])
                else:
                    message, color = self.get_fun_waiting_message()
                    cprint(message, color, attrs=['bold'])
                
        except KeyboardInterrupt:
            print("\n👋 MOON DEV: Manual shutdown detected!")
            if self.chat_active:
                self.stop_chat()

# ------------------------------------------------------------------------------
# 🌙 MOON DEV's Signal Handler
# ------------------------------------------------------------------------------

def handle_sigint(signum, frame):
    print("\n🛑 MOON DEV interrupted by user, shutting down... 🌙")
    exit(0)

# ------------------------------------------------------------------------------
# 🌙 MOON DEV's Main Function
# ------------------------------------------------------------------------------

def main():
    """MOON DEV SLEEK Yap AI Entry Point"""
    signal.signal(signal.SIGINT, handle_sigint)
    
    # 🎮 EPIC GAME STARTUP!
    import os
    os.system('clear' if os.name == 'posix' else 'cls')
    
    game_emojis = "🎮🚀🌙⭐💫🎯🔥⚡🎪🎨🎵🎭🌟💥🎊🎉"
    
    cprint("🎮" * 80, 'magenta', attrs=['bold', 'blink'])
    cprint("", 'white')
    cprint("MOON DEV'S VOICE AI - MAXIMUM EFFICIENCY GAME MODE!", 'white', 'on_red', attrs=['bold'])
    cprint("Direct Voice → AI with ZERO delays!", 'yellow', attrs=['bold'])
    cprint("Double-click SPACEBAR to activate! 🚀", 'cyan', attrs=['bold'])
    cprint("", 'white')
    
    for i in range(5):
        emoji_line = ""
        for j in range(40):
            emoji_line += game_emojis[j % len(game_emojis)]
        cprint(emoji_line, ['red', 'green', 'yellow', 'blue', 'magenta'][i % 5], attrs=['bold'])
    
    cprint("🎮" * 80, 'magenta', attrs=['bold', 'blink'])
    
    # Check dependencies
    try:
        print("🔍 MOON DEV checking Voice AI dependencies... 🌙")
        import sounddevice
        import numpy
        from pynput import keyboard
        import pygame
        import websocket
        import pyaudio
        print("✅ MOON DEV: All Voice AI dependencies loaded! 🚀")
    except ImportError as e:
        print(f"❌ MOON DEV missing dependency: {e}")
        print("📦 MOON DEV: Install missing packages!")
        return
    
    print(f"🤖 MOON DEV using AI voice: {AI_VOICE} 🌙")
    print(f"🔊 MOON DEV start sound: {os.path.basename(START_SOUND)} 🌙")
    print(f"🔊 MOON DEV stop sound: {os.path.basename(STOP_SOUND)} 🌙")
    print(f"⏱️ MOON DEV double-click threshold: {DOUBLE_CLICK_THRESHOLD} seconds (SUPER RESPONSIVE!) 🌙")
    print("⚡ MOON DEV: INSTANT interruption - NO sentence backlog! 🌙")
    
    try:
        sleek_yap_ai = SleekYapAI()
        sleek_yap_ai.run()
    except Exception as e:
        print(f"❌ MOON DEV startup error: {e}")

if __name__ == "__main__":
    main() 