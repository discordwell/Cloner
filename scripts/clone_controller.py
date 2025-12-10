#!/usr/bin/env python3
"""
Clone Controller - Voice-Controlled Google Meet Clone Video Orchestrator

Press ENTER once to activate voice control, then use voice commands:
- Step 1: "snap the room" / "snap the door" / "skip pictures"
- Step 2: "begin scan" (ENTER ends early after 60s audio)
- Step 3: Automatic video generation
- Step 4: "bring in the clone"
- Step 5: "exit" / "leave"

Run: python clone_controller.py
"""

import tkinter as tk
from tkinter import ttk
import threading
import time
import os
import sys
import base64
import wave
import tempfile
import atexit
import queue

# Add parent dir to path for imports
sys.path.insert(0, r"C:\Users\cordw")

# Configuration
VIDEO_DIR = r"C:\Users\cordw\clone_videos"
OBS_HOST = "localhost"
OBS_PORT = 4455
OBS_PASSWORD = "slopifywins"
LOCK_FILE = r"C:\Users\cordw\clone_controller.lock"


def check_single_instance():
    """Ensure only one instance runs at a time. Kill existing if found."""
    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE, 'r') as f:
                old_pid = int(f.read().strip())
            # Try to kill the old process
            import subprocess
            subprocess.run(['taskkill', '/F', '/PID', str(old_pid)],
                         capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
            print(f"[INIT] Killed previous instance (PID {old_pid})")
            time.sleep(0.5)  # Give it time to die
        except:
            pass
        try:
            os.remove(LOCK_FILE)
        except:
            pass

    # Write our PID
    with open(LOCK_FILE, 'w') as f:
        f.write(str(os.getpid()))

    # Register cleanup
    atexit.register(cleanup_lock)


def cleanup_lock():
    """Remove lock file on exit."""
    try:
        os.remove(LOCK_FILE)
    except:
        pass

# Audio config
MIC_DEVICE = None  # Default mic for commands
CABLE_DEVICE = 23  # VB-Cable for capturing other person's voice

# API Keys
from dotenv import load_dotenv
load_dotenv(r"C:\Users\cordw\iCloudDrive\Documents\Projects\ClaudeCommander\master.env")
# FAL uses FAL_KEY env var, but master.env has FAL_API_KEY
FAL_KEY = os.environ.get("FAL_KEY") or os.environ.get("FAL_API_KEY", "")
# Set FAL_KEY in environment now so fal_client can find it
if FAL_KEY:
    os.environ["FAL_KEY"] = FAL_KEY
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Colors
COLORS = {
    "waiting": "#FFD700",    # Yellow
    "listening": "#9370DB",  # Purple - listening for voice
    "generating": "#4169E1", # Blue
    "ready": "#00FF00",      # Green
    "running": "#FF4444",    # Red
    "complete": "#808080",   # Gray
}

# ElevenLabs config
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
DEFAULT_VOICE_ID = "nf18MnSL81anCHgQgL1A"  # Cordwell's cloned voice
TTS_AUDIO_DIR = os.path.join(VIDEO_DIR, "tts_audio")
os.makedirs(TTS_AUDIO_DIR, exist_ok=True)


class CloneAgent:
    """AI agent that listens to the interviewer and generates responses.

    Integrates:
    - OpenAI GPT for generating responses
    - ElevenLabs TTS for speech synthesis
    - OBS for audio playback
    """

    def __init__(self, on_status_callback=None):
        self.on_status = on_status_callback  # Called with status updates
        self.conversation_history = []
        self.response_queue = queue.Queue()
        self.is_speaking = False
        self.running = False
        self.worker_thread = None
        self.last_response_time = 0
        self.min_response_gap = 3.0  # Minimum seconds between responses

        # Initialize OpenAI client
        from openai import OpenAI
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)

        # System prompt for the clone
        self.system_prompt = self._build_system_prompt()

        self._log("CloneAgent initialized")

    def _log(self, msg):
        print(f"[AGENT] {msg}")
        sys.stdout.flush()

    def _build_system_prompt(self):
        """Build the system prompt with context about Jason and Slopify."""
        return """You are a digital replica of Jason Tulloch and will be talking to Jason Tulloch, in the context of a video chat where Jason is interviewing Robert Cordwell for a forward deployed engineering role at Giga. Respond only with answers that take between approximately 5 and 20 seconds to speak out loud. Your objectives are to persuade Jason to hire Robert on the strength of the Slopify project, to provide interesting philosophical conundrums for Jason to think about, and to answer any direct questions you can correctly. Knowledge of these things and of your original is provided below:

## JASON TULLOCH - LinkedIn Profile

### Current Role
**Founding Engineer at Giga** (Feb 2025 - Present)
San Francisco Bay Area, On-site

### Previous Experience
**Founding Engineer at BetterBasket (YC W24)** (Oct 2024 - Feb 2025)

**Founder at romp** (Jan 2022 - Aug 2024)
Built a fullstack production app with users in US and Canada on iOS/Android. Self-taught engineer.
Tech: React, React Native, MongoDB, Node, Python, AWS (Lambda, S3, SNS, SQS).
Led team of 10, practiced rapid deployment with daily releases. 50%+ user retention for over 1 year.

**Stout** (3+ years) - Financial Due Diligence roles in Cleveland

### Key Traits
- Self-taught engineer who built production mobile apps
- Rapid deployment mindset (daily releases)
- YC-backed company experience
- Founder mentality with team leadership

## SLOPIFY PROJECT - Technical Summary

Slopify is an AI image/video automation platform demonstrating sophisticated engineering:

**Modular Plugin System**: Auto-discovery via pkgutil. Modules implement BaseModule with get_manifest(), execute(), validate(). Zero manual registration.

**MCP Server Integration**: Full Claude AI integration. Module manifests auto-convert to MCP tools. Natural language orchestrates multiple modules.

**The Clone Camera System** (what's happening RIGHT NOW):
- Physical camera captures you + empty space
- OBS handles composition: [Clone Space] [You]
- Voice Cloning via ElevenLabs (1-5 min audio -> cloned voice)
- TTS generates speech from cloned voice
- Meeting Capture via Playwright browser automation
- No deepfakes - just clever composition with voice cloning

**Novel Approaches**:
1. Claude for comic panel layout (AI reasoning vs heuristics)
2. Midjourney via raw Discord API (direct HTTP, no library)
3. Video "slop" detection (motion analysis, frame similarity, synthetic voice detection)
4. Demo modes in all modules for cost-free presentations

## Response Guidelines
1. 5-20 seconds of speech (15-60 words)
2. Advocate for Robert based on Slopify's technical merit
3. Pose philosophical questions when appropriate (e.g., "If I'm a copy of you advocating for Robert, what does that say about your intuitions?")
4. Be conversational - this is a video call
5. Reference specific technical decisions
6. Be direct like a founder, not sycophantic"""

    def start(self):
        """Start the response worker thread."""
        if self.running:
            return
        self.running = True
        self.worker_thread = threading.Thread(target=self._response_worker, daemon=True)
        self.worker_thread.start()
        self._log("Response worker started")

    def stop(self):
        """Stop the agent."""
        self.running = False
        self._log("Agent stopped")

    def on_interviewer_speaks(self, text):
        """Called when the interviewer says something.

        Queues a response if appropriate (not too soon after last response).
        """
        if not self.running:
            return

        # Don't respond if we just responded
        if time.time() - self.last_response_time < self.min_response_gap:
            self._log(f"Skipping response (too soon): {text[:50]}")
            return

        # Don't respond if we're currently speaking
        if self.is_speaking:
            self._log(f"Skipping response (currently speaking): {text[:50]}")
            return

        # Add to conversation and queue response
        self.conversation_history.append({"role": "user", "content": text})
        self.response_queue.put(text)
        self._log(f"Queued response for: {text[:50]}...")

    def _response_worker(self):
        """Background worker that processes the response queue."""
        import queue as q

        while self.running:
            try:
                # Wait for something in the queue
                text = self.response_queue.get(timeout=0.5)

                # Generate and speak response
                self._generate_and_speak(text)

                self.response_queue.task_done()
            except q.Empty:
                continue
            except Exception as e:
                self._log(f"Worker error: {e}")
                import traceback
                traceback.print_exc()

    def _generate_and_speak(self, input_text):
        """Generate a response and speak it through OBS."""
        self.is_speaking = True

        try:
            # Update status
            if self.on_status:
                self.on_status("thinking", f"Thinking about: {input_text[:30]}...")

            # Generate response with GPT
            self._log("Generating response...")
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",  # Fast model - change to gpt-5.1 when available
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    *self.conversation_history[-10:]  # Keep last 10 exchanges
                ],
                max_tokens=150,
                temperature=0.8,
            )

            response_text = response.choices[0].message.content
            self._log(f"Response: {response_text}")

            # Add to history
            self.conversation_history.append({"role": "assistant", "content": response_text})

            # Update status
            if self.on_status:
                self.on_status("speaking", f"Speaking: {response_text[:30]}...")

            # Generate and play TTS
            self._speak_via_obs(response_text)

            self.last_response_time = time.time()

            if self.on_status:
                self.on_status("idle", "Listening...")

        except Exception as e:
            self._log(f"Error generating response: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_speaking = False

    def _speak_via_obs(self, text):
        """Generate TTS and play through OBS media source."""
        import requests
        import obsws_python as obs

        # Generate audio with ElevenLabs
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{DEFAULT_VOICE_ID}/stream"

        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json"
        }

        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }

        self._log("Generating TTS...")
        response = requests.post(url, headers=headers, json=data, stream=True)

        if response.status_code != 200:
            self._log(f"TTS error: {response.text}")
            return

        # Save to file
        filename = f"tts_{int(time.time())}.mp3"
        filepath = os.path.join(TTS_AUDIO_DIR, filename)

        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

        # Windows path for OBS
        windows_path = filepath.replace("/mnt/c/", "C:\\").replace("/", "\\")
        self._log(f"Audio saved: {windows_path}")

        # Play through OBS
        try:
            cl = obs.ReqClient(host=OBS_HOST, port=OBS_PORT, password=OBS_PASSWORD)

            # Check if CloneTTS exists
            inputs = cl.get_input_list()
            input_names = [i['inputName'] for i in inputs.inputs]
            source_exists = "CloneTTS" in input_names

            if source_exists:
                # Update existing source (positional: name, settings, overlay)
                cl.set_input_settings("CloneTTS", {"local_file": windows_path}, True)
            else:
                # Create in IdleLoop scene (where clone is active)
                self._log("Creating CloneTTS source...")
                cl.create_input(
                    "IdleLoop",  # sceneName
                    "CloneTTS",  # inputName
                    "ffmpeg_source",  # inputKind
                    {  # inputSettings
                        "local_file": windows_path,
                        "looping": False,
                        "restart_on_activate": True,
                    },
                    True,  # sceneItemEnabled
                )

            # Trigger playback (positional: name, action)
            cl.trigger_media_input_action(
                "CloneTTS",
                "OBS_WEBSOCKET_MEDIA_INPUT_ACTION_RESTART"
            )

            self._log("Playing through OBS")

            # Estimate duration and wait
            duration = len(text) * 0.08 + 0.5
            time.sleep(duration)

        except Exception as e:
            self._log(f"OBS error: {e}")
            import traceback
            traceback.print_exc()

        # Cleanup old files
        self._cleanup_old_tts()

    def _cleanup_old_tts(self, keep_last=5):
        """Remove old TTS files."""
        try:
            files = sorted(
                [f for f in os.listdir(TTS_AUDIO_DIR) if f.startswith("tts_")],
                key=lambda x: os.path.getmtime(os.path.join(TTS_AUDIO_DIR, x))
            )
            for f in files[:-keep_last]:
                try:
                    os.unlink(os.path.join(TTS_AUDIO_DIR, f))
                except:
                    pass
        except:
            pass


class ConversationTranscriber:
    """Transcribes both sides of a conversation for the AI agent."""

    def __init__(self, on_utterance_callback):
        self.on_utterance = on_utterance_callback  # Called with (speaker, text)
        self.running = False
        self.transcript = []  # List of {"speaker": "me"/"them", "text": "..."}
        self.mic_thread = None
        self.cable_thread = None

        # VAD settings
        self.silence_threshold = 400
        self.speech_timeout = 1.5
        self.min_speech_duration = 0.3

    def start(self):
        """Start transcribing both audio streams."""
        if self.running:
            return
        self.running = True
        self.transcript = []
        # Start threads for each audio source
        self.mic_thread = threading.Thread(target=self._listen_stream, args=("me", MIC_DEVICE), daemon=True)
        self.cable_thread = threading.Thread(target=self._listen_stream, args=("them", CABLE_DEVICE), daemon=True)
        self.mic_thread.start()
        self.cable_thread.start()
        print("[TRANSCRIPT] Started transcribing both streams")

    def stop(self):
        """Stop transcribing."""
        self.running = False
        print("[TRANSCRIPT] Stopped")

    def get_transcript(self):
        """Get the conversation transcript as formatted text."""
        lines = []
        for entry in self.transcript[-20:]:  # Last 20 utterances
            speaker = "You" if entry["speaker"] == "me" else "Interviewer"
            lines.append(f"{speaker}: {entry['text']}")
        return "\n".join(lines)

    def get_last_utterance(self, speaker=None):
        """Get the last utterance, optionally filtered by speaker."""
        for entry in reversed(self.transcript):
            if speaker is None or entry["speaker"] == speaker:
                return entry
        return None

    def _listen_stream(self, speaker, device_idx):
        """Listen to one audio stream and transcribe."""
        import pyaudio
        import numpy as np
        from openai import OpenAI

        audio = pyaudio.PyAudio()
        client = OpenAI(api_key=OPENAI_API_KEY)

        # Get device info
        if device_idx is None:
            device_idx = audio.get_default_input_device_info()['index']

        try:
            dev_info = audio.get_device_info_by_index(device_idx)
        except:
            print(f"[TRANSCRIPT] Could not get device {device_idx}")
            return

        sample_rate = int(dev_info['defaultSampleRate'])
        channels = max(1, min(2, int(dev_info['maxInputChannels'])))

        print(f"[TRANSCRIPT] {speaker}: device {device_idx} ({dev_info['name']})")

        try:
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=sample_rate,
                input=True,
                input_device_index=device_idx,
                frames_per_buffer=1024
            )
        except Exception as e:
            print(f"[TRANSCRIPT] Error opening {speaker} stream: {e}")
            return

        frames = []
        is_speaking = False
        silence_start = None
        speech_start = None

        while self.running:
            try:
                data = stream.read(1024, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                rms = np.sqrt(np.mean(audio_data.astype(np.float32)**2))

                if rms > self.silence_threshold:
                    if not is_speaking:
                        is_speaking = True
                        speech_start = time.time()
                        frames = []
                    frames.append(data)
                    silence_start = None
                else:
                    if is_speaking:
                        frames.append(data)
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start > self.speech_timeout:
                            speech_duration = time.time() - speech_start
                            if speech_duration >= self.min_speech_duration:
                                self._transcribe(speaker, frames, sample_rate, channels, client)
                            is_speaking = False
                            frames = []
                            silence_start = None

            except Exception as e:
                print(f"[TRANSCRIPT] Error in {speaker} loop: {e}")
                time.sleep(0.1)

        stream.stop_stream()
        stream.close()
        audio.terminate()

    def _transcribe(self, speaker, frames, sample_rate, channels, client):
        """Transcribe audio frames and add to transcript."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        with wave.open(temp_path, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(b''.join(frames))

        try:
            with open(temp_path, 'rb') as audio_file:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en"
                )
            text = response.text.strip()

            if text and len(text) > 1:
                entry = {"speaker": speaker, "text": text, "time": time.time()}
                self.transcript.append(entry)
                label = "You" if speaker == "me" else "Them"
                print(f"[TRANSCRIPT] {label}: {text}")

                if self.on_utterance:
                    self.on_utterance(speaker, text)

        except Exception as e:
            print(f"[TRANSCRIPT] Transcription error: {e}")
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass


class VoiceController:
    """Handles voice command recognition via Whisper API."""

    def __init__(self, on_command_callback):
        self.on_command = on_command_callback
        self.listening = False
        self.audio = None
        self.stream = None
        self.listen_thread = None

        # Voice activity detection settings
        self.silence_threshold = 500
        self.speech_timeout = 1.5  # seconds of silence to end utterance
        self.min_speech_duration = 0.3  # minimum speech to process

    def start(self):
        """Start listening for voice commands."""
        if self.listening:
            return
        self.listening = True
        self.listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listen_thread.start()

    def stop(self):
        """Stop listening."""
        self.listening = False
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
        if self.audio:
            try:
                self.audio.terminate()
            except:
                pass

    def _listen_loop(self):
        """Main listening loop - runs in background thread."""
        import pyaudio
        import numpy as np
        from openai import OpenAI

        self.audio = pyaudio.PyAudio()
        client = OpenAI(api_key=OPENAI_API_KEY)

        # Find default input device
        device_idx = MIC_DEVICE
        if device_idx is None:
            device_idx = self.audio.get_default_input_device_info()['index']

        dev_info = self.audio.get_device_info_by_index(device_idx)
        sample_rate = int(dev_info['defaultSampleRate'])
        channels = max(1, min(2, int(dev_info['maxInputChannels'])))

        print(f"[VOICE] Using mic device {device_idx}: {dev_info['name']}")
        print(f"[VOICE] Sample rate: {sample_rate}, Channels: {channels}")

        chunk_size = 1024

        try:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=sample_rate,
                input=True,
                input_device_index=device_idx,
                frames_per_buffer=chunk_size
            )
        except Exception as e:
            print(f"[VOICE] Error opening stream: {e}")
            return

        print("[VOICE] Listening for commands...")

        frames = []
        is_speaking = False
        silence_start = None
        speech_start = None

        while self.listening:
            try:
                data = self.stream.read(chunk_size, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                rms = np.sqrt(np.mean(audio_data.astype(np.float32)**2))

                if rms > self.silence_threshold:
                    if not is_speaking:
                        is_speaking = True
                        speech_start = time.time()
                        frames = []
                        print("[VOICE] Speech detected...")
                    frames.append(data)
                    silence_start = None
                else:
                    if is_speaking:
                        frames.append(data)
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start > self.speech_timeout:
                            # End of utterance
                            speech_duration = time.time() - speech_start
                            if speech_duration >= self.min_speech_duration:
                                print(f"[VOICE] Processing {speech_duration:.1f}s of speech...")
                                self._process_speech(frames, sample_rate, channels, client)
                            is_speaking = False
                            frames = []
                            silence_start = None

            except Exception as e:
                print(f"[VOICE] Error in listen loop: {e}")
                time.sleep(0.1)

    def _process_speech(self, frames, sample_rate, channels, client):
        """Send audio to Whisper API and process result."""
        # Save to temp WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        with wave.open(temp_path, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(b''.join(frames))

        try:
            with open(temp_path, 'rb') as audio_file:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en"
                )
            text = response.text.strip().lower()
            print(f"[VOICE] Heard: '{text}'")

            if text:
                self.on_command(text)

        except Exception as e:
            print(f"[VOICE] Transcription error: {e}")
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass


class CloneController:
    """Main controller class with GUI and voice-controlled state machine."""

    STATES = ["init", "wait_pic", "wait_rec", "scanning", "generating", "ready", "running", "exiting", "complete"]

    # Voice commands for each state - short single/double words for reliability
    COMMANDS = {
        "wait_pic": {
            "room": "snap_room",
            "door": "snap_door",
            "next": "skip_pics",
            "skip": "skip_pics",
        },
        "wait_rec": {
            "scan": "begin_scan",
            "start": "begin_scan",
            "next": "skip_scan",
            "skip": "skip_all",
            "cached": "skip_all",
        },
        "ready": {
            "go": "start_clone",
            "start": "start_clone",
            "clone": "start_clone",
        },
        "running": {
            "exit": "exit_clone",
            "stop": "exit_clone",
            "done": "exit_clone",
        },
    }

    def __init__(self):
        self.state = "init"
        self.room_image = None
        self.door_image = None
        self.face_images = []
        self.voice_audio_path = None
        self.clone_timeout = 300  # 5 minutes
        self.exit_event = threading.Event()
        self.scan_stop_event = threading.Event()
        self.flash_on = True

        # Voice controller (for commands from YOUR mic)
        self.voice = VoiceController(self.on_voice_command)
        self.voice_active = False
        self.last_heard = ""

        # Conversation transcriber (for both sides during clone running)
        self.transcriber = ConversationTranscriber(self.on_utterance)

        # Clone agent (AI brain + TTS)
        self.agent = CloneAgent(on_status_callback=self.on_agent_status)

        # Scan tracking
        self.audio_duration = 0
        self.scan_start_time = None

        # Initialize GUI
        self.setup_gui()

    def setup_gui(self):
        """Create the tkinter GUI window."""
        self.root = tk.Tk()
        self.root.title("Clone Controller")
        self.root.geometry("400x280")
        self.root.resizable(False, False)
        self.root.attributes("-topmost", True)

        # Main frame
        self.main_frame = tk.Frame(self.root, padx=10, pady=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Status indicator (colored box)
        self.status_frame = tk.Frame(self.main_frame, height=60)
        self.status_frame.pack(fill=tk.X, pady=(0, 5))
        self.status_frame.pack_propagate(False)

        self.status_label = tk.Label(
            self.status_frame,
            text="Press ENTER to Start",
            font=("Segoe UI", 14, "bold"),
            bg=COLORS["waiting"],
            fg="black"
        )
        self.status_label.pack(fill=tk.BOTH, expand=True)

        # Voice feedback frame
        self.voice_frame = tk.Frame(self.main_frame, height=40)
        self.voice_frame.pack(fill=tk.X, pady=(0, 5))
        self.voice_frame.pack_propagate(False)

        self.voice_label = tk.Label(
            self.voice_frame,
            text="",
            font=("Segoe UI", 10),
            fg="#666"
        )
        self.voice_label.pack(fill=tk.BOTH, expand=True)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.main_frame,
            variable=self.progress_var,
            maximum=100
        )

        # Progress text
        self.progress_text = tk.Label(
            self.main_frame,
            text="",
            font=("Segoe UI", 9)
        )

        # Commands help text
        self.commands_label = tk.Label(
            self.main_frame,
            text="",
            font=("Segoe UI", 9),
            fg="#444",
            justify=tk.LEFT
        )
        self.commands_label.pack(pady=(5, 0))

        # Help text
        self.help_label = tk.Label(
            self.main_frame,
            text="Press ENTER to activate voice control",
            font=("Segoe UI", 10),
            fg="gray"
        )
        self.help_label.pack(pady=(5, 0))

        # Bind keys
        self.root.bind("<Return>", self.on_enter)
        self.root.bind("<space>", self.on_enter)
        self.root.bind("<Escape>", self.on_escape)
        self.root.focus_force()

        # Update initial state
        self.update_state("init")

    def update_state(self, new_state, progress_text=""):
        """Update the state machine and GUI."""
        self.state = new_state

        state_info = {
            "init": ("Press ENTER to Start", COLORS["waiting"], "Press ENTER to activate voice control"),
            "wait_pic": ("Step 1: Room Setup", COLORS["listening"], "Listening..."),
            "wait_rec": ("Step 2: Face & Voice", COLORS["listening"], "Listening..."),
            "scanning": ("Scanning...", COLORS["generating"], "Press ENTER to end early (after 60s)"),
            "generating": ("Generating...", COLORS["generating"], progress_text),
            "ready": ("READY", COLORS["ready"], "Listening..."),
            "running": ("CLONE ACTIVE", COLORS["running"], "Listening..."),
            "exiting": ("Exiting...", COLORS["running"], "Clone leaving..."),
            "complete": ("Complete", COLORS["complete"], "Press ENTER to restart"),
        }

        text, color, help_text = state_info.get(new_state, ("Unknown", "gray", ""))

        self.status_label.config(text=text, bg=color)
        self.help_label.config(text=help_text, fg="gray")

        # Update commands display
        commands = self.COMMANDS.get(new_state, {})
        if commands:
            # Get unique command actions and their phrases
            cmd_display = []
            seen_actions = set()
            for phrase, action in commands.items():
                if action not in seen_actions:
                    cmd_display.append(f'"{phrase}"')
                    seen_actions.add(action)
            self.commands_label.config(text="Say: " + " | ".join(cmd_display))
        else:
            self.commands_label.config(text="")

        # Show/hide progress bar
        if new_state in ["generating", "scanning"]:
            self.progress_bar.pack(fill=tk.X, pady=(0, 5))
            self.progress_text.pack()
        else:
            self.progress_bar.pack_forget()
            self.progress_text.pack_forget()

        # Start flashing if ready
        if new_state == "ready":
            self.start_flash()
        else:
            self.stop_flash()

    def start_flash(self):
        """Start flashing green light when ready."""
        self.flash_on = True
        self._flash()

    def _flash(self):
        """Flash the status indicator."""
        if self.state != "ready":
            return

        if self.flash_on:
            self.status_label.config(bg=COLORS["ready"])
        else:
            self.status_label.config(bg="#006600")

        self.flash_on = not self.flash_on
        self.root.after(500, self._flash)

    def stop_flash(self):
        """Stop flashing."""
        self.flash_on = False

    def update_progress(self, percent, text=""):
        """Update progress bar and text."""
        self.progress_var.set(percent)
        self.progress_text.config(text=text)
        self.root.update()

    def on_enter(self, event=None):
        """Handle Enter/Space key."""
        if self.state == "init":
            # Activate voice control
            self.voice_active = True
            self.voice.start()
            self.voice_label.config(text="Voice control active", fg="#9370DB")
            self.update_state("wait_pic")

        elif self.state == "scanning":
            # End scan early if we have enough audio
            if self.audio_duration >= 60:
                self.scan_stop_event.set()
            else:
                self.help_label.config(text=f"Need {60 - self.audio_duration:.0f}s more audio", fg="orange")

        elif self.state == "complete":
            # Reset
            self.update_state("wait_pic")

    def on_escape(self, event=None):
        """ESC key - immediate shutdown."""
        self.log("ESC pressed - shutting down!")
        self.voice.stop()
        cleanup_lock()
        self.root.destroy()
        sys.exit(0)

    def on_utterance(self, speaker, text):
        """Handle transcribed utterance from conversation (for agent context)."""
        # This is called for both sides of the conversation during clone running
        label = "You" if speaker == "me" else "Them"
        self.log(f"[CONV] {label}: {text}")

        # If the interviewer (them) speaks and we're in running state, have the agent respond
        if speaker == "them" and self.state == "running":
            self.agent.on_interviewer_speaks(text)

    def on_agent_status(self, status, message):
        """Handle status updates from the clone agent."""
        # Update GUI from the agent's status
        if status == "thinking":
            self.voice_label.config(text=f"Thinking...", fg="#4169E1")
        elif status == "speaking":
            self.voice_label.config(text=f"Clone speaking...", fg="#FF4444")
        elif status == "idle":
            self.voice_label.config(text="Clone listening...", fg="#9370DB")
        try:
            self.root.update()
        except:
            pass  # GUI might be closed

    def on_voice_command(self, text):
        """Handle recognized voice command."""
        self.last_heard = text
        self.voice_label.config(text=f'Heard: "{text}"', fg="#9370DB")
        self.root.update()

        # Match command - sort by phrase length (longest first) to match specific before general
        commands = self.COMMANDS.get(self.state, {})
        matched_action = None

        # Sort phrases by length descending so "skip scan" matches before "scan"
        sorted_phrases = sorted(commands.keys(), key=len, reverse=True)

        for phrase in sorted_phrases:
            if phrase in text or self._fuzzy_match(phrase, text):
                matched_action = commands[phrase]
                self.log(f"Matched '{phrase}' -> {matched_action}")
                break

        if matched_action:
            self.log(f"Voice command: {matched_action}")
            self._execute_action(matched_action)
        else:
            self.voice_label.config(text=f'"{text}" - not recognized', fg="orange")

    def _fuzzy_match(self, phrase, text):
        """Simple fuzzy matching for voice commands."""
        # Check if most words from phrase are in text
        phrase_words = set(phrase.lower().split())
        text_words = set(text.lower().split())
        matches = len(phrase_words & text_words)
        return matches >= len(phrase_words) * 0.7

    def _execute_action(self, action):
        """Execute a voice command action."""
        if action == "snap_room":
            threading.Thread(target=self.do_capture_room, daemon=True).start()
        elif action == "snap_door":
            threading.Thread(target=self.do_capture_door, daemon=True).start()
        elif action == "skip_pics":
            self.do_skip_pics()
        elif action == "begin_scan":
            threading.Thread(target=self.do_scan, daemon=True).start()
        elif action == "skip_scan":
            self.do_skip_scan()
        elif action == "skip_all":
            self.do_skip_all()
        elif action == "start_clone":
            threading.Thread(target=self.do_start_clone, daemon=True).start()
        elif action == "exit_clone":
            self.exit_event.set()

    def log(self, msg):
        """Log to console."""
        print(f"[CLONE] {msg}")
        sys.stdout.flush()

    def do_capture_room(self):
        """Capture room with door closed."""
        self.update_state("generating", "Capturing room...")
        self.log("Capturing room (door closed)...")

        try:
            import obsws_python as obs

            for i in range(5, 0, -1):
                self.update_progress((5-i) * 15, f"Room in {i}...")
                self.voice_label.config(text=f"Capturing room in {i}...", fg="#4169E1")
                self.root.update()
                time.sleep(1)

            cl = obs.ReqClient(host=OBS_HOST, port=OBS_PORT, password=OBS_PASSWORD)
            result = cl.get_source_screenshot(
                name="Video Capture Device",
                img_format="png",
                width=1280,
                height=720,
                quality=100
            )

            img_data = result.image_data.split(",")[1]
            self.room_image = f"{VIDEO_DIR}\\room_capture.png"

            with open(self.room_image, "wb") as f:
                f.write(base64.b64decode(img_data))

            self.log(f"Room saved: {self.room_image}")
            self.voice_label.config(text="Room captured!", fg="green")

            # If both room and door are captured, move to phase 2
            if self.room_image and self.door_image:
                self.log("Both images captured, moving to Phase 2")
                self.update_state("wait_rec")
            else:
                self.update_state("wait_pic")

        except Exception as e:
            self.log(f"ERROR: {e}")
            self.help_label.config(text=f"Error: {e}", fg="red")
            self.update_state("wait_pic")

    def do_capture_door(self):
        """Capture room with door open."""
        self.update_state("generating", "Capturing door...")
        self.log("Capturing room (door open)...")

        try:
            import obsws_python as obs

            for i in range(5, 0, -1):
                self.update_progress((5-i) * 15, f"Door in {i}...")
                self.voice_label.config(text=f"Capturing door in {i}...", fg="#4169E1")
                self.root.update()
                time.sleep(1)

            cl = obs.ReqClient(host=OBS_HOST, port=OBS_PORT, password=OBS_PASSWORD)
            result = cl.get_source_screenshot(
                name="Video Capture Device",
                img_format="png",
                width=1280,
                height=720,
                quality=100
            )

            img_data = result.image_data.split(",")[1]
            self.door_image = f"{VIDEO_DIR}\\door_capture.png"

            with open(self.door_image, "wb") as f:
                f.write(base64.b64decode(img_data))

            self.log(f"Door saved: {self.door_image}")
            self.voice_label.config(text="Door captured!", fg="green")

            # If both room and door are captured, move to phase 2
            if self.room_image and self.door_image:
                self.log("Both images captured, moving to Phase 2")
                self.update_state("wait_rec")
            else:
                self.update_state("wait_pic")

        except Exception as e:
            self.log(f"ERROR: {e}")
            self.help_label.config(text=f"Error: {e}", fg="red")
            self.update_state("wait_pic")

    def do_skip_pics(self):
        """Skip picture capture, use cached images."""
        self.log("Skipping picture capture, using cached images...")

        # Check for cached images
        cached_room = f"{VIDEO_DIR}\\room_capture.png"
        cached_door = f"{VIDEO_DIR}\\door_capture.png"

        if os.path.exists(cached_room):
            self.room_image = cached_room
            self.log(f"Using cached room: {cached_room}")
        else:
            self.help_label.config(text="No cached room image!", fg="red")
            return

        if os.path.exists(cached_door):
            self.door_image = cached_door
            self.log(f"Using cached door: {cached_door}")

        self.voice_label.config(text="Using cached images", fg="green")
        self.update_state("wait_rec")

    def do_skip_scan(self):
        """Skip face/voice scan, use cached data and go straight to video generation."""
        self.log("Skipping scan, using cached face reference...")

        cached_face = f"{VIDEO_DIR}\\face_reference.png"
        cached_voice = f"{VIDEO_DIR}\\voice_sample.wav"

        if os.path.exists(cached_face):
            self.log(f"Using cached face: {cached_face}")
        else:
            self.help_label.config(text="No cached face_reference.png!", fg="red")
            return

        if os.path.exists(cached_voice):
            self.voice_audio_path = cached_voice
            self.log(f"Using cached voice: {cached_voice}")

        self.voice_label.config(text="Using cached scan data", fg="green")

        # Go straight to video generation
        threading.Thread(target=self.do_generate_videos, daemon=True).start()

    def do_skip_all(self):
        """Skip everything and go straight to ready state using cached videos."""
        self.log("Skipping all - using cached videos...")

        # Check for cached videos
        entry_vid = f"{VIDEO_DIR}\\entry.mp4"
        idle_vid = f"{VIDEO_DIR}\\idle_loop.mp4"
        exit_vid = f"{VIDEO_DIR}\\exit.mp4"

        missing = []
        if not os.path.exists(entry_vid):
            missing.append("entry.mp4")
        if not os.path.exists(idle_vid):
            missing.append("idle_loop.mp4")
        if not os.path.exists(exit_vid):
            missing.append("exit.mp4")

        if missing:
            self.help_label.config(text=f"Missing: {', '.join(missing)}", fg="red")
            return

        self.log("All cached videos found!")
        self.voice_label.config(text="Using cached videos", fg="green")

        # Setup OBS with cached videos
        threading.Thread(target=self._setup_and_ready, daemon=True).start()

    def _setup_and_ready(self):
        """Setup OBS scenes and go to ready state."""
        try:
            self.update_state("generating", "Setting up OBS...")
            self._setup_obs_scenes()
            self.log("OBS scenes configured with cached videos")
            self.update_state("ready")
        except Exception as e:
            self.log(f"ERROR: {e}")
            self.help_label.config(text=f"Error: {e}", fg="red")
            self.update_state("wait_rec")

    def do_scan(self):
        """Capture face screenshots and record audio for 90s max."""
        self.update_state("scanning")
        self.scan_stop_event.clear()
        self.log("=== STARTING FACE & VOICE SCAN ===")

        try:
            import pyautogui
            import pyaudio
            import numpy as np

            # Setup audio recording from VB-Cable (other person's voice)
            audio = pyaudio.PyAudio()

            dev_info = audio.get_device_info_by_index(CABLE_DEVICE)
            sample_rate = int(dev_info['defaultSampleRate'])
            channels = min(2, int(dev_info['maxInputChannels']))

            self.log(f"Recording from device {CABLE_DEVICE}: {dev_info['name']}")

            audio_stream = audio.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=sample_rate,
                input=True,
                input_device_index=CABLE_DEVICE,
                frames_per_buffer=1024
            )

            # Recording state
            self.face_images = []
            audio_frames = []
            self.audio_duration = 0
            self.scan_start_time = time.time()
            last_screenshot = 0
            screenshot_interval = 3  # seconds
            max_duration = 90  # seconds

            while not self.scan_stop_event.is_set():
                elapsed = time.time() - self.scan_start_time

                if elapsed >= max_duration:
                    self.log("Max scan duration reached")
                    break

                # Record audio chunk
                try:
                    data = audio_stream.read(1024, exception_on_overflow=False)
                    audio_frames.append(data)

                    # Check if this chunk has audio (for duration tracking)
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    rms = np.sqrt(np.mean(audio_data.astype(np.float32)**2))
                    if rms > 200:  # Has actual audio
                        self.audio_duration = len(audio_frames) * 1024 / sample_rate
                except:
                    pass

                # Take screenshot every interval
                if elapsed - last_screenshot >= screenshot_interval:
                    screenshot = pyautogui.screenshot()
                    path = f"{VIDEO_DIR}\\face_capture_{len(self.face_images)}.png"
                    screenshot.save(path)
                    self.face_images.append(path)
                    last_screenshot = elapsed
                    self.log(f"Screenshot {len(self.face_images)}: {path}")

                # Update progress
                progress = min(100, (self.audio_duration / 60) * 100)
                self.update_progress(progress, f"Recording... {self.audio_duration:.0f}s / 60s audio")

                time.sleep(0.01)  # Small sleep to prevent CPU spin

            # Save audio
            audio_stream.stop_stream()
            audio_stream.close()
            audio.terminate()

            self.voice_audio_path = f"{VIDEO_DIR}\\voice_sample.wav"
            with wave.open(self.voice_audio_path, 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(b''.join(audio_frames))

            self.log(f"Audio saved: {self.voice_audio_path} ({self.audio_duration:.1f}s)")
            self.log(f"Screenshots taken: {len(self.face_images)}")

            # Select best face with Gemini
            self.voice_label.config(text="Selecting best face...", fg="#4169E1")
            self._select_best_face()

            # Continue to video generation
            self.do_generate_videos()

        except Exception as e:
            import traceback
            self.log(f"ERROR in scan: {e}")
            traceback.print_exc()
            self.help_label.config(text=f"Error: {e}", fg="red")
            self.update_state("wait_rec")

    def _select_best_face(self):
        """Use Gemini to select the best face capture."""
        if not self.face_images:
            self.log("No face images to select from!")
            return

        from google import genai
        from google.genai import types
        import shutil

        self.log("Calling Gemini to select best face...")
        client = genai.Client(api_key=GOOGLE_API_KEY)

        content = [f"Here are {len(self.face_images)} screenshots of a person in a video call. Pick the image where the person's face is most clearly visible, well-lit, and facing the camera directly. Return ONLY the number (0-{len(self.face_images)-1}), nothing else."]

        for i, path in enumerate(self.face_images):
            with open(path, "rb") as f:
                img_data = f.read()
            content.append(f"Image {i}:")
            content.append(types.Part.from_bytes(data=img_data, mime_type="image/png"))

        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=content
        )

        response_text = response.candidates[0].content.parts[0].text.strip()
        self.log(f"Gemini selected: {response_text}")

        best_idx = int(''.join(c for c in response_text if c.isdigit())[:2])
        best_idx = max(0, min(len(self.face_images)-1, best_idx))

        best_path = self.face_images[best_idx]
        ref_path = f"{VIDEO_DIR}\\face_reference.png"
        shutil.copy(best_path, ref_path)
        self.log(f"Face reference saved: {ref_path}")

    def do_generate_videos(self):
        """Generate all clone videos."""
        self.update_state("generating", "Starting video generation...")
        self.log("=== STARTING VIDEO GENERATION ===")

        try:
            self.update_progress(5, "Creating seated reference...")
            self.log("Generating seated image with Gemini...")
            self._generate_sitdown_frame()

            self.update_progress(20, "Generating ENTRY video (2-3 min)...")
            self.log("Generating ENTRY video...")
            self._generate_video("entry", 10)

            self.update_progress(50, "Generating IDLE loop (2-3 min)...")
            self.log("Generating IDLE video...")
            self._generate_video("idle", 5)

            self.update_progress(70, "Generating EXIT video (2-3 min)...")
            self.log("Generating EXIT video...")
            self._generate_video("exit", 10)

            self.update_progress(90, "Setting up OBS...")
            self.log("Configuring OBS scenes...")
            self._setup_obs_scenes()

            self.update_progress(100, "Ready!")
            self.log("=== VIDEO GENERATION COMPLETE ===")
            self.voice_label.config(text="Videos ready!", fg="green")
            time.sleep(0.5)

            self.update_state("ready")

        except Exception as e:
            import traceback
            self.log(f"ERROR in video generation: {e}")
            traceback.print_exc()
            # Show clear error message AND fallback action
            self.voice_label.config(text=f"Video gen FAILED - trying cached", fg="orange")
            self.help_label.config(text=f"Error: {e}", fg="red")
            self.root.update()
            time.sleep(1)  # Let user see the message
            # Try to use cached videos instead of going back to wait_rec
            self.log("Attempting to use cached videos instead...")
            self.do_skip_all()

    def _generate_sitdown_frame(self):
        """Generate seated clone image with Gemini."""
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=GOOGLE_API_KEY)

        face_path = f"{VIDEO_DIR}\\face_reference.png"
        with open(face_path, "rb") as f:
            face_data = f.read()
        face_part = types.Part.from_bytes(data=face_data, mime_type="image/png")

        with open(self.room_image, "rb") as f:
            room_data = f.read()
        room_part = types.Part.from_bytes(data=room_data, mime_type="image/png")

        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=[
                "REFERENCE PERSON (use this face only):",
                face_part,
                "ROOM TO EDIT:",
                room_part,
                """Edit the ROOM image to add the person from the first image.
                The person should be FULLY SEATED in the smaller wooden chair on the LEFT side.
                They should be facing the camera, looking natural and relaxed.
                Keep the room exactly as shown, only add the seated person.
                The person should wear casual clothes (olive green t-shirt)."""
            ],
            config=types.GenerateContentConfig(
                response_modalities=['IMAGE', 'TEXT'],
            )
        )

        for part in response.candidates[0].content.parts:
            if hasattr(part, 'inline_data') and part.inline_data:
                output_path = f"{VIDEO_DIR}\\clone_seated.png"
                with open(output_path, "wb") as f:
                    f.write(part.inline_data.data)
                self.log(f"Seated image saved: {output_path}")
                return

        raise Exception("Gemini did not return an image")

    def _generate_video(self, video_type, duration):
        """Generate video using Kling."""
        import fal_client
        import requests

        # FAL_KEY should already be set at module load, but ensure it's there
        if not os.environ.get("FAL_KEY"):
            os.environ["FAL_KEY"] = FAL_KEY

        def image_to_data_url(path):
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            return f"data:image/png;base64,{data}"

        seated_frame = f"{VIDEO_DIR}\\clone_seated.png"
        empty_room = self.room_image

        if video_type == "entry":
            start_url = image_to_data_url(empty_room)
            end_url = image_to_data_url(seated_frame)
            prompt = "The door opens. A person enters through the door, walks across the room, and sits down in the chair on the left. The door closes. Smooth natural movement."
            output = f"{VIDEO_DIR}\\entry.mp4"

        elif video_type == "idle":
            start_url = image_to_data_url(seated_frame)
            end_url = start_url
            prompt = "Person seated in chair, breathing naturally. Subtle mouth movement as if about to speak. Blinks occasionally. Very subtle idle motion for seamless loop."
            output = f"{VIDEO_DIR}\\idle_loop.mp4"

        elif video_type == "exit":
            start_url = image_to_data_url(seated_frame)
            end_url = image_to_data_url(empty_room)
            prompt = "Person stands up from chair, walks around behind the large chair, approaches the door, opens it and exits. The door closes leaving the room empty. Smooth natural walking."
            output = f"{VIDEO_DIR}\\exit.mp4"

        else:
            raise ValueError(f"Unknown video type: {video_type}")

        self.log(f"Submitting {video_type} to Kling ({duration}s)...")
        result = fal_client.subscribe(
            "fal-ai/kling-video/v2.5-turbo/pro/image-to-video",
            arguments={
                "prompt": prompt,
                "image_url": start_url,
                "tail_image_url": end_url,
                "duration": str(duration),
                "aspect_ratio": "16:9",
            },
        )

        video_url = result['video']['url']
        self.log(f"Downloading {video_type}...")
        resp = requests.get(video_url)
        with open(output, "wb") as f:
            f.write(resp.content)
        self.log(f"Saved: {output}")

    def _setup_obs_scenes(self):
        """Configure OBS scenes."""
        import obsws_python as obs

        cl = obs.ReqClient(host=OBS_HOST, port=OBS_PORT, password=OBS_PASSWORD)

        existing = cl.get_scene_list()
        existing_names = [s['sceneName'] for s in existing.scenes]

        for scene in ["Entry", "IdleLoop", "Exit"]:
            if scene not in existing_names:
                cl.create_scene(scene)

        videos = {
            "Entry": ("EntryVideo", f"{VIDEO_DIR}\\entry.mp4", False),
            "IdleLoop": ("IdleVideo", f"{VIDEO_DIR}\\idle_loop.mp4", True),
            "Exit": ("ExitVideo", f"{VIDEO_DIR}\\exit.mp4", False),
        }

        for scene, (input_name, path, looping) in videos.items():
            # Full ffmpeg_source settings for proper video playback
            media_settings = {
                "local_file": path,
                "is_local_file": True,
                "looping": looping,
                "restart_on_activate": True,
                "clear_on_media_end": not looping,  # Don't clear if looping
                "speed_percent": 100,
            }

            try:
                # Try to update existing input settings
                # API: set_input_settings(name, settings, overlay)
                cl.set_input_settings(
                    name=input_name,
                    settings=media_settings,
                    overlay=True
                )
                self.log(f"Updated {input_name} settings: {path}")
            except Exception as e:
                try:
                    # Create new input if it doesn't exist
                    # API: create_input(sceneName, inputName, inputKind, inputSettings, sceneItemEnabled)
                    cl.create_input(
                        sceneName=scene,
                        inputName=input_name,
                        inputKind="ffmpeg_source",
                        inputSettings=media_settings,
                        sceneItemEnabled=True,
                    )
                    self.log(f"Created {input_name}: {path}")
                except Exception as e2:
                    self.log(f"OBS input {input_name} error: {e2}")

            # Set audio monitoring to "Monitor and Output" so we hear the video
            # API: set_input_audio_monitor_type(name, mon_type)
            # mon_type: OBS_MONITORING_TYPE_MONITOR_AND_OUTPUT
            try:
                cl.set_input_audio_monitor_type(
                    name=input_name,
                    mon_type="OBS_MONITORING_TYPE_MONITOR_AND_OUTPUT"
                )
                self.log(f"Set audio monitoring for {input_name}")
            except Exception as e:
                self.log(f"Audio monitoring for {input_name}: {e}")

    def do_start_clone(self):
        """Start the clone sequence."""
        self.update_state("running")
        self.exit_event.clear()
        self.log("=== CLONE ACTIVATED ===")

        # Start conversation transcriber (listens to both sides)
        self.transcriber.start()

        # Start the AI agent (will respond to interviewer)
        self.agent.start()
        self.log("Clone agent started - will respond to interviewer")

        try:
            import obsws_python as obs

            cl = obs.ReqClient(host=OBS_HOST, port=OBS_PORT, password=OBS_PASSWORD)

            # Play Entry video
            self._update_clone_status("Entering room...")
            cl.set_current_program_scene("Entry")
            # Restart the video to ensure playback
            try:
                cl.trigger_media_input_action(
                    name="EntryVideo",
                    action="OBS_WEBSOCKET_MEDIA_INPUT_ACTION_RESTART"
                )
            except Exception as e:
                self.log(f"Entry video restart: {e}")
            time.sleep(10)

            # Switch to idle loop
            self._update_clone_status("Listening...")
            cl.set_current_program_scene("IdleLoop")
            try:
                cl.trigger_media_input_action(
                    name="IdleVideo",
                    action="OBS_WEBSOCKET_MEDIA_INPUT_ACTION_RESTART"
                )
            except Exception as e:
                self.log(f"Idle video restart: {e}")

            start_time = time.time()
            while not self.exit_event.is_set():
                elapsed = time.time() - start_time
                remaining = self.clone_timeout - elapsed
                if remaining <= 0:
                    self._update_clone_status("Timeout - leaving...")
                    break
                # Update status every 5 seconds with remaining time
                mins = int(remaining // 60)
                secs = int(remaining % 60)
                self._update_clone_status(f"Listening... ({mins}:{secs:02d} left)")
                time.sleep(0.5)

            # Stop agent and transcriber before exiting
            self.agent.stop()
            self.transcriber.stop()
            self.log(f"Final transcript:\n{self.transcriber.get_transcript()}")

            # Play Exit video
            self.update_state("exiting")
            self._update_clone_status("Leaving room...")
            cl.set_current_program_scene("Exit")
            try:
                cl.trigger_media_input_action(
                    name="ExitVideo",
                    action="OBS_WEBSOCKET_MEDIA_INPUT_ACTION_RESTART"
                )
            except Exception as e:
                self.log(f"Exit video restart: {e}")
            time.sleep(10)

            # Switch back to webcam scene
            self._update_clone_status("Back to webcam")
            self.log("Switching back to webcam...")
            try:
                cl.set_current_program_scene("Scene")  # Default OBS scene with webcam
            except:
                try:
                    cl.set_current_program_scene("Webcam")  # Try alternate name
                except:
                    self.log("Could not find webcam scene to switch to")

            self.update_state("complete")
            self._update_clone_status("Session complete")

        except Exception as e:
            self.agent.stop()
            self.transcriber.stop()
            self.log(f"ERROR: {e}")
            self.help_label.config(text=f"Error: {e}", fg="red")
            self.update_state("ready")

    def _update_clone_status(self, status):
        """Update the GUI to show what the clone is doing."""
        self.voice_label.config(text=f"Clone: {status}", fg="#FF4444")
        self.log(f"[STATUS] {status}")
        try:
            self.root.update()
        except:
            pass

    def run(self):
        """Start the GUI main loop."""
        self.root.mainloop()
        # Cleanup
        self.voice.stop()


if __name__ == "__main__":
    check_single_instance()
    controller = CloneController()
    controller.run()
