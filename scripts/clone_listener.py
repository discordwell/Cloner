#!/usr/bin/env python3
"""
Clone Listener - Automatic speech detection and transcription

Listens to system audio (Meet output), detects when someone stops talking,
transcribes what was said, and triggers the clone brain to respond.

Flow:
    System Audio (Meet) → Detect speech → Transcribe → Clone Brain → TTS → OBS

Usage:
    python clone_listener.py              # Start listening
    python clone_listener.py --test       # Test audio capture
    python clone_listener.py --devices    # List audio devices
"""

import os
import sys
import time
import wave
import tempfile
import threading
import numpy as np
from pathlib import Path
from collections import deque

import pyaudio
from openai import OpenAI

# Load environment
from dotenv import load_dotenv
load_dotenv(Path(r"C:\Users\cordw\iCloudDrive\Documents\Projects\Cloner\.env"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Audio settings
SAMPLE_RATE = 48000  # Stereo Mix runs at 48kHz
CHANNELS = 2  # Stereo Mix is stereo
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16

# Voice activity detection settings
SILENCE_THRESHOLD = 500      # RMS threshold for silence
SPEECH_THRESHOLD = 800       # RMS threshold for speech
MIN_SPEECH_DURATION = 0.5    # Minimum seconds of speech to trigger
SILENCE_AFTER_SPEECH = 1.5   # Seconds of silence after speech to trigger transcription
MAX_RECORDING_DURATION = 30  # Maximum recording length in seconds

# Directory for audio files
AUDIO_DIR = Path("/mnt/c/Users/cordw/clone_videos/listener_audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


class CloneListener:
    """Listens for speech, transcribes, and triggers clone responses."""

    def __init__(self, device_index: int = None, on_speech_callback=None):
        """
        Args:
            device_index: Audio input device index (None for default)
            on_speech_callback: Function to call with transcribed text
        """
        self.device_index = device_index
        self.on_speech_callback = on_speech_callback
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_listening = False
        self.client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

        # State for voice activity detection
        self.is_speaking = False
        self.speech_start_time = None
        self.silence_start_time = None
        self.audio_buffer = deque(maxlen=int(SAMPLE_RATE * MAX_RECORDING_DURATION / CHUNK_SIZE))

    def list_devices(self):
        """List all available audio input devices."""
        devices = []
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                devices.append({
                    'index': i,
                    'name': info['name'],
                    'channels': info['maxInputChannels'],
                    'sample_rate': int(info['defaultSampleRate'])
                })
        return devices

    def find_stereo_mix(self):
        """Try to find a loopback/stereo mix device for capturing system audio.

        Priority:
        1. VB-Cable Output (best - captures only Meet audio)
        2. Stereo Mix / other loopback (captures all system audio)
        """
        devices = self.list_devices()

        # Priority 1: VB-Cable (best option)
        for device in devices:
            name_lower = device['name'].lower()
            if 'cable output' in name_lower or 'vb-audio' in name_lower:
                print(f"Found VB-Cable: {device['name']}")
                return device['index']

        # Priority 2: Other loopback devices
        keywords = ['stereo mix', 'loopback', 'what u hear', 'wave out']
        for device in devices:
            name_lower = device['name'].lower()
            for keyword in keywords:
                if keyword in name_lower:
                    return device['index']
        return None

    def get_rms(self, audio_data):
        """Calculate RMS (volume level) of audio data."""
        data = np.frombuffer(audio_data, dtype=np.int16)
        return np.sqrt(np.mean(data.astype(np.float32) ** 2))

    def start_listening(self):
        """Start listening for speech."""
        if self.is_listening:
            return

        device_idx = self.device_index
        if device_idx is None:
            device_idx = self.find_stereo_mix()
            if device_idx:
                print(f"Using loopback device: {self.audio.get_device_info_by_index(device_idx)['name']}")
            else:
                print("Warning: No loopback device found. Using default input.")

        # Get device info to use correct sample rate and channels
        if device_idx is not None:
            dev_info = self.audio.get_device_info_by_index(device_idx)
            sample_rate = int(dev_info['defaultSampleRate'])
            channels = int(dev_info['maxInputChannels'])
            print(f"Device: {sample_rate}Hz, {channels} channels")
        else:
            sample_rate = SAMPLE_RATE
            channels = CHANNELS

        self.actual_sample_rate = sample_rate
        self.actual_channels = channels

        try:
            self.stream = self.audio.open(
                format=FORMAT,
                channels=channels,
                rate=sample_rate,
                input=True,
                input_device_index=device_idx,
                frames_per_buffer=CHUNK_SIZE
            )
            self.is_listening = True
            print("Listening for speech...")

            # Start the listening loop in a thread
            self.listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
            self.listen_thread.start()

        except Exception as e:
            print(f"Error starting audio stream: {e}")
            raise

    def stop_listening(self):
        """Stop listening."""
        self.is_listening = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

    def _listen_loop(self):
        """Main listening loop with voice activity detection."""
        recording_frames = []

        while self.is_listening:
            try:
                audio_data = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
                rms = self.get_rms(audio_data)
                current_time = time.time()

                # Voice activity detection
                if rms > SPEECH_THRESHOLD:
                    # Speech detected
                    if not self.is_speaking:
                        self.is_speaking = True
                        self.speech_start_time = current_time
                        recording_frames = list(self.audio_buffer)  # Include pre-buffer
                        print("  [Speech started]")

                    self.silence_start_time = None
                    recording_frames.append(audio_data)

                elif self.is_speaking:
                    # Currently in speech mode but silence detected
                    recording_frames.append(audio_data)

                    if rms < SILENCE_THRESHOLD:
                        if self.silence_start_time is None:
                            self.silence_start_time = current_time
                        elif current_time - self.silence_start_time > SILENCE_AFTER_SPEECH:
                            # Silence long enough - process the recording
                            speech_duration = current_time - self.speech_start_time
                            if speech_duration >= MIN_SPEECH_DURATION:
                                print(f"  [Speech ended - {speech_duration:.1f}s]")
                                self._process_recording(recording_frames)
                            else:
                                print("  [Too short, ignoring]")

                            # Reset state
                            self.is_speaking = False
                            self.speech_start_time = None
                            self.silence_start_time = None
                            recording_frames = []
                    else:
                        self.silence_start_time = None

                # Keep a rolling buffer for pre-speech capture
                self.audio_buffer.append(audio_data)

            except Exception as e:
                print(f"Error in listen loop: {e}")
                time.sleep(0.1)

    def _process_recording(self, frames):
        """Save recording to file and transcribe."""
        if not frames:
            return

        # Save to temp file
        timestamp = int(time.time())
        filepath = AUDIO_DIR / f"speech_{timestamp}.wav"

        with wave.open(str(filepath), 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(FORMAT))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b''.join(frames))

        print(f"  [Saved: {filepath.name}]")

        # Transcribe
        if self.client:
            try:
                transcription = self._transcribe(filepath)
                if transcription and transcription.strip():
                    print(f"  [Heard: \"{transcription}\"]")
                    if self.on_speech_callback:
                        self.on_speech_callback(transcription)
            except Exception as e:
                print(f"  [Transcription error: {e}]")

    def _transcribe(self, audio_file: Path) -> str:
        """Transcribe audio using OpenAI Whisper."""
        with open(audio_file, 'rb') as f:
            response = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="en"
            )
        return response.text

    def cleanup_old_files(self, keep_last: int = 10):
        """Remove old audio files."""
        files = sorted(AUDIO_DIR.glob("speech_*.wav"), key=os.path.getmtime)
        for f in files[:-keep_last]:
            try:
                os.unlink(f)
            except:
                pass


def test_audio_capture(device_index: int = None):
    """Test audio capture and show volume levels."""
    listener = CloneListener(device_index=device_index)

    print("\nTesting audio capture (Ctrl+C to stop)")
    print("=" * 50)

    devices = listener.list_devices()
    print("\nAvailable input devices:")
    for d in devices:
        print(f"  [{d['index']}] {d['name']}")

    loopback = listener.find_stereo_mix()
    if loopback:
        print(f"\nFound loopback device: [{loopback}]")

    device_idx = device_index or loopback
    print(f"\nUsing device: {device_idx or 'default'}")
    print("\nVolume levels (speak or play audio):")
    print("-" * 50)

    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        input_device_index=device_idx,
        frames_per_buffer=CHUNK_SIZE
    )

    try:
        while True:
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            rms = listener.get_rms(data)
            bar_len = int(rms / 100)
            bar = "#" * min(bar_len, 50)
            status = "SPEECH" if rms > SPEECH_THRESHOLD else "silence" if rms < SILENCE_THRESHOLD else "noise"
            print(f"\r{rms:6.0f} [{bar:<50}] {status}  ", end="", flush=True)
    except KeyboardInterrupt:
        print("\n\nTest complete.")
    finally:
        stream.stop_stream()
        stream.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Clone Listener - Speech detection")
    parser.add_argument("--test", action="store_true", help="Test audio capture")
    parser.add_argument("--devices", action="store_true", help="List audio devices")
    parser.add_argument("--device", type=int, help="Audio device index to use")
    parser.add_argument("--speak", action="store_true", help="Also speak responses")

    args = parser.parse_args()

    if args.devices:
        listener = CloneListener()
        print("\nAvailable input devices:")
        for d in listener.list_devices():
            print(f"  [{d['index']}] {d['name']} ({d['channels']}ch, {d['sample_rate']}Hz)")
        return

    if args.test:
        test_audio_capture(args.device)
        return

    # Full listening mode with clone brain integration
    print("\n" + "=" * 60)
    print("CLONE LISTENER - Automatic Speech Detection")
    print("=" * 60)

    # Import the brain
    sys.path.insert(0, r"C:\Users\cordw")
    from clone_brain import CloneBrain

    brain = CloneBrain(speak=args.speak)

    def on_speech(text):
        """Called when speech is detected and transcribed."""
        print(f"\n[Interviewer]: {text}")
        print("[Clone thinking...]")
        response = brain.respond(text)
        print(f"[Clone]: {response}\n")

    listener = CloneListener(device_index=args.device, on_speech_callback=on_speech)

    try:
        listener.start_listening()
        print("\nListening... Press Ctrl+C to stop.\n")
        while True:
            time.sleep(1)
            listener.cleanup_old_files()
    except KeyboardInterrupt:
        print("\n\nStopping listener...")
    finally:
        listener.stop_listening()


if __name__ == "__main__":
    main()
