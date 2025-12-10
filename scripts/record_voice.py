#!/usr/bin/env python3
"""
Voice Recording Script for Cloner

Records audio from webcam microphone (or other input devices) for voice cloning.
Saves high-quality audio suitable for ElevenLabs voice cloning.

Usage:
    python scripts/record_voice.py list              # List audio devices
    python scripts/record_voice.py record -o out.wav # Record until Enter pressed
    python scripts/record_voice.py record -d 60      # Record for 60 seconds
    python scripts/record_voice.py record --device 1 # Record from device index 1
"""

import sys
import os
import wave
import time
import threading
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import click
import numpy as np

try:
    import pyaudio
except ImportError:
    print("ERROR: pyaudio not installed. Install with:")
    print("  pip install pyaudio")
    print("  # On Windows, you may need: pip install pipwin && pipwin install pyaudio")
    sys.exit(1)

# Recording settings optimized for voice cloning
SAMPLE_RATE = 44100  # CD quality
CHANNELS = 1         # Mono (best for voice cloning)
CHUNK_SIZE = 1024    # Frames per buffer
FORMAT = pyaudio.paInt16  # 16-bit audio


class VoiceRecorder:
    """Records audio from microphone with real-time level monitoring."""

    def __init__(self, device_index: int = None, sample_rate: int = SAMPLE_RATE):
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.is_recording = False
        self.peak_level = 0

    def list_devices(self) -> list:
        """List all available audio input devices."""
        devices = []
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:  # Input device
                devices.append({
                    'index': i,
                    'name': info['name'],
                    'channels': info['maxInputChannels'],
                    'sample_rate': int(info['defaultSampleRate']),
                    'is_default': i == self.audio.get_default_input_device_info()['index']
                })
        return devices

    def find_device_by_name(self, name_pattern: str) -> int:
        """Find device index by partial name match."""
        devices = self.list_devices()
        name_lower = name_pattern.lower()
        for dev in devices:
            if name_lower in dev['name'].lower():
                return dev['index']
        return None

    def start_recording(self):
        """Start recording audio."""
        self.frames = []
        self.is_recording = True
        self.peak_level = 0

        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=CHUNK_SIZE,
            stream_callback=self._audio_callback
        )
        self.stream.start_stream()

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream - captures frames and calculates level."""
        if self.is_recording:
            self.frames.append(in_data)
            # Calculate audio level for monitoring
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            self.peak_level = max(self.peak_level, np.abs(audio_data).max())
        return (None, pyaudio.paContinue)

    def get_level(self) -> float:
        """Get current audio level (0-100)."""
        level = (self.peak_level / 32768.0) * 100
        self.peak_level = 0  # Reset for next measurement
        return min(level, 100)

    def stop_recording(self):
        """Stop recording audio."""
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

    def save_wav(self, output_path: str) -> str:
        """Save recorded audio to WAV file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with wave.open(str(output_path), 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(FORMAT))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(self.frames))

        return str(output_path)

    def get_duration(self) -> float:
        """Get recording duration in seconds."""
        total_frames = len(self.frames) * CHUNK_SIZE
        return total_frames / self.sample_rate

    def cleanup(self):
        """Clean up audio resources."""
        if self.stream:
            self.stream.close()
        self.audio.terminate()


def print_level_bar(level: float, width: int = 40):
    """Print a visual audio level bar."""
    filled = int(level / 100 * width)
    bar = '#' * filled + '-' * (width - filled)
    color = '\033[92m' if level < 70 else '\033[93m' if level < 90 else '\033[91m'
    reset = '\033[0m'
    print(f"\r  Level: [{color}{bar}{reset}] {level:5.1f}%  ", end='', flush=True)


@click.group()
def cli():
    """Voice Recording Tool for Cloner - Record audio for voice cloning."""
    pass


@cli.command()
def list():
    """List available audio input devices."""
    recorder = VoiceRecorder()
    devices = recorder.list_devices()
    recorder.cleanup()

    if not devices:
        click.echo("No audio input devices found!")
        return

    click.echo("\nAvailable Audio Input Devices:")
    click.echo("-" * 60)
    for dev in devices:
        default = " [DEFAULT]" if dev['is_default'] else ""
        click.echo(f"  [{dev['index']}] {dev['name']}{default}")
        click.echo(f"      Channels: {dev['channels']}, Sample Rate: {dev['sample_rate']}Hz")
    click.echo("-" * 60)
    click.echo("\nUse --device INDEX or --device-name NAME to select a device.")


@cli.command()
@click.option('-o', '--output', type=click.Path(), help='Output file path (default: auto-generated)')
@click.option('-d', '--duration', type=int, default=None, help='Recording duration in seconds (default: until Enter)')
@click.option('--device', type=int, default=None, help='Device index to record from')
@click.option('--device-name', type=str, default=None, help='Device name pattern to match (e.g., "USB CAMERA")')
@click.option('--sample-rate', type=int, default=SAMPLE_RATE, help=f'Sample rate in Hz (default: {SAMPLE_RATE})')
@click.option('--no-monitor', is_flag=True, help='Disable audio level monitoring')
def record(output, duration, device, device_name, sample_rate, no_monitor):
    """Record audio from microphone.

    Records until Enter is pressed (or for specified duration).
    Audio is saved as high-quality WAV suitable for voice cloning.
    """
    recorder = VoiceRecorder(device_index=device, sample_rate=sample_rate)

    # Find device by name if specified
    if device_name:
        found_device = recorder.find_device_by_name(device_name)
        if found_device is not None:
            recorder.device_index = found_device
            click.echo(f"Found device: [{found_device}] matching '{device_name}'")
        else:
            click.echo(f"ERROR: No device matching '{device_name}' found.")
            recorder.list_devices()
            recorder.cleanup()
            return

    # Show selected device
    if recorder.device_index is not None:
        devices = recorder.list_devices()
        selected = next((d for d in devices if d['index'] == recorder.device_index), None)
        if selected:
            click.echo(f"\nRecording from: {selected['name']}")
    else:
        click.echo("\nRecording from: Default input device")

    # Generate output filename if not specified
    if not output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent.parent / "data" / "audio" / "recordings"
        output_dir.mkdir(parents=True, exist_ok=True)
        output = str(output_dir / f"voice_recording_{timestamp}.wav")

    click.echo(f"Output file: {output}")
    click.echo(f"Sample rate: {sample_rate}Hz, Channels: {CHANNELS} (mono)")
    click.echo("-" * 60)

    if duration:
        click.echo(f"Recording for {duration} seconds...")
    else:
        click.echo("Press ENTER to stop recording...")

    click.echo("-" * 60)

    # Start recording
    recorder.start_recording()
    start_time = time.time()

    stop_event = threading.Event()

    def wait_for_enter():
        input()
        stop_event.set()

    if not duration:
        # Start thread to wait for Enter key
        input_thread = threading.Thread(target=wait_for_enter, daemon=True)
        input_thread.start()

    try:
        while True:
            elapsed = time.time() - start_time

            if duration and elapsed >= duration:
                break
            if stop_event.is_set():
                break

            # Show level meter
            if not no_monitor:
                level = recorder.get_level()
                print_level_bar(level)

            time.sleep(0.1)

    except KeyboardInterrupt:
        click.echo("\n\nRecording interrupted!")

    # Stop and save
    recorder.stop_recording()
    print()  # New line after level bar

    recording_duration = recorder.get_duration()
    if recording_duration < 1:
        click.echo("WARNING: Recording too short (< 1 second). Not saving.")
        recorder.cleanup()
        return

    saved_path = recorder.save_wav(output)
    recorder.cleanup()

    click.echo("-" * 60)
    click.echo(f"Recording saved: {saved_path}")
    click.echo(f"Duration: {recording_duration:.1f} seconds")

    # Validate for voice cloning
    try:
        from voice.audio_processor import AudioProcessor
        validation = AudioProcessor.validate_for_cloning(saved_path)
        click.echo("\nVoice Cloning Validation:")
        if validation['warnings']:
            for warning in validation['warnings']:
                click.echo(f"  WARNING: {warning}")
        else:
            click.echo("  Audio file looks good for voice cloning!")
    except ImportError:
        pass  # Skip validation if audio_processor not available

    click.echo("\nNext steps:")
    click.echo(f"  python scripts/clone_voice.py clone -a {saved_path} --name 'MyVoice'")


@cli.command()
@click.option('--device', type=int, default=None, help='Device index to test')
@click.option('--device-name', type=str, default=None, help='Device name pattern to match')
@click.option('--duration', type=int, default=5, help='Test duration in seconds (default: 5)')
def test(device, device_name, duration):
    """Test microphone input levels.

    Records for a few seconds and shows audio levels.
    Useful for checking if the microphone is working.
    """
    recorder = VoiceRecorder(device_index=device)

    if device_name:
        found = recorder.find_device_by_name(device_name)
        if found is not None:
            recorder.device_index = found
        else:
            click.echo(f"No device matching '{device_name}'")
            recorder.cleanup()
            return

    click.echo(f"\nTesting microphone for {duration} seconds...")
    click.echo("Speak into the microphone to see level changes.\n")

    recorder.start_recording()
    start_time = time.time()

    try:
        while time.time() - start_time < duration:
            level = recorder.get_level()
            print_level_bar(level)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    recorder.stop_recording()
    recorder.cleanup()
    click.echo("\n\nMicrophone test complete!")


if __name__ == '__main__':
    cli()
