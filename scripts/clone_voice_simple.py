#!/usr/bin/env python3
"""
Simple voice cloning script using ElevenLabs API directly.
Works with elevenlabs==1.2.0
"""

import os
import sys
from pathlib import Path

# Load environment
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import click
import requests

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
BASE_URL = "https://api.elevenlabs.io/v1"


def clone_voice(name: str, audio_files: list, description: str = None) -> dict:
    """Clone a voice from audio files."""
    url = f"{BASE_URL}/voices/add"

    headers = {
        "xi-api-key": ELEVENLABS_API_KEY
    }

    data = {
        "name": name,
        "description": description or f"Cloned voice: {name}"
    }

    files = []
    for audio_path in audio_files:
        files.append(("files", (Path(audio_path).name, open(audio_path, "rb"), "audio/wav")))

    response = requests.post(url, headers=headers, data=data, files=files)

    # Close files
    for _, f in files:
        f[1].close()

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to clone voice: {response.status_code} - {response.text}")


def list_voices() -> list:
    """List all available voices."""
    url = f"{BASE_URL}/voices"
    headers = {"xi-api-key": ELEVENLABS_API_KEY}

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()["voices"]
    else:
        raise Exception(f"Failed to list voices: {response.text}")


def generate_speech(voice_id: str, text: str, output_path: str,
                   stability: float = 0.5, similarity_boost: float = 0.75,
                   stream: bool = False) -> str:
    """Generate speech from text using a voice."""
    url = f"{BASE_URL}/text-to-speech/{voice_id}"
    if stream:
        url += "/stream"

    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }

    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity_boost
        }
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if stream:
        # Streaming - write chunks as they arrive
        response = requests.post(url, headers=headers, json=data, stream=True)
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            return str(output_path)
        else:
            raise Exception(f"Failed to generate speech: {response.text}")
    else:
        # Non-streaming
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(response.content)
            return str(output_path)
        else:
            raise Exception(f"Failed to generate speech: {response.text}")


def stream_speech_realtime(voice_id: str, text: str,
                          stability: float = 0.5, similarity_boost: float = 0.75,
                          on_chunk=None):
    """
    Stream speech in realtime - yields audio chunks as they're generated.

    Args:
        voice_id: ElevenLabs voice ID
        text: Text to convert to speech
        stability: Voice stability (0-1)
        similarity_boost: Similarity boost (0-1)
        on_chunk: Optional callback function(chunk_bytes) called for each chunk

    Yields:
        Audio chunks as bytes
    """
    url = f"{BASE_URL}/text-to-speech/{voice_id}/stream"

    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }

    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity_boost
        }
    }

    response = requests.post(url, headers=headers, json=data, stream=True)

    if response.status_code != 200:
        raise Exception(f"Failed to stream speech: {response.text}")

    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            if on_chunk:
                on_chunk(chunk)
            yield chunk


def delete_voice(voice_id: str) -> bool:
    """Delete a cloned voice."""
    url = f"{BASE_URL}/voices/{voice_id}"
    headers = {"xi-api-key": ELEVENLABS_API_KEY}

    response = requests.delete(url, headers=headers)
    return response.status_code == 200


@click.group()
def cli():
    """Simple ElevenLabs Voice Cloning CLI"""
    if not ELEVENLABS_API_KEY:
        click.echo("ERROR: ELEVENLABS_API_KEY not set in .env file")
        sys.exit(1)


@cli.command()
@click.option('-a', '--audio', multiple=True, required=True, help='Audio file(s) for cloning')
@click.option('--name', required=True, help='Name for the cloned voice')
@click.option('--description', default=None, help='Description for the voice')
def clone(audio, name, description):
    """Clone a voice from audio samples."""
    click.echo(f"\nCloning voice '{name}' from {len(audio)} audio file(s)...")

    # Validate files exist
    for path in audio:
        if not Path(path).exists():
            click.echo(f"ERROR: Audio file not found: {path}")
            return

    try:
        result = clone_voice(name, list(audio), description)
        click.echo(f"\nVoice cloned successfully!")
        click.echo(f"  Voice ID: {result['voice_id']}")
        click.echo(f"  Name: {name}")
        click.echo(f"\nTo generate speech:")
        click.echo(f"  python scripts/clone_voice_simple.py speak --voice {result['voice_id']} -t \"Hello world\" -o output.mp3")
    except Exception as e:
        click.echo(f"ERROR: {e}")


@cli.command("list")
def list_cmd():
    """List all available voices."""
    click.echo("\nAvailable Voices:")
    click.echo("-" * 60)

    try:
        voices = list_voices()
        for v in voices:
            category = v.get("category", "unknown")
            click.echo(f"  [{v['voice_id'][:8]}...] {v['name']} ({category})")
        click.echo("-" * 60)
        click.echo(f"Total: {len(voices)} voices")
    except Exception as e:
        click.echo(f"ERROR: {e}")


@cli.command()
@click.option('--voice', required=True, help='Voice ID to use')
@click.option('-t', '--text', required=True, help='Text to speak')
@click.option('-o', '--output', required=True, help='Output audio file path')
@click.option('--stability', default=0.5, help='Voice stability (0-1)')
@click.option('--similarity', default=0.75, help='Similarity boost (0-1)')
@click.option('--stream', is_flag=True, help='Use streaming API (faster first audio)')
def speak(voice, text, output, stability, similarity, stream):
    """Generate speech from text."""
    mode = "streaming" if stream else "standard"
    click.echo(f"\nGenerating speech ({mode}) with voice {voice[:8]}...")
    click.echo(f"Text: \"{text[:50]}{'...' if len(text) > 50 else ''}\"")

    try:
        import time
        start = time.time()
        output_path = generate_speech(voice, text, output, stability, similarity, stream=stream)
        elapsed = time.time() - start
        click.echo(f"\nSpeech generated in {elapsed:.2f}s: {output_path}")
    except Exception as e:
        click.echo(f"ERROR: {e}")


@cli.command()
@click.option('--voice', required=True, help='Voice ID to use')
@click.option('-t', '--text', required=True, help='Text to speak')
@click.option('-o', '--output', required=True, help='Output audio file path')
@click.option('--play', is_flag=True, help='Play audio after streaming completes')
def stream(voice, text, output, play):
    """Stream speech in realtime (fastest time-to-first-audio).

    Audio chunks are written to file as they arrive from the API.
    Use --play to play the audio after streaming completes.
    """
    click.echo(f"\nStreaming speech with voice {voice[:8]}...")
    click.echo(f"Text: \"{text[:50]}{'...' if len(text) > 50 else ''}\"")

    import time
    start = time.time()
    first_chunk_time = None
    total_bytes = 0

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_path, "wb") as f:
            for chunk in stream_speech_realtime(voice, text):
                if first_chunk_time is None:
                    first_chunk_time = time.time() - start
                    click.echo(f"  First audio chunk in {first_chunk_time:.2f}s")

                f.write(chunk)
                total_bytes += len(chunk)

        elapsed = time.time() - start
        click.echo(f"\nStreaming complete!")
        click.echo(f"  Total time: {elapsed:.2f}s")
        click.echo(f"  First chunk: {first_chunk_time:.2f}s")
        click.echo(f"  File size: {total_bytes / 1024:.1f}KB")
        click.echo(f"  Saved to: {output_path}")

        # Play the completed file
        if play:
            click.echo(f"\nPlaying audio...")
            try:
                from pydub import AudioSegment
                from pydub.playback import play as pydub_play
                audio = AudioSegment.from_mp3(output_path)
                pydub_play(audio)
            except ImportError:
                # Fallback to system player
                import subprocess
                import platform
                if platform.system() == "Windows":
                    subprocess.run(["cmd", "/c", "start", "", str(output_path)], check=False)
                elif platform.system() == "Darwin":
                    subprocess.run(["open", str(output_path)], check=False)
                else:
                    subprocess.run(["xdg-open", str(output_path)], check=False)

    except Exception as e:
        click.echo(f"ERROR: {e}")


@cli.command()
@click.option('--voice', required=True, help='Voice ID to delete')
@click.confirmation_option(prompt='Are you sure you want to delete this voice?')
def delete(voice):
    """Delete a cloned voice."""
    try:
        if delete_voice(voice):
            click.echo(f"Voice {voice} deleted successfully.")
        else:
            click.echo(f"Failed to delete voice {voice}")
    except Exception as e:
        click.echo(f"ERROR: {e}")


if __name__ == "__main__":
    cli()
