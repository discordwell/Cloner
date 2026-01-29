#!/usr/bin/env python3
"""
Clone Interview Runner - CLI for the integrated interview system.

Usage:
    # Setup from webcam (first time)
    python run_interview.py setup --name "my_clone" --duration 10

    # Start interview with existing clone
    python run_interview.py start --name "my_clone"

    # Test with manual text input
    python run_interview.py test --name "my_clone" --text "Hello world"

    # List audio devices
    python run_interview.py devices
"""

import sys
import time
import logging
from pathlib import Path

import click

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def cli(verbose):
    """Clone Interview System CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


@cli.command()
@click.option('--name', '-n', required=True, help='Clone name/ID')
@click.option('--duration', '-d', default=10, help='Webcam capture duration (seconds)')
@click.option('--video', '-f', help='Use existing video file instead of webcam')
@click.option('--clone-voice/--no-voice', default=False, help='Clone voice with ElevenLabs')
@click.option('--context', '-c', default='', help='Context for LLM responses')
def setup(name, duration, video, clone_voice, context):
    """Setup a new clone from webcam or video."""
    from src.clone_interview import CloneInterview

    click.echo("=" * 50)
    click.echo("Clone Interview Setup")
    click.echo("=" * 50)

    interview = CloneInterview()

    if video:
        click.echo(f"\nSetting up from video: {video}")
        interview.setup_from_video(
            video_path=video,
            subject_id=name,
            clone_voice=clone_voice,
            context=context
        )
    else:
        click.echo(f"\nCapturing from webcam for {duration} seconds...")
        click.echo("Look at the camera and speak naturally.")
        interview.setup_from_webcam(
            subject_id=name,
            duration=duration,
            clone_voice=clone_voice,
            context=context
        )

    click.echo("\n" + "=" * 50)
    click.echo("Setup complete!")
    click.echo(f"Clone name: {name}")
    click.echo(f"Library: data/visemes/{name}_enhanced")
    click.echo("\nRun with: python run_interview.py start --name " + name)


@cli.command()
@click.option('--name', '-n', required=True, help='Clone name/ID')
@click.option('--context', '-c', default='', help='Context for LLM responses')
@click.option('--device', '-d', type=int, help='Audio device index')
def start(name, context, device):
    """Start interview mode with existing clone."""
    from src.clone_interview import CloneInterview

    click.echo("=" * 50)
    click.echo("Clone Interview System")
    click.echo("=" * 50)

    interview = CloneInterview()

    try:
        interview.setup_from_existing(name, context=context)
    except FileNotFoundError:
        click.echo(f"Error: No clone found with name '{name}'")
        click.echo("Run setup first: python run_interview.py setup --name " + name)
        sys.exit(1)

    click.echo(f"\nLoaded clone: {name}")
    click.echo("Starting interview mode...")
    click.echo("\nSpeak into your microphone to ask questions.")
    click.echo("Press Ctrl+C to stop.\n")
    click.echo("=" * 50 + "\n")

    def on_response(question, response, result):
        """Log each interaction."""
        total = result.get('timings', {}).get('total', 0)
        click.echo(f"\n[Q]: {question}")
        click.echo(f"[A]: {response}")
        click.echo(f"[Time: {total:.2f}s]\n")

    try:
        interview.start_listening(device_index=device, on_response=on_response)

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        click.echo("\n\nStopping interview...")
    finally:
        interview.stop_listening()

    click.echo("Interview ended.")


@cli.command()
@click.option('--name', '-n', required=True, help='Clone name/ID')
@click.option('--text', '-t', required=True, help='Text to speak')
@click.option('--question/--statement', default=True, help='Treat as question (use LLM)')
def test(name, text, question):
    """Test clone with manual text input."""
    from src.clone_interview import CloneInterview

    click.echo(f"Testing clone: {name}")
    click.echo(f"Input: {text}")
    click.echo()

    interview = CloneInterview()

    try:
        interview.setup_from_existing(name)
    except FileNotFoundError:
        click.echo(f"Error: No clone found with name '{name}'")
        sys.exit(1)

    if question:
        click.echo("Generating LLM response...")
        result = interview.respond_to_text(text, play=True)
        click.echo(f"\nResponse: {result.get('response_text', 'N/A')}")
    else:
        click.echo("Speaking text directly...")
        result = interview.say(text, play=True)

    click.echo(f"\nTimings:")
    for k, v in result.get('timings', {}).items():
        click.echo(f"  {k}: {v:.2f}s")


@cli.command()
def devices():
    """List available audio input devices."""
    from src.clone_interview import AudioListener

    listener = AudioListener()
    devices = listener.list_audio_devices()

    click.echo("Available audio input devices:")
    click.echo("-" * 50)

    for d in devices:
        click.echo(f"  [{d['index']:2d}] {d['name']}")
        click.echo(f"       {d['channels']} channels, {d['sample_rate']}Hz")

    loopback = listener.find_loopback_device()
    if loopback is not None:
        click.echo(f"\nDetected loopback device: [{loopback}]")
    else:
        click.echo("\nNo loopback device detected.")
        click.echo("For system audio capture, install BlackHole (macOS) or VB-Cable (Windows)")


@cli.command()
def list_clones():
    """List existing clones."""
    viseme_dir = project_root / "data" / "visemes"

    if not viseme_dir.exists():
        click.echo("No clones found.")
        return

    clones = []
    for item in viseme_dir.iterdir():
        if item.is_dir():
            metadata_file = item / "metadata.json"
            if metadata_file.exists():
                clones.append(item.name)

    if not clones:
        click.echo("No clones found.")
        return

    click.echo("Available clones:")
    click.echo("-" * 30)
    for name in sorted(clones):
        # Remove _enhanced suffix for display
        display_name = name.replace("_enhanced", "")
        click.echo(f"  - {display_name}")


@cli.command()
@click.option('--name', '-n', required=True, help='Clone name/ID')
@click.option('--text', '-t', default="Hello! This is a test of the clone interview system.",
              help='Text to speak')
def benchmark(name, text):
    """Benchmark clone response latency."""
    from src.clone_interview import CloneInterview

    click.echo("=" * 50)
    click.echo("Clone Interview Benchmark")
    click.echo("=" * 50)
    click.echo(f"Clone: {name}")
    click.echo(f"Text: {text}")
    click.echo()

    interview = CloneInterview()

    try:
        interview.setup_from_existing(name)
    except FileNotFoundError:
        click.echo(f"Error: No clone found with name '{name}'")
        sys.exit(1)

    # Warmup
    click.echo("Warmup run...")
    interview.say(text, play=False)

    # Benchmark runs
    click.echo("\nRunning 5 benchmark iterations...")
    times = []

    for i in range(5):
        start = time.time()
        result = interview.say(text, play=False)
        elapsed = time.time() - start
        times.append(elapsed)
        click.echo(f"  Run {i+1}: {elapsed:.3f}s")

    avg = sum(times) / len(times)
    min_t = min(times)
    max_t = max(times)

    click.echo()
    click.echo("Results:")
    click.echo(f"  Average: {avg:.3f}s")
    click.echo(f"  Min:     {min_t:.3f}s")
    click.echo(f"  Max:     {max_t:.3f}s")

    # Breakdown from last run
    timings = result.get('timings', {})
    if timings:
        click.echo("\nBreakdown (last run):")
        for k, v in timings.items():
            click.echo(f"  {k}: {v:.3f}s")


if __name__ == "__main__":
    cli()
