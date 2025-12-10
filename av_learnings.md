# Audio/Video Setup Learnings - Clone Controller

## What Finally Works

### Video Pipeline
```
Physical Webcam → OBS (with filters) → OBS Virtual Camera → Google Meet (in Edge)
```

### Audio Pipeline (Capturing Other Person's Voice)
```
Google Meet Audio Output → VB-Cable Input (as Meet speaker) → VB-Cable Output → clone_listener.py
                                                            ↓
                                          "Listen to this device" → Your Headphones
```

## Key Setup Steps

### 1. OBS Configuration
- **OBS must be running** at all times during the meeting
- Add "Video Capture Device" source for your webcam
- **Start Virtual Camera** (button in OBS) - this creates the fake camera Meet will use
- The Virtual Camera outputs whatever scene is currently active

### 2. Color Correction for Webcam
- Problem: Webcam had yellow tint through virtual camera
- Solution: Add Color Correction filter in OBS
  - Right-click video source → Filters → Add Color Correction
  - Adjust color/saturation/gamma as needed

### 3. VB-Cable Installation
- Downloaded from: https://vb-audio.com/Cable/
- Run `VBCABLE_Setup_x64.exe` as Administrator
- Restart Windows after install
- Creates two virtual audio devices:
  - **CABLE Input** - audio goes IN here
  - **CABLE Output** - audio comes OUT here

### 4. VB-Cable Audio Routing
- **In Google Meet settings:**
  - Microphone: Your regular mic (so the other person hears you)
  - Speaker: **CABLE Input (VB-Audio Virtual Cable)** - sends their voice to VB-Cable

- **To hear the other person yourself:**
  - Windows Sound Settings → Recording → CABLE Output
  - Right-click → Properties → Listen tab
  - Check "Listen to this device"
  - Select your headphones as playback device

- **For clone_listener.py:**
  - Use device 23: "CABLE Output (VB-Audio Virtual Cable) (2ch, 48000Hz)"
  - This captures ONLY the interviewer's voice, not your mic

### 5. Browser Choice
- **Use Microsoft Edge** - Chrome had issues with camera access
- Meet in Edge works better with OBS Virtual Camera

### 6. Meet Camera Selection
- In Google Meet video settings, select "OBS Virtual Camera"
- If it says "another app is using the camera" - that's OBS holding the real webcam (correct behavior)
- The Virtual Camera is separate from the physical camera

## Sticking Points & Solutions

| Problem | Cause | Solution |
|---------|-------|----------|
| "Another app using camera" | OBS has physical webcam | Select OBS Virtual Camera in Meet, not the real one |
| Black screen in OBS Virtual Camera | Wrong OBS scene active | Switch to scene with webcam source |
| Can't hear interviewer | Audio going to VB-Cable only | Enable "Listen to this device" on CABLE Output |
| Yellow tint on video | Webcam white balance | Add Color Correction filter in OBS |
| Stereo Mix not working | Wrong sample rate / not enabled | Use VB-Cable instead - cleaner routing |
| clone_listener not detecting speech | Wrong audio device | Use `--devices` flag to list, pick CABLE Output |
| Meet zooms in on face | Meet's auto-framing | Generated videos will be full-frame, should look normal |

## Dependencies Installed

```bash
pip install pyautogui Pillow pyscreeze  # Screenshots
pip install google-genai                 # Gemini API
pip install fal-client                   # Kling video generation
pip install openai python-dotenv         # Whisper API
pip install pyaudio numpy                # Audio capture
pip install obsws-python                 # OBS WebSocket control
```

## Audio Device Reference

From `clone_listener.py --devices`:
```
[23] CABLE Output (VB-Audio Virtual Cable) (2ch, 48000Hz)  ← USE THIS
[22] Microphone (USB CAMERA) (1ch, 48000Hz)
[24] Stereo Mix (Realtek High Definition Audio) (2ch, 48000Hz)
```

## OBS WebSocket Settings
- Port: 4455
- Password: slopifywins
- Used by clone_controller.py to switch scenes and capture screenshots

## File Paths (Windows format required)
Scripts use Windows paths like `C:\Users\cordw\...` NOT WSL paths like `/mnt/c/Users/cordw/...`
The .env file must also use Windows paths.
