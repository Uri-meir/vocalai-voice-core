# Gemini Native Audio Realtime POC

This project is a Proof of Concept (POC) demonstrating real-time, bidirectional audio communication with Google's `gemini-2.5-flash-native-audio-preview-09-2025` model using WebSockets.

## Features

- **Real-time Audio Streaming**: Captures microphone input and streams it to Gemini.
- **Native Audio Response**: Receives high-quality (24kHz) PCM audio from Gemini and plays it back instantly.
- **Modular Architecture**: Clean separation of concerns (Audio Input, Audio Output, Gemini Client, Config).
- **AsyncIO**: Fully asynchronous implementation using Python's `asyncio`.
- **Extensible**: Ready for future integration with Telephony (Twilio/Plivo) and Tool Calling.

## Prerequisites

- Python 3.10+
- macOS, Linux, or Windows
- A Google Cloud Project with Gemini API access
- API Key (Get it from [Google AI Studio](https://aistudio.google.com/))

## Installation

1.  **Clone the repository** (if applicable) or navigate to the project directory:
    ```bash
    cd POC_gemini_realtime
    ```

2.  **Create a Virtual Environment** (Recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

   *Note: On macOS, you might need to install `portaudio` first if `pyaudio` installation fails:*
   ```bash
   brew install portaudio
   ```

4.  **Configure Environment**:
    copy `.env.example` to `.env` and add your API key:
    ```bash
    cp .env.example .env
    ```
    Edit `.env`:
    ```ini
    GEMINI_API_KEY=your_actual_api_key_here
    USER_PHONE_NUMBER=+972500000000
    ```

## Usage

Run the main application:

```bash
python src/main.py
```

- You should see logs indicating successful connection.
- Speak into your microphone.
- You will hear Gemini respond through your speakers.
- Press `Ctrl+C` to stop.

## Project Structure

```
src/
├── audio/
│   ├── audio_input.py   # Microphone Helper
│   └── audio_output.py  # Speaker Helper
├── gemini/
│   └── client.py        # Gemini Live API Client
├── config/
│   └── environment.py   # Config loading
├── utils/
│   └── logging_setup.py # Logger config
└── main.py              # Entry point
```

## Troubleshooting

- **Microphone issues**: Ensure your OS permissions allow the terminal/python to access the microphone.
- **Audio glitches**: Adjust `CHUNK_SIZE` in `src/config/environment.py`.
- **Connection errors**: Verify your API Key and internet connection.
