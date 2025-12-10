🎙️ VocalAI Realtime Voice Engine
High-Performance Telephony ↔ Voice AI Bridge

Author: Uri Meir

🚀 Overview

VocalAI Realtime Voice Engine is a low-latency, bidirectional audio streaming server that connects traditional telephony (Twilio) with modern Voice AI models in real time.

It enables:

📞 Real-time phone conversations with AI

🔁 Two-way streaming audio (user ↔ model)

🧠 Intelligent session handling (start/end, duration, metadata)

📡 Webhook event delivery compatible with existing Vapi-style workflows

🧱 Modular AI adapters — model-agnostic architecture (Gemini / OpenAI / Deepgram / Custom)

This engine forms the foundation of the VocalAI Voice Provider - a fully self-hosted, scalable solution.

🏗️ Architecture
User (Phone)
   │
PSTN / SIP
   │
Twilio
   │          HTTP (Webhooks), WebSocket (Media)
   ▼
FastAPI Realtime Engine
 ┌──────────────────────────────────────────┐
 │ Telephony Layer                          │
 │   • /twilio/voice-hook                   │
 │   • /twilio/media-stream                 │
 │                                          │
 │ Core Logic                               │
 │   • CallSession + SessionStore           │
 │   • EventEmitter (call.started / ended)  │
 │                                          │
 │ Audio Layer                              │
 │   • Transcoding Mulaw ↔ PCM              │
 │   • Resampling (8k ↔ 16/24k)             │
 │                                          │
 │ Model Adapters                           │
 │   • AIClient Interface                   │
 │   • GeminiClient / OpenAIClient / …      │
 └──────────────────────────────────────────┘
   │
   ▼
Supabase (Webhooks, Logs, Usage)


The system is built to be:

Scalable (websocket-per-call model with horizontal autoscaling)

Model-agnostic (swap AI providers without touching telephony)

Extensible (tool calling, structured reasoning, multiple voices)

Low-latency (optimized audio pipeline)

🔌 Supported Workflows
1. Inbound Calls

Twilio triggers /twilio/voice-hook

Server responds with TwiML <Stream>

Twilio opens a WebSocket to /twilio/media-stream

Real-time streaming begins (two-way)

On disconnect → call.ended event is emitted

2. Outbound Calls

UI → N8N → /call/start

Twilio dials user

Once call connects → streaming begins

call.started / call.ended callbacks sent to Supabase

Both flows appear identical to the rest of your platform.

🔧 Module Overview
Telephony Layer

Handles Twilio HTTP & WebSocket traffic

Converts telephony audio into model-ready audio and back

Owns streaming loops

Core Logic

SessionStore: tracks all live calls

CallSession: encapsulates call state

EventEmitter: forwards events to Supabase in Vapi-compatible format

Model Adapter Layer

A flexible interface:

class AIClient:
    async def connect(self): ...
    async def send_audio(self, pcm_bytes): ...
    async def receive_audio(self): ...
    async def close(self): ...


You may plug in:

Gemini Live

OpenAI Realtime

Deepgram Aura

Custom local inference
…and the system behaves the same.

Audio Processing

Mulaw ↔ PCM16

Resample 8k ↔ 16/24k

Level normalization

🌐 REST API
POST /call/start

Initiates an outbound call through Twilio.

Request
{
  "assistantId": "abc123",
  "phoneNumberId": "twilio_number_uuid",
  "customer": { "number": "+972500000000" }
}

Response
{ "status": "initiated", "call_id": "xyz789" }

🧪 Development
uvicorn src.main:app --reload


Optionally expose locally for Twilio:

ngrok http 8000

🗂️ Project Structure
src/
├── api/                 # REST endpoints (call/start)
├── telephony/           # Twilio voice hook + media stream
├── core/                # Session and event logic
├── ai_providers/        # Swappable model clients
├── audio/               # Transcoding & resampling
├── utils/               # Logging, helpers
└── config/              # Environment + config loader

📣 Why This Engine?

Full control over latency, cost, and routing

Battle-tested audio pipeline

Scales to thousands of concurrent calls

Adaptable to any future AI provider

Already plugged into your existing Supabase + N8N + React system

🏁 Status

✔ Inbound calls fully operational

✔ Outbound calls integrated

✔ Supabase event compatibility verified

✔ Session management stable

✔ Model-agnostic architecture

⏳ Next: multi-model routing, tool calling, diarization
