# SmartRotator — Live Voice/Realtime Gateway

SmartRotator ek multi-provider LLM gateway hai. Text mode (`/v1/chat/completions`)
ke alawa ab isme **Gemini Live API (realtime voice/audio)** support bhi hai.

## Text vs Live — kya farak hai

| | Text | Live / Voice |
|---|---|---|
| Endpoint | `POST /v1/chat/completions` | `wss://…/v1/live` (WebSocket) |
| Models | `gemini-3.5-flash` etc. | `gemini-3.1-flash-live-preview`, `gemini-*-tts-*`, `native-audio` |
| Protocol | OpenAI-compatible (stateless) | Gemini BidiGenerateContent (stateful, bidirectional) |
| Auth | `Authorization: Bearer <token>` | `?key=<API_KEY>` ya `Authorization` header |

**Important:** `gemini-*-live-preview` / `gemini-*-tts-*` jaise models Google ke
**WebSocket Live API** se chalti hain — unhe `generateContent` / OpenAI
`/chat/completions` se call nahi kar sakte. Isliye `/v1/models` me unpe
`"live": true` flag hota hai. Agar kisi app ne live model ko chat/completions
pe bheja toh wo fail hoga — isliye app ko `"live": true` wale models ke liye
WebSocket `/v1/live` route karna chahiye.

## `/v1/live` — Gemini Live API proxy

SmartRotator apni (rotated) gemini key user ke liye inject karta hai aur saare
frames Google tak relay karta hai. Aap apni user API key (`sk-…` ya JWT token)
se connect karte ho — SmartRotator ke credentials client ko expose nahi hote.

### Connection

```
wss://smartrotator.onrender.com/v1/live?key=<YOUR_API_KEY>
```

Ya phir `Authorization: Bearer <token>` header se. Quota per-session reserve
hota hai (text endpoints jaisa hi).

### Protocol (raw Gemini BidiGenerateContent)

Pahla message **setup** hona chahiye (snake_case — Gemini Live API raw schema):

```json
{
  "setup": {
    "model": "models/gemini-3.1-flash-live-preview",
    "generation_config": {
      "response_modalities": ["AUDIO"]
    },
    "system_instruction": {"parts": [{"text": "You are a helpful assistant."}]},
    "input_audio_transcription": {},
    "output_audio_transcription": {}
  }
}
```

Phir client frames bhejta hai, server frames wapas aati hain:

**Client → Server:**
```json
{"realtimeInput": {"audio": {"data": "<base64 PCM 16kHz 16bit>", "mimeType": "audio/pcm"}, "text": "..."}}
{"clientContent": {"turns": [{"role": "user", "parts": [{"text": "hi"}]}], "turnComplete": true}}
{"toolResponse": {"functionResponses": [{"id": "...", "name": "...", "response": {}}]}}
```

**Server → Client:**
```json
{"setupComplete": {}}
{"serverContent": {"modelTurn": {"parts": [{"inlineData": {"data": "<base64 audio 24kHz>", "mimeType": "audio/pcm"}}, {"text": "..."}]}, "turnComplete": true}}
{"toolCall": {"functionCalls": [{"id": "...", "name": "...", "args": {}}]}}
```

> **Note:** Raw Live API schema **snake_case** hai (protobuf-JSON), na ki
> camelCase jo SDKs use karte hain. Jo bhi clameCase bhejega, Google
> websocket 1007 error dega. Saare examples yahan raw schema ke hain.

### SDKs / frameworks

Raw WebSocket ke saath kaam karte hain, isliye ye frameworks seedha
`/v1/live` se connect ho sakte hain (apne Watkins endpoint me
`wss://smartrotator.onrender.com/v1/live` daalo + apni user key):

- **LiveKit** — `LiveKitAgent`/plugin `gemini_live` me WebSocket WSS URL set karo
- **Pipecat** — GoogleLLMService / Gemini live transport
- **Twilio MediaStreams** — Gemini Live bridge
- **Custom JS/RN**: `new WebSocket("wss://smartrotator.onrender.com/v1/live?key=...")`

## Error handling

Server invalid auth / quota / setup pe graceful JSON error bhejta hai aur
websocket close code ke saath band karta hai:

```json
{"error": {"code": "UNAUTHORIZED", "message": "..."}}
{"error": {"code": "QUOTA_EXCEEDED", "message": "..."}}
{"error": {"code": "INVALID_SETUP", "message": "..."}}
{"error": {"code": "UPSTREAM_CONNECT_FAILED", "message": "Gemini Live API se connect nahi hua: ..."}}
{"error": {"code": "SESSION_ERROR", "message": "..."}}
```

## Local dev

```bash
pip install -r requirements.txt
GEMINI_KEYS="key1,key2" SMARTROTATOR_DATA_DIR=./data python cli.py serve
# phir websocket client: wss://localhost:8000/v1/live?key=<apikey>
```
