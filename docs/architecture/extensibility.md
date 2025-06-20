# MCP Extensibility

The Modular Control Plane is designed to grow. New services can be added without disrupting existing ones. The LLM Gateway already exposes additional endpoints:

- `POST /translate` – translate text into a target language.
- `POST /vision` – placeholder for multimodal image understanding.

Planned integrations include Whisper for speech-to-text, CLIP or BLIP for vision tasks and Gemini/OpenAI tools for advanced reasoning.

Each service communicates over HTTP and can be scaled independently. Configuration lives in `config/services.yaml` and allows enabling or duplicating services per environment.
