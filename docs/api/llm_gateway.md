# LLM Gateway API

`POST /generate` – Send a prompt and receive generated text from the configured language model.

`POST /chain/qa` – Perform retrieval augmented generation. The gateway queries the Vector Store Service and uses the results as context for the model.

`GET /health` – Health check endpoint returning `{"status": "ok"}`.

