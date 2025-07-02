# Flowise AgentNN Component (v1.0.2)

This component forwards user input to the Agent‑NN API gateway. Run `npm install` and `npx tsc` in this folder to generate `dist/AgentNN.js`.
Upload that file through the Flowise UI and configure the `endpoint` of your Agent‑NN instance. Optional parameters allow you to modify `taskType`, set a custom API `path`, choose the HTTP `method`, pass additional `headers`, configure Basic or Bearer authentication via `auth` and adjust the request `timeout`.

Example flow definitions are provided in `sample_flow.json`. The integration catches HTTP errors and returns them in a structured form.
