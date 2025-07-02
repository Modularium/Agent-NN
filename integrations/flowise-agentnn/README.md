# Flowise AgentNN Component

A Flowise component that forwards input to the Agent-NN API gateway.
Run `npm install` and then `npx tsc` to compile the script.
Upload the generated `dist/AgentNN.js` through the Flowise UI and configure the
`endpoint` of your Agent-NN instance. Optional parameters allow you to change
`taskType`, send HTTP headers and adjust the request timeout.
