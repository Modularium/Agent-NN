module.exports = {
  docs: [
    'overview',
    'setup',
    {
      type: 'category',
      label: 'Installation & Setup',
      items: ['install', 'developer_setup'],
    },
    {
      type: 'category',
      label: 'CLI & API',
      items: ['cli', 'api/README', 'api/api_reference'],
    },
    {
      type: 'category',
      label: 'Agenten & Sessions',
      items: [ 'agent_deployment', 'sessions'],
    },
    {
      type: 'category',
      label: 'Tool Integrationen',
      items: ['integrations/n8n', 'integrations/flowise', 'development/model_tracking'],
    },
    'config_reference',
    {
      type: 'category',
      label: 'Entwicklung',
      items: ['development/api_overview'],
    },
    'troubleshooting/troubleshooting_guide',
  ],
};
