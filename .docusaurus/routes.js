import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/Agent-NN/',
    component: ComponentCreator('/Agent-NN/', '682'),
    routes: [
      {
        path: '/Agent-NN/',
        component: ComponentCreator('/Agent-NN/', '9d3'),
        routes: [
          {
            path: '/Agent-NN/',
            component: ComponentCreator('/Agent-NN/', '3b7'),
            routes: [
              {
                path: '/Agent-NN/agent_deployment',
                component: ComponentCreator('/Agent-NN/agent_deployment', 'd19'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/Agent-NN/api',
                component: ComponentCreator('/Agent-NN/api', '0d7'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/Agent-NN/api_reference',
                component: ComponentCreator('/Agent-NN/api_reference', 'fe4'),
                exact: true
              },
              {
                path: '/Agent-NN/api/advanced_endpoints',
                component: ComponentCreator('/Agent-NN/api/advanced_endpoints', '3b0'),
                exact: true
              },
              {
                path: '/Agent-NN/api/agent_registry',
                component: ComponentCreator('/Agent-NN/api/agent_registry', 'e7f'),
                exact: true
              },
              {
                path: '/Agent-NN/api/api_reference',
                component: ComponentCreator('/Agent-NN/api/api_reference', 'a6c'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/Agent-NN/api/chat',
                component: ComponentCreator('/Agent-NN/api/chat', 'b0a'),
                exact: true
              },
              {
                path: '/Agent-NN/api/dispatcher',
                component: ComponentCreator('/Agent-NN/api/dispatcher', 'bc8'),
                exact: true
              },
              {
                path: '/Agent-NN/api/errors',
                component: ComponentCreator('/Agent-NN/api/errors', 'dee'),
                exact: true
              },
              {
                path: '/Agent-NN/api/llm_gateway',
                component: ComponentCreator('/Agent-NN/api/llm_gateway', 'd6a'),
                exact: true
              },
              {
                path: '/Agent-NN/api/openapi-overview',
                component: ComponentCreator('/Agent-NN/api/openapi-overview', '573'),
                exact: true
              },
              {
                path: '/Agent-NN/api/plugin_agent',
                component: ComponentCreator('/Agent-NN/api/plugin_agent', '43c'),
                exact: true
              },
              {
                path: '/Agent-NN/api/session_manager',
                component: ComponentCreator('/Agent-NN/api/session_manager', '2d1'),
                exact: true
              },
              {
                path: '/Agent-NN/api/vector_store',
                component: ComponentCreator('/Agent-NN/api/vector_store', 'd32'),
                exact: true
              },
              {
                path: '/Agent-NN/api/worker_services',
                component: ComponentCreator('/Agent-NN/api/worker_services', 'be1'),
                exact: true
              },
              {
                path: '/Agent-NN/architecture_overview',
                component: ComponentCreator('/Agent-NN/architecture_overview', '8ed'),
                exact: true
              },
              {
                path: '/Agent-NN/architecture/access_control',
                component: ComponentCreator('/Agent-NN/architecture/access_control', '23d'),
                exact: true
              },
              {
                path: '/Agent-NN/architecture/agent_bus',
                component: ComponentCreator('/Agent-NN/architecture/agent_bus', '5f5'),
                exact: true
              },
              {
                path: '/Agent-NN/architecture/agent_evolution',
                component: ComponentCreator('/Agent-NN/architecture/agent_evolution', 'f83'),
                exact: true
              },
              {
                path: '/Agent-NN/architecture/agent_levels',
                component: ComponentCreator('/Agent-NN/architecture/agent_levels', '791'),
                exact: true
              },
              {
                path: '/Agent-NN/architecture/agent_memory',
                component: ComponentCreator('/Agent-NN/architecture/agent_memory', '505'),
                exact: true
              },
              {
                path: '/Agent-NN/architecture/agent_missions',
                component: ComponentCreator('/Agent-NN/architecture/agent_missions', '38b'),
                exact: true
              },
              {
                path: '/Agent-NN/architecture/agent_profiles',
                component: ComponentCreator('/Agent-NN/architecture/agent_profiles', '531'),
                exact: true
              },
              {
                path: '/Agent-NN/architecture/agent_teams',
                component: ComponentCreator('/Agent-NN/architecture/agent_teams', '6a0'),
                exact: true
              },
              {
                path: '/Agent-NN/architecture/analysis',
                component: ComponentCreator('/Agent-NN/architecture/analysis', 'cad'),
                exact: true
              },
              {
                path: '/Agent-NN/architecture/audit_logging',
                component: ComponentCreator('/Agent-NN/architecture/audit_logging', '0c6'),
                exact: true
              },
              {
                path: '/Agent-NN/architecture/coalitions',
                component: ComponentCreator('/Agent-NN/architecture/coalitions', 'd22'),
                exact: true
              },
              {
                path: '/Agent-NN/architecture/Code-Dokumentation',
                component: ComponentCreator('/Agent-NN/architecture/Code-Dokumentation', '6b2'),
                exact: true
              },
              {
                path: '/Agent-NN/architecture/decisions/phase5_phase6',
                component: ComponentCreator('/Agent-NN/architecture/decisions/phase5_phase6', 'ad9'),
                exact: true
              },
              {
                path: '/Agent-NN/architecture/delegation_model',
                component: ComponentCreator('/Agent-NN/architecture/delegation_model', '458'),
                exact: true
              },
              {
                path: '/Agent-NN/architecture/dispatch_queue',
                component: ComponentCreator('/Agent-NN/architecture/dispatch_queue', '28b'),
                exact: true
              },
              {
                path: '/Agent-NN/architecture/dynamic_roles',
                component: ComponentCreator('/Agent-NN/architecture/dynamic_roles', 'a94'),
                exact: true
              },
              {
                path: '/Agent-NN/architecture/extensibility',
                component: ComponentCreator('/Agent-NN/architecture/extensibility', '3c2'),
                exact: true
              },
              {
                path: '/Agent-NN/architecture/federation',
                component: ComponentCreator('/Agent-NN/architecture/federation', '16e'),
                exact: true
              },
              {
                path: '/Agent-NN/architecture/feedback',
                component: ComponentCreator('/Agent-NN/architecture/feedback', 'f89'),
                exact: true
              },
              {
                path: '/Agent-NN/architecture/feedback_loop',
                component: ComponentCreator('/Agent-NN/architecture/feedback_loop', '857'),
                exact: true
              },
              {
                path: '/Agent-NN/architecture/governance',
                component: ComponentCreator('/Agent-NN/architecture/governance', 'a93'),
                exact: true
              },
              {
                path: '/Agent-NN/architecture/identity_and_signatures',
                component: ComponentCreator('/Agent-NN/architecture/identity_and_signatures', '9c0'),
                exact: true
              },
              {
                path: '/Agent-NN/architecture/mcp_components',
                component: ComponentCreator('/Agent-NN/architecture/mcp_components', '726'),
                exact: true
              },
              {
                path: '/Agent-NN/architecture/mcp_dataflow',
                component: ComponentCreator('/Agent-NN/architecture/mcp_dataflow', '203'),
                exact: true
              },
              {
                path: '/Agent-NN/architecture/mcp_stabilisierung',
                component: ComponentCreator('/Agent-NN/architecture/mcp_stabilisierung', 'cb3'),
                exact: true
              },
              {
                path: '/Agent-NN/architecture/multi_agent',
                component: ComponentCreator('/Agent-NN/architecture/multi_agent', '7c4'),
                exact: true
              },
              {
                path: '/Agent-NN/architecture/overview',
                component: ComponentCreator('/Agent-NN/architecture/overview', '86d'),
                exact: true
              },
              {
                path: '/Agent-NN/architecture/overview_mcp',
                component: ComponentCreator('/Agent-NN/architecture/overview_mcp', 'db0'),
                exact: true
              },
              {
                path: '/Agent-NN/architecture/plugin_architecture',
                component: ComponentCreator('/Agent-NN/architecture/plugin_architecture', 'ed5'),
                exact: true
              },
              {
                path: '/Agent-NN/architecture/privacy_model',
                component: ComponentCreator('/Agent-NN/architecture/privacy_model', '017'),
                exact: true
              },
              {
                path: '/Agent-NN/architecture/reputation_system',
                component: ComponentCreator('/Agent-NN/architecture/reputation_system', '669'),
                exact: true
              },
              {
                path: '/Agent-NN/architecture/resource_model',
                component: ComponentCreator('/Agent-NN/architecture/resource_model', 'a4a'),
                exact: true
              },
              {
                path: '/Agent-NN/architecture/role_capabilities',
                component: ComponentCreator('/Agent-NN/architecture/role_capabilities', '0cc'),
                exact: true
              },
              {
                path: '/Agent-NN/architecture/skill_system',
                component: ComponentCreator('/Agent-NN/architecture/skill_system', '332'),
                exact: true
              },
              {
                path: '/Agent-NN/architecture/system_architecture',
                component: ComponentCreator('/Agent-NN/architecture/system_architecture', '464'),
                exact: true
              },
              {
                path: '/Agent-NN/architecture/token_budgeting',
                component: ComponentCreator('/Agent-NN/architecture/token_budgeting', '586'),
                exact: true
              },
              {
                path: '/Agent-NN/architecture/training_paths',
                component: ComponentCreator('/Agent-NN/architecture/training_paths', '38f'),
                exact: true
              },
              {
                path: '/Agent-NN/architecture/trust_network',
                component: ComponentCreator('/Agent-NN/architecture/trust_network', '64c'),
                exact: true
              },
              {
                path: '/Agent-NN/architecture/ÜberblickSystemarchitektur',
                component: ComponentCreator('/Agent-NN/architecture/ÜberblickSystemarchitektur', 'ebb'),
                exact: true
              },
              {
                path: '/Agent-NN/architecture/voting_logic',
                component: ComponentCreator('/Agent-NN/architecture/voting_logic', '3e2'),
                exact: true
              },
              {
                path: '/Agent-NN/BenutzerHandbuch',
                component: ComponentCreator('/Agent-NN/BenutzerHandbuch', '021'),
                exact: true
              },
              {
                path: '/Agent-NN/BenutzerHandbuch/Marketing-Agent',
                component: ComponentCreator('/Agent-NN/BenutzerHandbuch/Marketing-Agent', '996'),
                exact: true
              },
              {
                path: '/Agent-NN/BenutzerHandbuch/smolitux-ui',
                component: ComponentCreator('/Agent-NN/BenutzerHandbuch/smolitux-ui', 'ee7'),
                exact: true
              },
              {
                path: '/Agent-NN/BenutzerHandbuch/uebersicht',
                component: ComponentCreator('/Agent-NN/BenutzerHandbuch/uebersicht', '4d3'),
                exact: true
              },
              {
                path: '/Agent-NN/catalog',
                component: ComponentCreator('/Agent-NN/catalog', 'b88'),
                exact: true
              },
              {
                path: '/Agent-NN/CHANGELOG',
                component: ComponentCreator('/Agent-NN/CHANGELOG', 'ee0'),
                exact: true
              },
              {
                path: '/Agent-NN/cli',
                component: ComponentCreator('/Agent-NN/cli', '228'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/Agent-NN/cli_dev',
                component: ComponentCreator('/Agent-NN/cli_dev', '8e4'),
                exact: true
              },
              {
                path: '/Agent-NN/cli_quickstart',
                component: ComponentCreator('/Agent-NN/cli_quickstart', '051'),
                exact: true
              },
              {
                path: '/Agent-NN/cli/cli_reference',
                component: ComponentCreator('/Agent-NN/cli/cli_reference', 'f5f'),
                exact: true
              },
              {
                path: '/Agent-NN/cli/CLI-Dokumentation',
                component: ComponentCreator('/Agent-NN/cli/CLI-Dokumentation', 'a09'),
                exact: true
              },
              {
                path: '/Agent-NN/config_reference',
                component: ComponentCreator('/Agent-NN/config_reference', '8fa'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/Agent-NN/context_map',
                component: ComponentCreator('/Agent-NN/context_map', 'd0e'),
                exact: true
              },
              {
                path: '/Agent-NN/deployment',
                component: ComponentCreator('/Agent-NN/deployment', '71d'),
                exact: true
              },
              {
                path: '/Agent-NN/deployment/cloud',
                component: ComponentCreator('/Agent-NN/deployment/cloud', '940'),
                exact: true
              },
              {
                path: '/Agent-NN/deployment/docker',
                component: ComponentCreator('/Agent-NN/deployment/docker', '5e6'),
                exact: true
              },
              {
                path: '/Agent-NN/deployment/kubernetes',
                component: ComponentCreator('/Agent-NN/deployment/kubernetes', '228'),
                exact: true
              },
              {
                path: '/Agent-NN/deployment/storage',
                component: ComponentCreator('/Agent-NN/deployment/storage', 'd42'),
                exact: true
              },
              {
                path: '/Agent-NN/dev_managers',
                component: ComponentCreator('/Agent-NN/dev_managers', 'd92'),
                exact: true
              },
              {
                path: '/Agent-NN/developer_setup',
                component: ComponentCreator('/Agent-NN/developer_setup', '1f8'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/Agent-NN/development/api_overview',
                component: ComponentCreator('/Agent-NN/development/api_overview', 'c4f'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/Agent-NN/development/cicd',
                component: ComponentCreator('/Agent-NN/development/cicd', '8f2'),
                exact: true
              },
              {
                path: '/Agent-NN/development/cli_usage',
                component: ComponentCreator('/Agent-NN/development/cli_usage', 'b3f'),
                exact: true
              },
              {
                path: '/Agent-NN/development/contributing',
                component: ComponentCreator('/Agent-NN/development/contributing', '97b'),
                exact: true
              },
              {
                path: '/Agent-NN/development/model_tracking',
                component: ComponentCreator('/Agent-NN/development/model_tracking', 'a1e'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/Agent-NN/development/nxlv-python',
                component: ComponentCreator('/Agent-NN/development/nxlv-python', '392'),
                exact: true
              },
              {
                path: '/Agent-NN/development/plugins',
                component: ComponentCreator('/Agent-NN/development/plugins', '85d'),
                exact: true
              },
              {
                path: '/Agent-NN/development/sdk_usage',
                component: ComponentCreator('/Agent-NN/development/sdk_usage', '458'),
                exact: true
              },
              {
                path: '/Agent-NN/development/services',
                component: ComponentCreator('/Agent-NN/development/services', '83a'),
                exact: true
              },
              {
                path: '/Agent-NN/development/setup',
                component: ComponentCreator('/Agent-NN/development/setup', 'df1'),
                exact: true
              },
              {
                path: '/Agent-NN/development/smolitux-ui-dev',
                component: ComponentCreator('/Agent-NN/development/smolitux-ui-dev', '66e'),
                exact: true
              },
              {
                path: '/Agent-NN/errors',
                component: ComponentCreator('/Agent-NN/errors', '921'),
                exact: true
              },
              {
                path: '/Agent-NN/flowise_nodes',
                component: ComponentCreator('/Agent-NN/flowise_nodes', 'd67'),
                exact: true
              },
              {
                path: '/Agent-NN/flowise_plugin',
                component: ComponentCreator('/Agent-NN/flowise_plugin', '3b6'),
                exact: true
              },
              {
                path: '/Agent-NN/flowisehub_publish',
                component: ComponentCreator('/Agent-NN/flowisehub_publish', 'fa8'),
                exact: true
              },
              {
                path: '/Agent-NN/frontend_bridge',
                component: ComponentCreator('/Agent-NN/frontend_bridge', '569'),
                exact: true
              },
              {
                path: '/Agent-NN/frontend_ux_checklist',
                component: ComponentCreator('/Agent-NN/frontend_ux_checklist', '7e5'),
                exact: true
              },
              {
                path: '/Agent-NN/frontend/overview',
                component: ComponentCreator('/Agent-NN/frontend/overview', '817'),
                exact: true
              },
              {
                path: '/Agent-NN/governance/goals',
                component: ComponentCreator('/Agent-NN/governance/goals', '7f1'),
                exact: true
              },
              {
                path: '/Agent-NN/governance/maintenance',
                component: ComponentCreator('/Agent-NN/governance/maintenance', '52a'),
                exact: true
              },
              {
                path: '/Agent-NN/governance/release_policy',
                component: ComponentCreator('/Agent-NN/governance/release_policy', '3cf'),
                exact: true
              },
              {
                path: '/Agent-NN/install',
                component: ComponentCreator('/Agent-NN/install', '52e'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/Agent-NN/integrations',
                component: ComponentCreator('/Agent-NN/integrations', '9d6'),
                exact: true
              },
              {
                path: '/Agent-NN/integrations/flowise',
                component: ComponentCreator('/Agent-NN/integrations/flowise', 'e4b'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/Agent-NN/integrations/full_integration_plan',
                component: ComponentCreator('/Agent-NN/integrations/full_integration_plan', '332'),
                exact: true
              },
              {
                path: '/Agent-NN/integrations/langchain_bridge',
                component: ComponentCreator('/Agent-NN/integrations/langchain_bridge', 'a8d'),
                exact: true
              },
              {
                path: '/Agent-NN/integrations/n8n',
                component: ComponentCreator('/Agent-NN/integrations/n8n', 'f4a'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/Agent-NN/integrations/openhands_workflows',
                component: ComponentCreator('/Agent-NN/integrations/openhands_workflows', 'b99'),
                exact: true
              },
              {
                path: '/Agent-NN/integrations/OpenHands-Flowise-Flow',
                component: ComponentCreator('/Agent-NN/integrations/OpenHands-Flowise-Flow', '08c'),
                exact: true
              },
              {
                path: '/Agent-NN/llm_provider_overview',
                component: ComponentCreator('/Agent-NN/llm_provider_overview', '0b1'),
                exact: true
              },
              {
                path: '/Agent-NN/maintenance',
                component: ComponentCreator('/Agent-NN/maintenance', '851'),
                exact: true
              },
              {
                path: '/Agent-NN/mcp',
                component: ComponentCreator('/Agent-NN/mcp', 'd11'),
                exact: true
              },
              {
                path: '/Agent-NN/metrics_reference',
                component: ComponentCreator('/Agent-NN/metrics_reference', 'af5'),
                exact: true
              },
              {
                path: '/Agent-NN/metrics_snapshot',
                component: ComponentCreator('/Agent-NN/metrics_snapshot', '82f'),
                exact: true
              },
              {
                path: '/Agent-NN/migration_status',
                component: ComponentCreator('/Agent-NN/migration_status', '12b'),
                exact: true
              },
              {
                path: '/Agent-NN/models',
                component: ComponentCreator('/Agent-NN/models', '3e4'),
                exact: true
              },
              {
                path: '/Agent-NN/observability/monitoring',
                component: ComponentCreator('/Agent-NN/observability/monitoring', 'd26'),
                exact: true
              },
              {
                path: '/Agent-NN/orchestrator',
                component: ComponentCreator('/Agent-NN/orchestrator', '9b1'),
                exact: true
              },
              {
                path: '/Agent-NN/overview',
                component: ComponentCreator('/Agent-NN/overview', 'eed'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/Agent-NN/plugins',
                component: ComponentCreator('/Agent-NN/plugins', 'fb4'),
                exact: true
              },
              {
                path: '/Agent-NN/prompting',
                component: ComponentCreator('/Agent-NN/prompting', '97e'),
                exact: true
              },
              {
                path: '/Agent-NN/realtime',
                component: ComponentCreator('/Agent-NN/realtime', 'ac5'),
                exact: true
              },
              {
                path: '/Agent-NN/reasoning',
                component: ComponentCreator('/Agent-NN/reasoning', 'ceb'),
                exact: true
              },
              {
                path: '/Agent-NN/release_checklist',
                component: ComponentCreator('/Agent-NN/release_checklist', 'e04'),
                exact: true
              },
              {
                path: '/Agent-NN/RELEASE_NOTES',
                component: ComponentCreator('/Agent-NN/RELEASE_NOTES', '637'),
                exact: true
              },
              {
                path: '/Agent-NN/releases/hotfix_template',
                component: ComponentCreator('/Agent-NN/releases/hotfix_template', '987'),
                exact: true
              },
              {
                path: '/Agent-NN/releases/v1.0.0-beta',
                component: ComponentCreator('/Agent-NN/releases/v1.0.0-beta', '640'),
                exact: true
              },
              {
                path: '/Agent-NN/releases/v1.0.2',
                component: ComponentCreator('/Agent-NN/releases/v1.0.2', '093'),
                exact: true
              },
              {
                path: '/Agent-NN/releases/v1.0.3',
                component: ComponentCreator('/Agent-NN/releases/v1.0.3', '56f'),
                exact: true
              },
              {
                path: '/Agent-NN/roadmap',
                component: ComponentCreator('/Agent-NN/roadmap', '96d'),
                exact: true
              },
              {
                path: '/Agent-NN/roles',
                component: ComponentCreator('/Agent-NN/roles', 'e2a'),
                exact: true
              },
              {
                path: '/Agent-NN/scripts',
                component: ComponentCreator('/Agent-NN/scripts', '9e9'),
                exact: true
              },
              {
                path: '/Agent-NN/sdk/python',
                component: ComponentCreator('/Agent-NN/sdk/python', '7ea'),
                exact: true
              },
              {
                path: '/Agent-NN/security/authentication',
                component: ComponentCreator('/Agent-NN/security/authentication', 'e9d'),
                exact: true
              },
              {
                path: '/Agent-NN/security/hardening',
                component: ComponentCreator('/Agent-NN/security/hardening', '08b'),
                exact: true
              },
              {
                path: '/Agent-NN/security/mcp_security',
                component: ComponentCreator('/Agent-NN/security/mcp_security', '131'),
                exact: true
              },
              {
                path: '/Agent-NN/security/scopes',
                component: ComponentCreator('/Agent-NN/security/scopes', 'c58'),
                exact: true
              },
              {
                path: '/Agent-NN/service_audit',
                component: ComponentCreator('/Agent-NN/service_audit', 'be6'),
                exact: true
              },
              {
                path: '/Agent-NN/sessions',
                component: ComponentCreator('/Agent-NN/sessions', 'ca7'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/Agent-NN/setup',
                component: ComponentCreator('/Agent-NN/setup', '75b'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/Agent-NN/smolitux-ui-integration',
                component: ComponentCreator('/Agent-NN/smolitux-ui-integration', 'db1'),
                exact: true
              },
              {
                path: '/Agent-NN/snapshots',
                component: ComponentCreator('/Agent-NN/snapshots', 'd79'),
                exact: true
              },
              {
                path: '/Agent-NN/test_strategy',
                component: ComponentCreator('/Agent-NN/test_strategy', 'd6b'),
                exact: true
              },
              {
                path: '/Agent-NN/tools',
                component: ComponentCreator('/Agent-NN/tools', '1cb'),
                exact: true
              },
              {
                path: '/Agent-NN/troubleshooting/troubleshooting_guide',
                component: ComponentCreator('/Agent-NN/troubleshooting/troubleshooting_guide', '49f'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/Agent-NN/ui_migration_audit',
                component: ComponentCreator('/Agent-NN/ui_migration_audit', 'c7f'),
                exact: true
              },
              {
                path: '/Agent-NN/use-cases/adaptive_agent_behavior',
                component: ComponentCreator('/Agent-NN/use-cases/adaptive_agent_behavior', 'e29'),
                exact: true
              },
              {
                path: '/Agent-NN/use-cases/adaptive_agents',
                component: ComponentCreator('/Agent-NN/use-cases/adaptive_agents', 'bbf'),
                exact: true
              },
              {
                path: '/Agent-NN/use-cases/adaptive_self_optimizing_agents',
                component: ComponentCreator('/Agent-NN/use-cases/adaptive_self_optimizing_agents', 'eb1'),
                exact: true
              },
              {
                path: '/Agent-NN/use-cases/budget_constrained_execution',
                component: ComponentCreator('/Agent-NN/use-cases/budget_constrained_execution', 'e7b'),
                exact: true
              },
              {
                path: '/Agent-NN/use-cases/collaborative_responsibility',
                component: ComponentCreator('/Agent-NN/use-cases/collaborative_responsibility', 'e15'),
                exact: true
              },
              {
                path: '/Agent-NN/use-cases/collective_learning',
                component: ComponentCreator('/Agent-NN/use-cases/collective_learning', '1c3'),
                exact: true
              },
              {
                path: '/Agent-NN/use-cases/community_quality_feedback',
                component: ComponentCreator('/Agent-NN/use-cases/community_quality_feedback', 'ea2'),
                exact: true
              },
              {
                path: '/Agent-NN/use-cases/community_validation',
                component: ComponentCreator('/Agent-NN/use-cases/community_validation', 'd8f'),
                exact: true
              },
              {
                path: '/Agent-NN/use-cases/contextual_interaction',
                component: ComponentCreator('/Agent-NN/use-cases/contextual_interaction', 'da5'),
                exact: true
              },
              {
                path: '/Agent-NN/use-cases/contract_enforced_execution',
                component: ComponentCreator('/Agent-NN/use-cases/contract_enforced_execution', 'a09'),
                exact: true
              },
              {
                path: '/Agent-NN/use-cases/distributed_teamwork',
                component: ComponentCreator('/Agent-NN/use-cases/distributed_teamwork', '74a'),
                exact: true
              },
              {
                path: '/Agent-NN/use-cases/efficient_team_assignment',
                component: ComponentCreator('/Agent-NN/use-cases/efficient_team_assignment', 'd7b'),
                exact: true
              },
              {
                path: '/Agent-NN/use-cases/examples',
                component: ComponentCreator('/Agent-NN/use-cases/examples', '739'),
                exact: true
              },
              {
                path: '/Agent-NN/use-cases/interactive_session',
                component: ComponentCreator('/Agent-NN/use-cases/interactive_session', '33a'),
                exact: true
              },
              {
                path: '/Agent-NN/use-cases/learning_agents',
                component: ComponentCreator('/Agent-NN/use-cases/learning_agents', '843'),
                exact: true
              },
              {
                path: '/Agent-NN/use-cases/liability_compliance',
                component: ComponentCreator('/Agent-NN/use-cases/liability_compliance', 'dc5'),
                exact: true
              },
              {
                path: '/Agent-NN/use-cases/multi-agent',
                component: ComponentCreator('/Agent-NN/use-cases/multi-agent', 'fab'),
                exact: true
              },
              {
                path: '/Agent-NN/use-cases/privacy_preserving_tasks',
                component: ComponentCreator('/Agent-NN/use-cases/privacy_preserving_tasks', '971'),
                exact: true
              },
              {
                path: '/Agent-NN/use-cases/progressive_agent_specialization',
                component: ComponentCreator('/Agent-NN/use-cases/progressive_agent_specialization', '5ec'),
                exact: true
              },
              {
                path: '/Agent-NN/use-cases/project_based_learning',
                component: ComponentCreator('/Agent-NN/use-cases/project_based_learning', 'ecb'),
                exact: true
              },
              {
                path: '/Agent-NN/use-cases/role_limited_task_execution',
                component: ComponentCreator('/Agent-NN/use-cases/role_limited_task_execution', 'f01'),
                exact: true
              },
              {
                path: '/Agent-NN/use-cases/role_restricted_tasks',
                component: ComponentCreator('/Agent-NN/use-cases/role_restricted_tasks', '062'),
                exact: true
              },
              {
                path: '/Agent-NN/use-cases/signed_agent_decisions',
                component: ComponentCreator('/Agent-NN/use-cases/signed_agent_decisions', '4da'),
                exact: true
              },
              {
                path: '/Agent-NN/use-cases/skill_matched_execution',
                component: ComponentCreator('/Agent-NN/use-cases/skill_matched_execution', 'ad4'),
                exact: true
              },
              {
                path: '/Agent-NN/use-cases/time_constrained_tasks',
                component: ComponentCreator('/Agent-NN/use-cases/time_constrained_tasks', '175'),
                exact: true
              },
              {
                path: '/Agent-NN/use-cases/trust_driven_authorization',
                component: ComponentCreator('/Agent-NN/use-cases/trust_driven_authorization', '1d2'),
                exact: true
              },
              {
                path: '/Agent-NN/use-cases/voting',
                component: ComponentCreator('/Agent-NN/use-cases/voting', 'c2a'),
                exact: true
              },
              {
                path: '/Agent-NN/Wiki/Finanzierung/Budget&Ressourcen',
                component: ComponentCreator('/Agent-NN/Wiki/Finanzierung/Budget&Ressourcen', 'fc3'),
                exact: true
              },
              {
                path: '/Agent-NN/Wiki/Finanzierung/Budgetplanung&Meilensteinen',
                component: ComponentCreator('/Agent-NN/Wiki/Finanzierung/Budgetplanung&Meilensteinen', '8c7'),
                exact: true
              },
              {
                path: '/Agent-NN/',
                component: ComponentCreator('/Agent-NN/', 'dda'),
                exact: true
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
