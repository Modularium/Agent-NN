"""Dashboard for monitoring agent system."""
import os
import asyncio
from typing import Dict, Any, List
from datetime import datetime, timedelta
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output
from dash.exceptions import PreventUpdate

from agents.agent_generator import AgentGenerator
from agents.agent_communication import AgentCommunicationHub
from agents.domain_knowledge import DomainKnowledgeManager
from utils.logging_util import setup_logger

logger = setup_logger(__name__)

class AgentDashboard:
    """Dashboard for monitoring agent system."""
    
    def __init__(self,
                 generator: AgentGenerator,
                 update_interval: int = 5):
        """Initialize dashboard.
        
        Args:
            generator: Agent generator instance
            update_interval: Data update interval in seconds
        """
        self.generator = generator
        self.update_interval = update_interval
        
        # Initialize Dash app
        self.app = Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
        
        # Data storage
        self.performance_data: Dict[str, List[Dict[str, Any]]] = {}
        self.creation_history: List[Dict[str, Any]] = []
        self.message_stats: List[Dict[str, Any]] = []
        
    def setup_layout(self):
        """Set up dashboard layout."""
        self.app.layout = html.Div([
            html.H1("Agent System Dashboard"),
            
            # Tabs for different views
            dcc.Tabs([
                # Overview tab
                dcc.Tab(label="System Overview", children=[
                    html.Div([
                        html.H3("Active Agents"),
                        dcc.Graph(id="active-agents-graph"),
                        
                        html.H3("System Performance"),
                        dcc.Graph(id="system-performance-graph"),
                        
                        html.H3("Recent Activity"),
                        html.Div(id="activity-feed")
                    ])
                ]),
                
                # Agent details tab
                dcc.Tab(label="Agent Details", children=[
                    html.Div([
                        html.H3("Agent Selection"),
                        dcc.Dropdown(id="agent-selector"),
                        
                        html.H3("Agent Performance"),
                        dcc.Graph(id="agent-performance-graph"),
                        
                        html.H3("Agent Capabilities"),
                        html.Div(id="agent-capabilities")
                    ])
                ]),
                
                # Communication tab
                dcc.Tab(label="Communication", children=[
                    html.Div([
                        html.H3("Message Flow"),
                        dcc.Graph(id="message-flow-graph"),
                        
                        html.H3("Message Types"),
                        dcc.Graph(id="message-types-graph")
                    ])
                ]),
                
                # Knowledge tab
                dcc.Tab(label="Knowledge Base", children=[
                    html.Div([
                        html.H3("Knowledge Distribution"),
                        dcc.Graph(id="knowledge-distribution-graph"),
                        
                        html.H3("Recent Updates"),
                        html.Div(id="knowledge-updates")
                    ])
                ])
            ]),
            
            # Hidden div for storing data
            html.Div(id="data-store", style={"display": "none"}),
            
            # Update interval
            dcc.Interval(
                id="interval-component",
                interval=self.update_interval * 1000,
                n_intervals=0
            )
        ])
        
    def setup_callbacks(self):
        """Set up dashboard callbacks."""
        @self.app.callback(
            [Output("data-store", "children"),
             Output("active-agents-graph", "figure"),
             Output("system-performance-graph", "figure"),
             Output("activity-feed", "children")],
            [Input("interval-component", "n_intervals")]
        )
        def update_data(n):
            """Update dashboard data."""
            # Get current data
            data = self._gather_current_data()
            
            # Create active agents graph
            active_agents = self._create_active_agents_graph(data)
            
            # Create system performance graph
            performance = self._create_system_performance_graph(data)
            
            # Create activity feed
            activity = self._create_activity_feed(data)
            
            return json.dumps(data), active_agents, performance, activity
            
        @self.app.callback(
            [Output("agent-selector", "options"),
             Output("agent-performance-graph", "figure"),
             Output("agent-capabilities", "children")],
            [Input("data-store", "children"),
             Input("agent-selector", "value")]
        )
        def update_agent_details(data_json, selected_agent):
            """Update agent details view."""
            if not data_json or not selected_agent:
                raise PreventUpdate
                
            data = json.loads(data_json)
            
            # Update agent selector options
            options = [
                {"label": agent, "value": agent}
                for agent in data["active_agents"]
            ]
            
            # Create agent performance graph
            performance = self._create_agent_performance_graph(
                data,
                selected_agent
            )
            
            # Create capabilities view
            capabilities = self._create_capabilities_view(
                data,
                selected_agent
            )
            
            return options, performance, capabilities
            
        @self.app.callback(
            [Output("message-flow-graph", "figure"),
             Output("message-types-graph", "figure")],
            [Input("data-store", "children")]
        )
        def update_communication_view(data_json):
            """Update communication view."""
            if not data_json:
                raise PreventUpdate
                
            data = json.loads(data_json)
            
            # Create message flow graph
            flow = self._create_message_flow_graph(data)
            
            # Create message types graph
            types = self._create_message_types_graph(data)
            
            return flow, types
            
        @self.app.callback(
            [Output("knowledge-distribution-graph", "figure"),
             Output("knowledge-updates", "children")],
            [Input("data-store", "children")]
        )
        def update_knowledge_view(data_json):
            """Update knowledge base view."""
            if not data_json:
                raise PreventUpdate
                
            data = json.loads(data_json)
            
            # Create knowledge distribution graph
            distribution = self._create_knowledge_distribution_graph(data)
            
            # Create updates view
            updates = self._create_knowledge_updates_view(data)
            
            return distribution, updates
            
    def _gather_current_data(self) -> Dict[str, Any]:
        """Gather current system data.
        
        Returns:
            Dict containing current system state
        """
        data = {
            "timestamp": datetime.now().isoformat(),
            "active_agents": list(self.generator.factory.agents.keys()),
            "agent_specs": {
                name: spec.__dict__
                for name, spec in self.generator.factory.agent_specs.items()
            },
            "creation_requests": self.generator.creation_requests,
            "performance_data": self.performance_data,
            "message_stats": self.message_stats
        }
        
        return data
        
    def _create_active_agents_graph(self, data: Dict[str, Any]) -> go.Figure:
        """Create active agents graph.
        
        Args:
            data: Current system data
            
        Returns:
            Plotly figure
        """
        df = pd.DataFrame([
            {
                "agent": name,
                "domain": spec["domain"],
                "capabilities": len(spec["capabilities"])
            }
            for name, spec in data["agent_specs"].items()
        ])
        
        fig = px.bar(
            df,
            x="agent",
            y="capabilities",
            color="domain",
            title="Active Agents and Capabilities"
        )
        
        return fig
        
    def _create_system_performance_graph(self,
                                       data: Dict[str, Any]) -> go.Figure:
        """Create system performance graph.
        
        Args:
            data: Current system data
            
        Returns:
            Plotly figure
        """
        # Convert performance data to DataFrame
        performance_records = []
        for agent, records in data["performance_data"].items():
            for record in records:
                performance_records.append({
                    "agent": agent,
                    "timestamp": record["timestamp"],
                    "success_rate": record["success_rate"],
                    "response_time": record["response_time"]
                })
                
        df = pd.DataFrame(performance_records)
        
        fig = px.line(
            df,
            x="timestamp",
            y="success_rate",
            color="agent",
            title="System Performance Over Time"
        )
        
        return fig
        
    def _create_activity_feed(self, data: Dict[str, Any]) -> List[html.Div]:
        """Create activity feed.
        
        Args:
            data: Current system data
            
        Returns:
            List of Dash components
        """
        activities = []
        
        # Add creation requests
        for req_id, req_data in data["creation_requests"].items():
            activities.append({
                "timestamp": req_data["timestamp"],
                "type": "creation",
                "content": f"Agent creation request from {req_data['requester']}"
            })
            
        # Sort by timestamp
        activities.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # Create components
        return [
            html.Div([
                html.Strong(f"{act['type'].title()}: "),
                html.Span(act['content']),
                html.Br(),
                html.Small(act['timestamp'])
            ])
            for act in activities[:10]  # Show last 10 activities
        ]
        
    def _create_agent_performance_graph(self,
                                      data: Dict[str, Any],
                                      agent: str) -> go.Figure:
        """Create agent performance graph.
        
        Args:
            data: Current system data
            agent: Selected agent name
            
        Returns:
            Plotly figure
        """
        if agent not in data["performance_data"]:
            return go.Figure()
            
        records = data["performance_data"][agent]
        df = pd.DataFrame(records)
        
        fig = px.line(
            df,
            x="timestamp",
            y=["success_rate", "response_time"],
            title=f"Performance Metrics for {agent}"
        )
        
        return fig
        
    def _create_capabilities_view(self,
                                data: Dict[str, Any],
                                agent: str) -> List[html.Div]:
        """Create capabilities view.
        
        Args:
            data: Current system data
            agent: Selected agent name
            
        Returns:
            List of Dash components
        """
        if agent not in data["agent_specs"]:
            return []
            
        spec = data["agent_specs"][agent]
        
        return [
            html.Div([
                html.H4("Capabilities"),
                html.Ul([
                    html.Li(cap) for cap in spec["capabilities"]
                ]),
                
                html.H4("Knowledge Requirements"),
                html.Ul([
                    html.Li(req) for req in spec["knowledge_requirements"]
                ]),
                
                html.H4("Tools"),
                html.Ul([
                    html.Li(tool) for tool in spec["specialized_tools"]
                ])
            ])
        ]
        
    def _create_message_flow_graph(self, data: Dict[str, Any]) -> go.Figure:
        """Create message flow graph.
        
        Args:
            data: Current system data
            
        Returns:
            Plotly figure
        """
        # Create network graph
        message_flows = []
        for stat in data["message_stats"]:
            message_flows.append({
                "from": stat["sender"],
                "to": stat["receiver"],
                "count": stat["count"]
            })
            
        df = pd.DataFrame(message_flows)
        
        fig = go.Figure(data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=list(set(df["from"].unique()) | set(df["to"].unique()))
                ),
                link=dict(
                    source=df["from"],
                    target=df["to"],
                    value=df["count"]
                )
            )
        ])
        
        fig.update_layout(title="Message Flow Between Agents")
        return fig
        
    def _create_message_types_graph(self, data: Dict[str, Any]) -> go.Figure:
        """Create message types graph.
        
        Args:
            data: Current system data
            
        Returns:
            Plotly figure
        """
        message_types = []
        for stat in data["message_stats"]:
            message_types.append({
                "type": stat["message_type"],
                "count": stat["count"]
            })
            
        df = pd.DataFrame(message_types)
        
        fig = px.pie(
            df,
            values="count",
            names="type",
            title="Message Types Distribution"
        )
        
        return fig
        
    def _create_knowledge_distribution_graph(self,
                                           data: Dict[str, Any]) -> go.Figure:
        """Create knowledge distribution graph.
        
        Args:
            data: Current system data
            
        Returns:
            Plotly figure
        """
        knowledge_dist = []
        for name, spec in data["agent_specs"].items():
            knowledge_dist.append({
                "agent": name,
                "domain": spec["domain"],
                "knowledge": len(spec["knowledge_requirements"])
            })
            
        df = pd.DataFrame(knowledge_dist)
        
        fig = px.treemap(
            df,
            path=["domain", "agent"],
            values="knowledge",
            title="Knowledge Distribution"
        )
        
        return fig
        
    def _create_knowledge_updates_view(self,
                                     data: Dict[str, Any]) -> List[html.Div]:
        """Create knowledge updates view.
        
        Args:
            data: Current system data
            
        Returns:
            List of Dash components
        """
        updates = []
        for name, spec in data["agent_specs"].items():
            if "updated_at" in spec["metadata"]:
                updates.append({
                    "agent": name,
                    "timestamp": spec["metadata"]["updated_at"],
                    "type": "Knowledge Update"
                })
                
        # Sort by timestamp
        updates.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return [
            html.Div([
                html.Strong(f"{update['type']}: "),
                html.Span(f"{update['agent']}"),
                html.Br(),
                html.Small(update['timestamp'])
            ])
            for update in updates[:10]  # Show last 10 updates
        ]
        
    async def update_performance_data(self):
        """Update performance data periodically."""
        while True:
            try:
                # Get performance data for each agent
                for name, agent in self.generator.factory.agents.items():
                    performance = await self.generator._analyze_agent_performance(
                        agent
                    )
                    
                    if name not in self.performance_data:
                        self.performance_data[name] = []
                        
                    self.performance_data[name].append({
                        "timestamp": datetime.now().isoformat(),
                        **performance
                    })
                    
                    # Keep last 100 records
                    self.performance_data[name] = \
                        self.performance_data[name][-100:]
                        
            except Exception as e:
                logger.error(f"Error updating performance data: {str(e)}")
                
            await asyncio.sleep(self.update_interval)
            
    async def update_message_stats(self):
        """Update message statistics periodically."""
        while True:
            try:
                # Get message statistics from communication hub
                if self.generator.communication_hub:
                    stats = []
                    for msg in self.generator.communication_hub.message_history:
                        stats.append({
                            "sender": msg.sender,
                            "receiver": msg.receiver,
                            "message_type": msg.message_type.value,
                            "timestamp": msg.timestamp
                        })
                        
                    # Group by sender, receiver, and type
                    from collections import defaultdict
                    grouped_stats = defaultdict(int)
                    
                    for stat in stats:
                        key = (stat["sender"], stat["receiver"],
                              stat["message_type"])
                        grouped_stats[key] += 1
                        
                    # Convert to list
                    self.message_stats = [
                        {
                            "sender": k[0],
                            "receiver": k[1],
                            "message_type": k[2],
                            "count": v
                        }
                        for k, v in grouped_stats.items()
                    ]
                    
            except Exception as e:
                logger.error(f"Error updating message stats: {str(e)}")
                
            await asyncio.sleep(self.update_interval)
            
    def run(self, host: str = "localhost", port: int = 8050):
        """Run the dashboard.
        
        Args:
            host: Host to run on
            port: Port to run on
        """
        # Start update tasks
        asyncio.create_task(self.update_performance_data())
        asyncio.create_task(self.update_message_stats())
        
        # Run dashboard
        self.app.run_server(host=host, port=port)