"""Specialized domain knowledge management."""
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import networkx as nx
from rich.console import Console
from rich.table import Table

from utils.logging_util import setup_logger
from datastores.worker_agent_db import WorkerAgentDB

logger = setup_logger(__name__)
console = Console()

@dataclass
class KnowledgeNode:
    """Node in the knowledge graph."""
    id: str
    domain: str
    content: str
    metadata: Dict[str, Any]
    connections: Set[str] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.connections is None:
            self.connections = set()
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary."""
        return {
            "id": self.id,
            "domain": self.domain,
            "content": self.content,
            "metadata": self.metadata,
            "connections": list(self.connections),
            "timestamp": self.timestamp
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeNode':
        """Create node from dictionary."""
        return cls(
            id=data["id"],
            domain=data["domain"],
            content=data["content"],
            metadata=data["metadata"],
            connections=set(data["connections"]),
            timestamp=data["timestamp"]
        )

class DomainKnowledgeManager:
    """Manager for specialized domain knowledge."""
    
    def __init__(self, knowledge_dir: str = "data/domain_knowledge"):
        """Initialize domain knowledge manager.
        
        Args:
            knowledge_dir: Directory for knowledge storage
        """
        self.knowledge_dir = Path(knowledge_dir)
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        
        # Knowledge graph
        self.graph = nx.Graph()
        
        # Domain-specific databases
        self.domain_dbs: Dict[str, WorkerAgentDB] = {}
        
        # Load existing knowledge
        self._load_knowledge()
        
    def add_knowledge(self,
                     domain: str,
                     content: str,
                     metadata: Dict[str, Any] = None,
                     connections: List[str] = None):
        """Add new knowledge to a domain.
        
        Args:
            domain: Knowledge domain
            content: Knowledge content
            metadata: Optional metadata
            connections: Optional connections to other nodes
        """
        # Create node
        node_id = f"{domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        node = KnowledgeNode(
            id=node_id,
            domain=domain,
            content=content,
            metadata=metadata or {},
            connections=set(connections or [])
        )
        
        # Add to graph
        self.graph.add_node(node_id, data=node)
        
        # Add connections
        if connections:
            for conn in connections:
                if self.graph.has_node(conn):
                    self.graph.add_edge(node_id, conn)
                    
        # Update domain database
        self._get_domain_db(domain).ingest_documents(
            [content],
            {"node_id": node_id, **node.metadata}
        )
        
        # Save knowledge
        self._save_knowledge()
        
        logger.info(f"Added knowledge node: {node_id}")
        
    def get_knowledge(self, node_id: str) -> Optional[KnowledgeNode]:
        """Get knowledge node by ID.
        
        Args:
            node_id: ID of the node
            
        Returns:
            KnowledgeNode if found, None otherwise
        """
        if not self.graph.has_node(node_id):
            return None
            
        return self.graph.nodes[node_id]["data"]
        
    def search_knowledge(self,
                        query: str,
                        domain: Optional[str] = None,
                        limit: int = 5) -> List[KnowledgeNode]:
        """Search knowledge nodes.
        
        Args:
            query: Search query
            domain: Optional domain to search in
            limit: Maximum number of results
            
        Returns:
            List of matching knowledge nodes
        """
        if domain:
            # Search in specific domain
            db = self._get_domain_db(domain)
            docs = db.search(query, limit)
            
            # Convert to nodes
            nodes = []
            for doc in docs:
                node_id = doc.metadata.get("node_id")
                if node_id and self.graph.has_node(node_id):
                    nodes.append(self.graph.nodes[node_id]["data"])
            return nodes
            
        else:
            # Search across all domains
            results = []
            for domain_db in self.domain_dbs.values():
                docs = domain_db.search(query, limit)
                for doc in docs:
                    node_id = doc.metadata.get("node_id")
                    if node_id and self.graph.has_node(node_id):
                        results.append(self.graph.nodes[node_id]["data"])
                        if len(results) >= limit:
                            return results
            return results
            
    def get_related_knowledge(self,
                            node_id: str,
                            max_depth: int = 2) -> List[KnowledgeNode]:
        """Get knowledge related to a node.
        
        Args:
            node_id: ID of the starting node
            max_depth: Maximum connection depth to explore
            
        Returns:
            List of related knowledge nodes
        """
        if not self.graph.has_node(node_id):
            return []
            
        # Get nodes within max_depth
        related_nodes = set()
        current_nodes = {node_id}
        
        for _ in range(max_depth):
            next_nodes = set()
            for node in current_nodes:
                neighbors = set(self.graph.neighbors(node))
                next_nodes.update(neighbors - related_nodes)
            related_nodes.update(current_nodes)
            current_nodes = next_nodes
            if not current_nodes:
                break
                
        # Convert to nodes
        return [
            self.graph.nodes[n]["data"]
            for n in related_nodes
            if n != node_id
        ]
        
    def merge_knowledge(self, node_ids: List[str], new_content: str):
        """Merge multiple knowledge nodes.
        
        Args:
            node_ids: IDs of nodes to merge
            new_content: Content for merged node
        """
        if not all(self.graph.has_node(nid) for nid in node_ids):
            raise ValueError("All nodes must exist")
            
        # Get nodes
        nodes = [self.graph.nodes[nid]["data"] for nid in node_ids]
        
        # Create merged node
        domain = nodes[0].domain
        if not all(n.domain == domain for n in nodes):
            raise ValueError("All nodes must be from same domain")
            
        # Combine metadata
        merged_metadata = {}
        for node in nodes:
            merged_metadata.update(node.metadata)
            
        # Get all connections
        connections = set()
        for node in nodes:
            connections.update(node.connections)
            
        # Create new node
        self.add_knowledge(
            domain=domain,
            content=new_content,
            metadata={
                "merged_from": node_ids,
                **merged_metadata
            },
            connections=list(connections - set(node_ids))
        )
        
        # Remove old nodes
        for node_id in node_ids:
            self.remove_knowledge(node_id)
            
    def remove_knowledge(self, node_id: str):
        """Remove a knowledge node.
        
        Args:
            node_id: ID of node to remove
        """
        if not self.graph.has_node(node_id):
            return
            
        # Get node
        node = self.graph.nodes[node_id]["data"]
        
        # Remove from graph
        self.graph.remove_node(node_id)
        
        # Remove from domain database
        db = self._get_domain_db(node.domain)
        # Note: This is a simplification. In practice, you'd need a way to
        # remove specific documents from the vector store
        
        # Save knowledge
        self._save_knowledge()
        
        logger.info(f"Removed knowledge node: {node_id}")
        
    def show_domain_stats(self):
        """Show statistics about domain knowledge."""
        # Count nodes by domain
        domain_counts = {}
        for node in self.graph.nodes.values():
            domain = node["data"].domain
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            
        # Create table
        table = Table(title="Domain Knowledge Statistics")
        table.add_column("Domain")
        table.add_column("Nodes")
        table.add_column("Connections")
        
        for domain, count in domain_counts.items():
            # Count connections within domain
            domain_nodes = [
                n for n in self.graph.nodes
                if self.graph.nodes[n]["data"].domain == domain
            ]
            connections = sum(
                1 for u, v in self.graph.edges
                if u in domain_nodes or v in domain_nodes
            )
            
            table.add_row(domain, str(count), str(connections))
            
        console.print(table)
        
    def _get_domain_db(self, domain: str) -> WorkerAgentDB:
        """Get or create domain database.
        
        Args:
            domain: Domain name
            
        Returns:
            WorkerAgentDB for the domain
        """
        if domain not in self.domain_dbs:
            self.domain_dbs[domain] = WorkerAgentDB(domain)
        return self.domain_dbs[domain]
        
    def _load_knowledge(self):
        """Load knowledge from disk."""
        knowledge_file = self.knowledge_dir / "knowledge_graph.json"
        if not knowledge_file.exists():
            return
            
        try:
            with open(knowledge_file, 'r') as f:
                data = json.load(f)
                
            # Recreate graph
            self.graph.clear()
            for node_data in data["nodes"]:
                node = KnowledgeNode.from_dict(node_data)
                self.graph.add_node(node.id, data=node)
                
            for edge in data["edges"]:
                self.graph.add_edge(edge[0], edge[1])
                
            logger.info("Loaded knowledge graph")
            
        except Exception as e:
            logger.error(f"Error loading knowledge: {str(e)}")
            
    def _save_knowledge(self):
        """Save knowledge to disk."""
        try:
            data = {
                "nodes": [
                    self.graph.nodes[n]["data"].to_dict()
                    for n in self.graph.nodes
                ],
                "edges": list(self.graph.edges)
            }
            
            with open(self.knowledge_dir / "knowledge_graph.json", 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info("Saved knowledge graph")
            
        except Exception as e:
            logger.error(f"Error saving knowledge: {str(e)}")
            
    def visualize_graph(self, output_file: Optional[str] = None):
        """Visualize the knowledge graph.
        
        Args:
            output_file: Optional file to save visualization
        """
        try:
            import matplotlib.pyplot as plt
            
            # Create position layout
            pos = nx.spring_layout(self.graph)
            
            # Draw graph
            plt.figure(figsize=(12, 8))
            
            # Draw nodes
            colors = []
            labels = {}
            for node in self.graph.nodes:
                domain = self.graph.nodes[node]["data"].domain
                colors.append(hash(domain) % 20)  # Simple domain-based coloring
                labels[node] = f"{domain}\n{node}"
                
            nx.draw_networkx_nodes(
                self.graph, pos,
                node_color=colors,
                node_size=1000,
                alpha=0.7
            )
            
            # Draw edges
            nx.draw_networkx_edges(
                self.graph, pos,
                edge_color='gray',
                alpha=0.5
            )
            
            # Draw labels
            nx.draw_networkx_labels(
                self.graph, pos,
                labels=labels,
                font_size=8
            )
            
            plt.title("Knowledge Graph")
            
            if output_file:
                plt.savefig(output_file)
            else:
                plt.show()
                
        except ImportError:
            logger.error("matplotlib is required for visualization")