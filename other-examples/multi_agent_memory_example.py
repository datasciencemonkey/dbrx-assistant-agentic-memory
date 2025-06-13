"""
Multi-Agent Memory Sharing with PostgreSQL
==========================================

This example demonstrates multiple specialized agents sharing memory infrastructure using PostgreSQL.
Features:
- Shared memory database across multiple agents
- Specialized agents (Financial Analyst, Research Assistant, Team Coordinator)
- Cross-agent information sharing
- Collaborative analysis and decision making

Prerequisites:
- PostgreSQL database running
- Database created (e.g., CREATE DATABASE agno_memory;)
- Connection string in POSTGRES_URL environment variable
"""

import os
from typing import Optional

# Core Agno imports
from agno.agent import Agent
from agno.models.openai import OpenAIChat

# Memory and Storage imports
from agno.memory.v2.db.postgres import PostgresMemoryDb
from agno.memory.v2.memory import Memory

# Tool imports
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agno.tools.reasoning import ReasoningTools
from dotenv import load_dotenv

# For pretty printing
from rich.console import Console
from rich.panel import Panel

console = Console()

load_dotenv()


class PostgresConfig:
    """PostgreSQL configuration helper."""
    
    @staticmethod
    def get_db_url():
        """Get PostgreSQL connection URL from environment or use default."""
        if os.getenv("POSTGRES_URL"):
            return os.getenv("POSTGRES_URL")
        
        # Build from components
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = os.getenv("POSTGRES_PORT", "5432")
        user = os.getenv("POSTGRES_USER", "postgres")
        password = os.getenv("POSTGRES_PASSWORD", "")
        database = os.getenv("POSTGRES_DB", "agno_memory")
        
        if password:
            return f"postgresql://{user}:{password}@{host}:{port}/{database}"
        else:
            return f"postgresql://{user}@{host}:{port}/{database}"


class MultiAgentTeam:
    """
    A team of specialized agents sharing memory infrastructure in PostgreSQL.
    """
    
    def __init__(self, team_id: str = "default_team"):
        self.team_id = team_id
        
        # Get PostgreSQL connection
        db_url = PostgresConfig.get_db_url()
        
        console.print("[cyan]Setting up shared memory infrastructure...[/cyan]")
        
        try:
            # Shared memory database using PostgreSQL
            self.shared_memory_db = PostgresMemoryDb(
                table_name="shared_team_memories",
                db_url=db_url
            )
            self.shared_memory = Memory(db=self.shared_memory_db)
            
            # Create specialized agents
            self.financial_agent = self._create_financial_agent()
            self.research_agent = self._create_research_agent()
            self.coordinator = self._create_coordinator()
            
            console.print("[green]✓ Multi-agent team setup complete[/green]")
            
        except Exception as e:
            console.print(f"[red]✗ Failed to setup multi-agent team: {str(e)}[/red]")
            raise
    
    def _create_financial_agent(self) -> Agent:
        """Create a specialized financial analyst agent."""
        
        return Agent(
            name="Financial Analyst",
            model=OpenAIChat(id="gpt-4"),
            memory=self.shared_memory,
            enable_agentic_memory=True,
            tools=[
                YFinanceTools(
                    stock_price=True, 
                    analyst_recommendations=True,
                    company_info=True,
                    stock_fundamentals=True
                ),
                ReasoningTools(add_instructions=True)
            ],
            instructions=[
                "You are a financial analyst specializing in stock market analysis.",
                "Store important financial insights, trends, and recommendations in shared memory.",
                "Focus on quantitative analysis, financial metrics, and market trends.",
                "Always provide data-driven insights with specific numbers and ratios.",
                "When storing memories, tag them as 'financial-analysis' for easy retrieval.",
            ],
            markdown=True,
            show_tool_calls=True,
        )
    
    def _create_research_agent(self) -> Agent:
        """Create a specialized research assistant agent."""
        
        return Agent(
            name="Research Assistant",
            model=OpenAIChat(id="gpt-4"),
            memory=self.shared_memory,
            enable_agentic_memory=True,
            tools=[
                DuckDuckGoTools(),
                ReasoningTools(add_instructions=True)
            ],
            instructions=[
                "You are a research assistant specializing in market research and news analysis.",
                "Store important research findings, news insights, and market sentiment in shared memory.",
                "Focus on qualitative analysis, industry trends, and competitive landscape.",
                "Provide context and background information to support decision making.",
                "When storing memories, tag them as 'market-research' for easy retrieval.",
            ],
            markdown=True,
            show_tool_calls=True,
        )
    
    def _create_coordinator(self) -> Agent:
        """Create a team coordinator agent."""
        
        return Agent(
            name="Team Coordinator",
            model=OpenAIChat(id="gpt-4"),
            team=[self.financial_agent, self.research_agent],
            memory=self.shared_memory,
            enable_agentic_memory=True,
            tools=[ReasoningTools(add_instructions=True)],
            instructions=[
                "You coordinate between financial and research agents to provide comprehensive analysis.",
                "Synthesize insights from both quantitative (financial) and qualitative (research) perspectives.",
                "Store key conclusions and strategic recommendations in shared memory.",
                "Delegate specific tasks to appropriate team members based on their expertise.",
                "When storing memories, tag them as 'strategic-analysis' for easy retrieval.",
                "Always provide a balanced view incorporating both financial data and market research.",
            ],
            markdown=True,
            show_tool_calls=True,
        )
    
    def analyze_investment(self, query: str, user_id: str = "team_user") -> str:
        """
        Perform comprehensive investment analysis using the team of agents.
        
        Args:
            query: Investment analysis request
            user_id: User identifier for memory management
            
        Returns:
            Coordinated analysis from the team
        """
        console.print(f"[bold green]Team Analysis Request:[/bold green] {query}")
        console.print("\n" + "="*60 + "\n")
        
        try:
            # Use coordinator to manage the team analysis
            response = self.coordinator.print_response(
                query,
                user_id=user_id,
                stream=True
            )
            
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            console.print(f"[red]Error in team analysis: {str(e)}[/red]")
            return f"Team analysis encountered an error: {str(e)}"
    
    def individual_agent_analysis(self, query: str, agent_type: str = "financial", user_id: str = "team_user"):
        """
        Get analysis from a specific agent.
        
        Args:
            query: Analysis request
            agent_type: 'financial', 'research', or 'coordinator'
            user_id: User identifier
        """
        agent_map = {
            "financial": self.financial_agent,
            "research": self.research_agent,
            "coordinator": self.coordinator
        }
        
        agent = agent_map.get(agent_type)
        if not agent:
            console.print(f"[red]Unknown agent type: {agent_type}[/red]")
            return
        
        console.print(f"[bold blue]{agent.name} Analysis:[/bold blue]")
        console.print("-" * 40)
        
        try:
            response = agent.print_response(
                query,
                user_id=user_id,
                stream=True
            )
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")
    
    def show_shared_memories(self, limit: int = 20):
        """Display shared memories across all agents."""
        
        try:
            memories = self.shared_memory_db.read_memories(
                user_id="team_user",  # Team shared user ID
                limit=limit
            )
            
            if memories:
                console.print("\n[bold cyan]Shared Team Memories (PostgreSQL):[/bold cyan]")
                for idx, memory in enumerate(memories, 1):
                    # Handle different memory object structures
                    if hasattr(memory, 'memory'):
                        memory_text = memory.memory
                        created_at = getattr(memory, 'created_at', 'Unknown')
                    elif hasattr(memory, 'content'):
                        memory_text = memory.content
                        created_at = getattr(memory, 'created_at', 'Unknown')
                    elif isinstance(memory, dict):
                        memory_text = memory.get('memory', memory.get('content', str(memory)))
                        created_at = memory.get('created_at', 'Unknown')
                    else:
                        memory_text = str(memory)
                        created_at = 'Unknown'
                    
                    console.print(Panel(
                        f"[yellow]Team Memory {idx}:[/yellow]\n{memory_text}",
                        title=f"Created: {created_at}",
                        expand=False
                    ))
            else:
                console.print("[yellow]No shared memories found.[/yellow]")
                
        except Exception as e:
            console.print(f"[red]Error accessing shared memories: {str(e)}[/red]")


def multi_agent_example():
    """Example with multiple agents sharing memory infrastructure in PostgreSQL."""
    
    console.print("[bold green]Multi-Agent Memory Sharing Example (PostgreSQL)[/bold green]\n")
    
    try:
        # Create the multi-agent team
        team = MultiAgentTeam(team_id="investment_team")
        
        # Comprehensive investment analysis using team coordination
        team.analyze_investment(
            "Analyze Tesla's current position in the EV market. "
            "I want both financial metrics and recent news analysis to make an investment decision."
        )
        
        console.print("\n" + "="*60 + "\n")
        
        # Follow-up analysis on a different company
        team.analyze_investment(
            "Now compare Apple's recent performance and provide investment recommendations. "
            "Consider both the financial fundamentals and recent market developments."
        )
        
        console.print("\n" + "="*60 + "\n")
        
        # Show how agents can reference shared memories
        team.analyze_investment(
            "Based on our previous analysis of Tesla and Apple, which would be a better investment right now?"
        )
        
        console.print("\n" + "="*60 + "\n")
        
        # Display shared memories
        team.show_shared_memories()
        
    except Exception as e:
        console.print(f"[red]Error in multi-agent example: {str(e)}[/red]")


def specialized_agent_example():
    """Example showing individual agent capabilities and specialization."""
    
    console.print("[bold green]Specialized Agent Capabilities[/bold green]\n")
    
    try:
        team = MultiAgentTeam(team_id="specialized_team")
        
        # Financial agent specialized analysis
        console.print("[bold cyan]Financial Agent Analysis:[/bold cyan]")
        team.individual_agent_analysis(
            "Provide a detailed financial analysis of Microsoft including key ratios, revenue trends, and valuation metrics.",
            agent_type="financial"
        )
        
        console.print("\n" + "="*40 + "\n")
        
        # Research agent specialized analysis
        console.print("[bold cyan]Research Agent Analysis:[/bold cyan]")
        team.individual_agent_analysis(
            "Research the latest developments in cloud computing and AI that might affect Microsoft's competitive position.",
            agent_type="research"
        )
        
        console.print("\n" + "="*40 + "\n")
        
        # Coordinator synthesis
        console.print("[bold cyan]Coordinator Synthesis:[/bold cyan]")
        team.individual_agent_analysis(
            "Based on the financial and research analysis, provide a comprehensive investment recommendation for Microsoft.",
            agent_type="coordinator"
        )
        
        console.print("\n" + "="*40 + "\n")
        team.show_shared_memories(limit=10)
        
    except Exception as e:
        console.print(f"[red]Error in specialized agent example: {str(e)}[/red]")


def collaborative_memory_example():
    """Example showing how agents build on each other's insights over time."""
    
    console.print("[bold green]Collaborative Memory Building[/bold green]\n")
    
    try:
        team = MultiAgentTeam(team_id="collaborative_team")
        
        # Build knowledge base over multiple interactions
        queries = [
            "What are the key trends in the semiconductor industry right now?",
            "How are AI developments affecting chip demand and which companies are best positioned?",
            "Analyze NVIDIA's financial performance in the context of AI chip demand.",
            "Should we invest in semiconductor stocks now, and if so, which ones?"
        ]
        
        for i, query in enumerate(queries, 1):
            console.print(f"[bold blue]Query {i}:[/bold blue] {query}")
            team.analyze_investment(query)
            console.print("\n" + "-"*50 + "\n")
        
        # Show how memory has been built up
        console.print("[bold cyan]Collaborative Memory Summary:[/bold cyan]")
        team.show_shared_memories(limit=15)
        
    except Exception as e:
        console.print(f"[red]Error in collaborative memory example: {str(e)}[/red]")


if __name__ == "__main__":
    # Show PostgreSQL configuration
    console.print("[bold cyan]PostgreSQL Configuration:[/bold cyan]")
    console.print(f"Database URL: {PostgresConfig.get_db_url()}")
    console.print()
    
    # Run multi-agent example
    console.print("\n" + "="*70 + "\n")
    multi_agent_example()
    
    # Run specialized agent example
    console.print("\n" + "="*70 + "\n")
    specialized_agent_example()
    
    # Run collaborative memory example
    console.print("\n" + "="*70 + "\n")
    collaborative_memory_example()
    
    # Show setup instructions
    console.print("\n" + "="*70 + "\n")
    console.print("[bold cyan]Setup Notes:[/bold cyan]")
    console.print("1. Make sure PostgreSQL is running")
    console.print("2. Create database: CREATE DATABASE agno_memory;")
    console.print("3. Set POSTGRES_URL environment variable")
    console.print("4. Set OPENAI_API_KEY environment variable")
    console.print("5. Install dependencies: pip install agno[postgres] python-dotenv rich")
    console.print("\n[bold cyan]Multi-Agent Features:[/bold cyan]")
    console.print("- Shared memory across specialized agents")
    console.print("- Financial analysis specialist")
    console.print("- Market research specialist")
    console.print("- Team coordination and synthesis")
    console.print("- Collaborative knowledge building")
    console.print("- Cross-agent information sharing") 