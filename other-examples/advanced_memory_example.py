"""
Advanced Agno Agent with Memory Implementation (PostgreSQL)
=========================================================

This example demonstrates an advanced AI agent with persistent memory using Agno and PostgreSQL.
The agent features:
- Session memory (short-term storage) in PostgreSQL
- User memories (long-term preferences and facts) in PostgreSQL
- Tool integration (web search, finance tools, reasoning)
- Multi-modal support
- Advanced memory management

Prerequisites:
- PostgreSQL database running
- Database created (e.g., CREATE DATABASE agno_memory;)
- Connection string in POSTGRES_URL environment variable
"""

import json
import os
from typing import Optional
from pathlib import Path

# Core Agno imports
from agno.agent import Agent
from agno.models.openai import OpenAIChat

# Memory and Storage imports
from agno.memory.v2.db.postgres import PostgresMemoryDb
from agno.memory.v2.memory import Memory
from agno.storage.postgres import PostgresStorage

# Tool imports
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agno.tools.reasoning import ReasoningTools
from dotenv import load_dotenv

# For pretty printing
from rich.console import Console
from rich.panel import Panel
from rich.json import JSON

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


class AdvancedMemoryAgent:
    """
    An advanced agent implementation with persistent memory capabilities and tools using PostgreSQL.
    """
    
    def __init__(self, user_id: str = "default_user", session_id: Optional[str] = None):
        self.user_id = user_id
        self.session_id = session_id
        
        # Get PostgreSQL connection parameters
        db_url = PostgresConfig.get_db_url()
        
        console.print("[cyan]Connecting to PostgreSQL database...[/cyan]")
        
        try:
            # Initialize memory database with PostgreSQL
            self.memory_db = PostgresMemoryDb(
                table_name="advanced_user_memories",
                db_url=db_url
            )
            
            # Initialize session storage with PostgreSQL
            self.storage = PostgresStorage(
                table_name="advanced_agent_sessions",
                db_url=db_url
            )
            
            # Create memory object
            self.memory = Memory(db=self.memory_db)
            
            # Initialize the agent
            self.agent = self._create_agent()
            
            console.print("[green]✓ Successfully connected to PostgreSQL[/green]")
            
        except Exception as e:
            console.print(f"[red]✗ Failed to connect to PostgreSQL: {str(e)}[/red]")
            console.print("[yellow]Make sure PostgreSQL is running and the database exists.[/yellow]")
            console.print("[yellow]Set POSTGRES_URL environment variable or individual POSTGRES_* variables.[/yellow]")
            raise
    
    def _create_agent(self) -> Agent:
        """Create and configure the advanced agent with memory and tools."""
        
        agent = Agent(
            # Model configuration
            model=OpenAIChat(id="gpt-4"),
            
            # Agent metadata
            name="Advanced Memory Assistant",
            description="An intelligent assistant with persistent memory and advanced tools",
            
            # Memory configuration
            memory=self.memory,
            storage=self.storage,
            enable_agentic_memory=True,  # Enable automatic memory creation
            
            # Session configuration
            session_id=self.session_id,
            
            # Memory settings
            add_history_to_messages=True,  # Include chat history in context
            read_tool_call_history=True,  # Allow agent to read its tool usage history
            
            # Tool configuration
            tools=[
                DuckDuckGoTools(),  # Web search capability
                YFinanceTools(      # Financial data access
                    stock_price=True,
                    analyst_recommendations=True,
                    company_info=True
                ),
                ReasoningTools(add_instructions=True),  # Enhanced reasoning
            ],
            
            # Instructions for the agent
            instructions=[
                "You are a helpful assistant with persistent memory stored in PostgreSQL.",
                "Remember important details about users and reference them naturally.",
                "Use your tools wisely to provide accurate, up-to-date information.",
                "When you learn new facts about the user, store them in memory.",
                "Always be truthful about what you remember or don't remember.",
                "Use tables to display structured data when appropriate.",
                "Leverage your reasoning tools for complex analysis.",
                "When users mention interests, remember them and use relevant tools.",
            ],
            
            # Output formatting
            markdown=True,
            show_tool_calls=True,  # Show when tools are being used
        )
        
        return agent
    
    def chat(self, message: str, image_path: Optional[str] = None) -> str:
        """
        Process a chat message with the agent.
        
        Args:
            message: The user's input message
            image_path: Optional path to an image file for multi-modal input
            
        Returns:
            The agent's response
        """
        try:
            # Prepare kwargs for the agent
            kwargs = {
                "stream": True,
                "user_id": self.user_id,
            }
            
            # Add image if provided (multi-modal support)
            if image_path and Path(image_path).exists():
                from agno.media import Image
                kwargs["images"] = [Image(filepath=Path(image_path))]
            
            # Get response from agent
            response = self.agent.print_response(message, **kwargs)
            
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")
            return f"I encountered an error: {str(e)}"
    
    def show_memories(self):
        """Display all stored memories for the current user."""
        
        try:
            user_memories = self.memory_db.read_memories(
                user_id=self.user_id,
                limit=100
            )
            
            if user_memories:
                console.print("\n[bold cyan]User Memories (from PostgreSQL):[/bold cyan]")
                for idx, memory in enumerate(user_memories, 1):
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
                        f"[yellow]Memory {idx}:[/yellow]\n{memory_text}",
                        title=f"Created: {created_at}",
                        expand=False
                    ))
            else:
                console.print("[yellow]No memories stored yet in PostgreSQL.[/yellow]")
                
        except Exception as e:
            console.print(f"[red]Error accessing memories: {str(e)}[/red]")
    
    def show_session_history(self):
        """Display the current session's conversation history."""
        
        if self.agent.session_id:
            messages = []
            session_run = self.agent.memory.runs.get(self.agent.session_id, [])
            
            if session_run:
                for message in session_run[-1].messages:
                    messages.append(message.to_dict())
            
            if messages:
                console.print(Panel(
                    JSON(json.dumps(messages, indent=2)),
                    title=f"Session History: {self.agent.session_id}",
                    expand=True
                ))
            else:
                console.print("[yellow]No messages in current session.[/yellow]")


def advanced_example():
    """Advanced example with tools and multi-session support."""
    
    console.print("[bold green]Advanced Memory Agent Example (PostgreSQL)[/bold green]\n")
    
    try:
        # Create agent
        agent = AdvancedMemoryAgent(user_id="jane_smith")
        
        # Interaction with personal info and interests
        console.print("[bold blue]User:[/bold blue] I'm Jane, and I'm interested in investing in tech stocks, especially AI companies. I also like to stay updated on market trends.")
        agent.chat("I'm Jane, and I'm interested in investing in tech stocks, especially AI companies. I also like to stay updated on market trends.")
        
        console.print("\n" + "="*50 + "\n")
        
        # Use tools based on remembered interests
        console.print("[bold blue]User:[/bold blue] Can you show me the current performance of NVDA and give me analyst recommendations?")
        agent.chat("Can you show me the current performance of NVDA and give me analyst recommendations?")
        
        console.print("\n" + "="*50 + "\n")
        
        # Search for recent news (using remembered context)
        console.print("[bold blue]User:[/bold blue] Find me recent news about AI companies I might be interested in.")
        agent.chat("Find me recent news about AI companies I might be interested in.")
        
        console.print("\n" + "="*50 + "\n")
        
        # Test reasoning capabilities
        console.print("[bold blue]User:[/bold blue] Based on what you know about my interests and the current market data, should I consider investing in NVDA right now?")
        agent.chat("Based on what you know about my interests and the current market data, should I consider investing in NVDA right now?")
        
        console.print("\n" + "="*50 + "\n")
        
        # Show memories and session info
        agent.show_memories()
        agent.show_session_history()
        
    except Exception as e:
        console.print(f"[red]Error in advanced example: {str(e)}[/red]")


def demonstration_with_different_interests():
    """Demonstrate the agent remembering different user interests and providing personalized responses."""
    
    console.print("[bold green]Personalized Interest Demonstration[/bold green]\n")
    
    try:
        # Create agent for a user interested in renewable energy
        agent = AdvancedMemoryAgent(user_id="eco_investor")
        
        console.print("[bold blue]User:[/bold blue] Hi, I'm interested in renewable energy investments, particularly solar and wind power companies.")
        agent.chat("Hi, I'm interested in renewable energy investments, particularly solar and wind power companies.")
        
        console.print("\n" + "="*30 + "\n")
        
        # Agent should use this context for personalized recommendations
        console.print("[bold blue]User:[/bold blue] What are some good investment opportunities for me?")
        agent.chat("What are some good investment opportunities for me?")
        
        console.print("\n" + "="*30 + "\n")
        
        # Show how memory persists
        console.print("[bold blue]User:[/bold blue] What do you remember about my investment preferences?")
        agent.chat("What do you remember about my investment preferences?")
        
        console.print("\n" + "="*30 + "\n")
        agent.show_memories()
        
    except Exception as e:
        console.print(f"[red]Error in demonstration: {str(e)}[/red]")


if __name__ == "__main__":
    # Show PostgreSQL configuration
    console.print("[bold cyan]PostgreSQL Configuration:[/bold cyan]")
    console.print(f"Database URL: {PostgresConfig.get_db_url()}")
    console.print()
    
    # Run advanced example
    console.print("\n" + "="*60 + "\n")
    advanced_example()
    
    # Run personalized demonstration
    console.print("\n" + "="*60 + "\n")
    demonstration_with_different_interests()
    
    # Show setup instructions
    console.print("\n" + "="*60 + "\n")
    console.print("[bold cyan]Setup Notes:[/bold cyan]")
    console.print("1. Make sure PostgreSQL is running")
    console.print("2. Create database: CREATE DATABASE if not exists;")
    console.print("3. Set POSTGRES_URL environment variable")
    console.print("4. Set OPENAI_API_KEY environment variable")
    console.print("5. Install dependencies: pip install agno[postgres] python-dotenv rich")
    console.print("\n[bold cyan]Features Demonstrated:[/bold cyan]")
    console.print("- Persistent memory across conversations")
    console.print("- Tool integration (web search, finance data)")
    console.print("- Reasoning capabilities")
    console.print("- Personalized responses based on user interests")
    console.print("- Multi-modal support (can be extended with images)") 