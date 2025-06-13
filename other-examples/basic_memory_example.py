"""
Basic Agno Agent with Memory Implementation (PostgreSQL)
======================================================

This example demonstrates the basic setup of an AI agent with persistent memory using Agno and PostgreSQL.

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
from agno.storage.postgres import PostgresStorage

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


class BasicMemoryAgent:
    """
    A basic agent implementation with persistent memory capabilities using PostgreSQL.
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
                table_name="user_memories",
                db_url=db_url
            )
            
            # Initialize session storage with PostgreSQL
            self.storage = PostgresStorage(
                table_name="agent_sessions",
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
        """Create and configure the basic agent with memory."""
        
        agent = Agent(
            # Model configuration
            model=OpenAIChat(id="gpt-4"),
            
            # Agent metadata
            name="Basic Memory Assistant",
            description="A basic intelligent assistant with persistent memory",
            
            # Memory configuration
            memory=self.memory,
            storage=self.storage,
            enable_agentic_memory=True,  # Enable automatic memory creation
            
            # Session configuration
            session_id=self.session_id,
            
            # Memory settings
            add_history_to_messages=True,  # Include chat history in context
            
            # Basic instructions
            instructions=[
                "You are a helpful assistant with persistent memory stored in PostgreSQL.",
                "Remember important details about users and reference them naturally.",
                "When you learn new facts about the user, store them in memory.",
                "Always be truthful about what you remember or don't remember.",
            ],
            
            # Output formatting
            markdown=True,
        )
        
        return agent
    
    def chat(self, message: str) -> str:
        """
        Process a chat message with the agent.
        
        Args:
            message: The user's input message
            
        Returns:
            The agent's response
        """
        try:
            # Get response from agent
            response = self.agent.print_response(
                message, 
                stream=True,
                user_id=self.user_id
            )
            
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


def basic_example():
    """Basic usage example with PostgreSQL."""
    
    console.print("[bold green]Basic Memory Agent Example (PostgreSQL)[/bold green]\n")
    
    try:
        # Create agent
        agent = BasicMemoryAgent(user_id="john_doe")
        
        # First interaction
        console.print("[bold blue]User:[/bold blue] Hi! My name is John and I live in New York. I work as a software engineer.")
        agent.chat("Hi! My name is John and I live in New York. I work as a software engineer.")
        
        console.print("\n" + "="*50 + "\n")
        
        # Second interaction (agent should remember)
        console.print("[bold blue]User:[/bold blue] What do you remember about me?")
        agent.chat("What do you remember about me?")
        
        console.print("\n" + "="*50 + "\n")
        
        # Show stored memories
        agent.show_memories()
        
    except Exception as e:
        console.print(f"[red]Error in basic example: {str(e)}[/red]")


if __name__ == "__main__":
    # Show PostgreSQL configuration
    console.print("[bold cyan]PostgreSQL Configuration:[/bold cyan]")
    console.print(f"Database URL: {PostgresConfig.get_db_url()}")
    console.print()
    
    # Run basic example
    basic_example()