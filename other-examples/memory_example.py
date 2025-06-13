"""
Agno Agent with Memory Implementation (PostgreSQL)
=================================================

This example demonstrates how to build an AI agent with persistent memory using Agno and PostgreSQL.
The agent features:
- Session memory (short-term storage) in PostgreSQL
- User memories (long-term preferences and facts) in PostgreSQL
- Database persistence across sessions
- Multi-modal support
- Connection pooling and proper database configuration

Prerequisites:
- PostgreSQL database running
- Database created (e.g., CREATE DATABASE agno_memory;)
- Connection string in POSTGRES_URL environment variable
"""

import json
import os
from typing import Optional
from pathlib import Path
from datetime import datetime

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
        # Priority order:
        # 1. POSTGRES_URL environment variable
        # 2. Individual components
        # 3. Default local connection
        
        if os.getenv("POSTGRES_URL"):
            return os.getenv("POSTGRES_URL")
        
        # Build from components
        host = os.getenv("POSTGRES_HOST")
        port = os.getenv("POSTGRES_PORT")
        user = os.getenv("POSTGRES_USER")
        password = os.getenv("POSTGRES_PASSWORD")
        database = os.getenv("POSTGRES_DB")
        
        if password:
            return f"postgresql://{user}:{password}@{host}:{port}/{database}"
        else:
            return f"postgresql://{user}@{host}:{port}/{database}"


class MemoryAgent:
    """
    A comprehensive agent implementation with persistent memory capabilities using PostgreSQL.
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
        """Create and configure the agent with memory and tools."""
        
        agent = Agent(
            # Model configuration (you can swap models easily)
            model=OpenAIChat(id="gpt-4.1"),
            # model=Claude(id="claude-3-sonnet-20240229"),
            # model=Groq(id="llama-3.3-70b-versatile"),
            
            # Agent metadata
            name="Memory Assistant",
            description="An intelligent assistant with persistent memory and reasoning capabilities",
            
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
            
            # The response is already printed by print_response
            # Return the content for programmatic use
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")
            return f"I encountered an error: {str(e)}"
    
    def show_memories(self):
        """Display all stored memories for the current user."""
        
        try:
            # Use the correct API method: read_memories
            user_memories = self.memory_db.read_memories(
                user_id=self.user_id,
                limit=100  # Adjust limit as needed
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
            console.print("[yellow]Memory retrieval API may have changed. Check Agno documentation.[/yellow]")
    
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
    
    def list_sessions(self):
        """List all available sessions from PostgreSQL."""
        
        sessions = self.storage.get_all_session_ids(user_id=self.user_id)
        
        if sessions:
            console.print("\n[bold cyan]Available Sessions (from PostgreSQL):[/bold cyan]")
            for session_id in sessions:
                console.print(f"  - {session_id}")
        else:
            console.print("[yellow]No sessions found in PostgreSQL.[/yellow]")
    
    def clear_memories(self):
        """Clear all memories for the current user."""
        
        # This would require implementing a clear method in the memory DB
        # For now, we'll note this as a limitation
        console.print("[yellow]Note: Memory clearing would need to be implemented in the PostgreSQL memory DB.[/yellow]")
        console.print("[yellow]You can manually clear memories using SQL: DELETE FROM user_memories WHERE user_id = 'user_id';[/yellow]")



# Example usage functions
def basic_example():
    """Basic usage example with PostgreSQL."""
    
    console.print("[bold green]Basic Memory Agent Example (PostgreSQL)[/bold green]\n")
    
    try:
        # Create agent
        agent = MemoryAgent(user_id="john_doe")
        
        # First interaction
        agent.chat("Hi! My name is John and I live in New York. I work as a software engineer.")
        
        # Second interaction (agent should remember)
        agent.chat("What do you remember about me?")
        
        # Show stored memories
        agent.show_memories()
        
    except Exception as e:
        console.print(f"[red]Error in basic example: {str(e)}[/red]")



def advanced_example():
    """Advanced example with tools and multi-session support."""
    
    console.print("[bold green]Advanced Memory Agent Example (PostgreSQL)[/bold green]\n")
    
    try:
        # Create agent
        agent = MemoryAgent(user_id="jane_smith")
        
        # Interaction with personal info
        agent.chat("I'm Jane, and I'm interested in investing in tech stocks, especially AI companies.")
        
        # Use tools based on remembered interests
        agent.chat("Can you show me the current performance of NVDA and give me analyst recommendations?")
        
        # Search for recent news (using remembered context)
        agent.chat("Find me recent news about AI companies I might be interested in.")
        
        # Show memories and session info
        agent.show_memories()
        agent.show_session_history()
        
    except Exception as e:
        console.print(f"[red]Error in advanced example: {str(e)}[/red]")


def multi_agent_example():
    """Example with multiple agents sharing memory infrastructure in PostgreSQL."""
    
    console.print("[bold green]Multi-Agent Memory Sharing Example (PostgreSQL)[/bold green]\n")
    
    try:
        # Get PostgreSQL connection
        db_url = PostgresConfig.get_db_url()
        
        # Shared memory database using PostgreSQL
        shared_memory_db = PostgresMemoryDb(
            table_name="shared_memories",
            db_url=db_url
        )
        shared_memory = Memory(db=shared_memory_db)
        
        # Financial analyst agent
        financial_agent = Agent(
            name="Financial Analyst",
            model=OpenAIChat(id="gpt-4"),
            memory=shared_memory,
            enable_agentic_memory=True,
            tools=[YFinanceTools(stock_price=True, analyst_recommendations=True)],
            instructions=["You are a financial analyst. Store important financial insights in memory."],
        )
        
        # Research agent
        research_agent = Agent(
            name="Research Assistant",
            model=OpenAIChat(id="gpt-4"),
            memory=shared_memory,
            enable_agentic_memory=True,
            tools=[DuckDuckGoTools()],
            instructions=["You are a research assistant. Store important research findings in memory."],
        )
        
        # Team coordinator agent
        coordinator = Agent(
            name="Team Coordinator",
            model=OpenAIChat(id="gpt-4"),
            team=[financial_agent, research_agent],
            memory=shared_memory,
            enable_agentic_memory=True,
            instructions=[
                "Coordinate between financial and research agents.",
                "Synthesize insights from both agents.",
                "Store key conclusions in memory.",
            ],
        )
        
        # Example usage
        user_id = "team_user"
        coordinator.print_response(
            "Analyze Tesla's current position in the EV market. "
            "I want both financial metrics and recent news analysis.",
            user_id=user_id,
            stream=True
        )
        
    except Exception as e:
        console.print(f"[red]Error in multi-agent example: {str(e)}[/red]")


def persistent_session_example():
    """Example showing session persistence across script runs with PostgreSQL."""
    
    console.print("[bold green]Persistent Session Example (PostgreSQL)[/bold green]\n")
    
    try:
        user_id = "persistent_user"
        
        # Get PostgreSQL connection
        db_url = PostgresConfig.get_db_url()
        
        # Check for existing sessions
        storage = PostgresStorage(
            table_name="agent_sessions",
            db_url=db_url
        )
        existing_sessions = storage.get_all_session_ids(user_id=user_id)
        
        if existing_sessions:
            console.print(f"Found {len(existing_sessions)} existing sessions in PostgreSQL")
            # Find the actual most recent session by parsing timestamps from session IDs
            # Session IDs should be in format: user_id_YYYYMMDD_HHMMSS_random
            sessions_with_timestamps = []
            for session_id in existing_sessions:
                try:
                    # Parse timestamp from session ID
                    parts = session_id.split("_")
                    if len(parts) >= 3:
                        date_part = parts[-3]  # YYYYMMDD (3rd from end)
                        time_part = parts[-2]  # HHMMSS (2nd from end)
                        timestamp_str = f"{date_part}_{time_part}"
                        timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                        sessions_with_timestamps.append((session_id, timestamp))
                except (ValueError, IndexError):
                    # If timestamp parsing fails, skip this session
                    continue
            
            # Sort by timestamp and get the most recent
            if sessions_with_timestamps:
                sessions_with_timestamps.sort(key=lambda x: x[1], reverse=True)  # Sort by timestamp, newest first
                session_id = sessions_with_timestamps[0][0]  # Get session ID of most recent
                console.print(f"Continuing session: {session_id}")
            else:
                # Fallback to the last session in the list if no timestamps can be parsed
                session_id = existing_sessions[-1]  # Use most recent
                console.print(f"Continuing session: {session_id}")
        else:
            session_id = None
            console.print("Starting new session in PostgreSQL")
        
        # Create agent with session
        agent = MemoryAgent(user_id=user_id, session_id=session_id)
        
        # Continue conversation
        agent.chat("What have we discussed before?")
        
        # Show session info
        agent.list_sessions()
        
    except Exception as e:
        console.print(f"[red]Error in persistent session example: {str(e)}[/red]")



if __name__ == "__main__":
    # Show PostgreSQL configuration
    console.print("[bold cyan]PostgreSQL Configuration:[/bold cyan]")
    console.print(f"Database URL: {PostgresConfig.get_db_url()}")
    console.print()
    
    # Run examples
    console.print("\n" + "="*50 + "\n")
    basic_example()
    
    # console.print("\n" + "="*50 + "\n")
    # advanced_example()
    
    # console.print("\n" + "="*50 + "\n")
    # multi_agent_example()
    
    # console.print("\n" + "="*50 + "\n")
    # persistent_session_example()
    
    # Show additional setup instructions if needed
    console.print("\n" + "="*50 + "\n")
    console.print("[bold cyan]Additional Notes:[/bold cyan]")