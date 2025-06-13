"""
Persistent Session Example with PostgreSQL
==========================================

This example demonstrates session persistence across script runs using PostgreSQL.
Features:
- Session continuity across application restarts
- Session management and retrieval
- Long-term conversation history
- User session preferences
- UUID-based user identification

Prerequisites:
- PostgreSQL database running
- Database created (e.g., CREATE DATABASE agno_memory;)
- Connection string in POSTGRES_URL environment variable

Usage:
- python persistent_session_example.py <username>
- Example: python persistent_session_example.py john_doe
"""

import os
import sys
import uuid
from typing import Optional, List
from datetime import datetime, timedelta

# Core Agno imports
from agno.agent import Agent
from agno.models.openai import OpenAIChat

# Memory and Storage imports
from agno.memory.v2.db.postgres import PostgresMemoryDb
from agno.memory.v2.memory import Memory
from agno.storage.postgres import PostgresStorage

# Tool imports
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.reasoning import ReasoningTools
from dotenv import load_dotenv

# For pretty printing
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

load_dotenv("../.env")


def get_user_id_from_argument():
    """
    Get user ID from command line argument and generate a deterministic UUID.

    Returns:
        tuple: (original_username, uuid_user_id)
    """
    if len(sys.argv) < 2:
        console.print("[red]Error: Please provide a username as an argument.[/red]")
        console.print(
            "[yellow]Usage: python persistent_session_example.py <username>[/yellow]"
        )
        console.print(
            "[yellow]Example: python persistent_session_example.py john_doe[/yellow]"
        )
        sys.exit(1)

    username = sys.argv[1].strip()

    # Validate username
    if not username:
        console.print("[red]Error: Username cannot be empty.[/red]")
        sys.exit(1)

    # Generate deterministic UUID based on username
    # Using uuid5 with a custom namespace to ensure same username = same UUID
    namespace = uuid.UUID(
        "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
    )  # Standard DNS namespace
    user_uuid = str(uuid.uuid5(namespace, username))

    return username, user_uuid


def get_session_timestamp(session_id: str) -> Optional[datetime]:
    """
    Extract timestamp from session ID.

    Args:
        session_id: Session ID in format 'user_uuid_YYYYMMDD_HHMMSS_random'

    Returns:
        datetime object or None if parsing fails
    """
    try:
        parts = session_id.split("_")
        if len(parts) >= 3:
            date_part = parts[1]  # YYYYMMDD
            time_part = parts[2]  # HHMMSS

            # Parse timestamp
            timestamp_str = f"{date_part}_{time_part}"
            return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
    except (ValueError, IndexError):
        pass

    return None


def is_session_expired(session_id: str, max_age_minutes: int = 30) -> bool:
    """
    Check if a session is older than the specified age.

    Args:
        session_id: Session ID to check
        max_age_minutes: Maximum age in minutes (default: 30)

    Returns:
        True if session is expired, False otherwise
    """
    session_time = get_session_timestamp(session_id)
    if session_time is None:
        return True  # Consider unparseable sessions as expired

    age = datetime.now() - session_time
    return age > timedelta(minutes=max_age_minutes)


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


class PersistentSessionAgent:
    """
    An agent with persistent session management capabilities using PostgreSQL.
    """

    def __init__(self, user_id: str = "default_user", session_id: Optional[str] = None):
        self.user_id = user_id
        self.session_id = session_id

        # Get PostgreSQL connection parameters
        db_url = PostgresConfig.get_db_url()

        console.print(
            "[cyan]Connecting to PostgreSQL for persistent sessions...[/cyan]"
        )

        try:
            # Initialize memory database with PostgreSQL
            self.memory_db = PostgresMemoryDb(
                table_name="persistent_user_memories", db_url=db_url
            )

            # Initialize session storage with PostgreSQL
            self.storage = PostgresStorage(
                table_name="persistent_agent_sessions", db_url=db_url
            )

            # Create memory object
            self.memory = Memory(db=self.memory_db)

            # Initialize the agent
            self.agent = self._create_agent()

            console.print("[green]✓ Successfully connected to PostgreSQL[/green]")

        except Exception as e:
            console.print(f"[red]✗ Failed to connect to PostgreSQL: {str(e)}[/red]")
            console.print(
                "[yellow]Make sure PostgreSQL is running and the database exists.[/yellow]"
            )
            raise

    def _create_agent(self) -> Agent:
        """Create and configure the agent with memory and tools."""

        agent = Agent(
            # Model configuration
            model=OpenAIChat(id="gpt-4o"),
            # Agent metadata
            name="Persistent Memory Assistant",
            description="An intelligent assistant with persistent sessions and memory",
            # Memory configuration
            memory=self.memory,
            storage=self.storage,
            enable_agentic_memory=True,
            # Session configuration
            session_id=self.session_id,
            # Memory settings
            add_history_to_messages=True,  # Include chat history in context
            read_tool_call_history=True,
            # Tool configuration
            tools=[
                DuckDuckGoTools(),
                ReasoningTools(add_instructions=True),
            ],
            # Instructions for the agent
            instructions=[
                "You are a helpful assistant with persistent memory and session continuity.",
                "Remember important details about users across multiple conversations.",
                "Reference previous conversations naturally when relevant.",
                "Store important information in memory for future reference.",
                "If this is a continuing session, acknowledge previous interactions.",
                "Be aware of the conversation history and build upon it.",
                "Focus on the most recent parts of the conversation for immediate context.",
            ],
            # Output formatting
            markdown=True,
            show_tool_calls=True,
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
                message, stream=True, user_id=self.user_id
            )

            return response.content if hasattr(response, "content") else str(response)

        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")
            return f"I encountered an error: {str(e)}"

    def list_sessions(self) -> List[str]:
        """List all available sessions for the current user."""

        try:
            sessions = self.storage.get_all_session_ids(user_id=self.user_id)
            return sessions if sessions else []
        except Exception as e:
            console.print(f"[red]Error retrieving sessions: {str(e)}[/red]")
            return []

    def show_session_summary(self, session_id: str = None):
        """Show a summary of a specific session."""

        target_session = session_id or self.session_id
        if not target_session:
            console.print("[yellow]No session specified.[/yellow]")
            return

        try:
            # This is a simplified version - in practice you'd need to implement
            # session metadata retrieval from your storage system
            console.print(
                f"[bold cyan]Session Summary: {self.agent.memory.get_session_summary(target_session, user_id=self.user_id)}[/bold cyan]"
            )
            console.print(f"User ID: {self.user_id}")
            console.print(
                f"Current Session: {'Yes' if target_session == self.session_id else 'No'}"
            )

        except Exception as e:
            console.print(f"[red]Error showing session summary: {str(e)}[/red]")

    def show_memories(self):
        """Display all stored memories for the current user."""

        try:
            user_memories = self.memory_db.read_memories(user_id=self.user_id, limit=50)

            if user_memories:
                console.print(
                    "\n[bold cyan]User Memories (from PostgreSQL):[/bold cyan]"
                )
                for idx, memory in enumerate(user_memories, 1):
                    # Handle different memory object structures

                    if hasattr(memory, "memory"):
                        memory_text = memory.memory

                    elif hasattr(memory, "content"):
                        memory_text = memory.content

                    elif isinstance(memory, dict):
                        memory_text = memory.get(
                            "memory", memory.get("content", str(memory))
                        )

                    else:
                        memory_text = str(memory)

                    console.print(
                        Panel(
                            f"[yellow]Memory {idx}:[/yellow]\n{memory_text}",
                            expand=False,
                        )
                    )
            else:
                console.print("[yellow]No memories stored yet in PostgreSQL.[/yellow]")

        except Exception as e:
            console.print(f"[red]Error accessing memories: {str(e)}[/red]")


class SessionManager:
    """Helper class for managing persistent sessions."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.db_url = PostgresConfig.get_db_url()

        # Initialize storage for session management
        self.storage = PostgresStorage(
            table_name="persistent_agent_sessions", db_url=self.db_url
        )

    def create_new_session(self) -> str:
        """Create a new session ID."""
        session_id = f"{self.user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        return session_id

    def get_available_sessions(self) -> List[str]:
        """Get all available sessions for the user."""
        try:
            return self.storage.get_all_session_ids(user_id=self.user_id)
        except Exception as e:
            console.print(f"[red]Error retrieving sessions: {str(e)}[/red]")
            return []

    def get_most_recent_session(self) -> Optional[str]:
        """Get the most recent session by parsing timestamps from session IDs."""
        existing_sessions = self.get_available_sessions()
        if not existing_sessions:
            return None

        # Find the actual most recent session by parsing timestamps
        sessions_with_timestamps = []
        for session_id in existing_sessions:
            timestamp = get_session_timestamp(session_id)
            if timestamp:
                sessions_with_timestamps.append((session_id, timestamp))

        # Sort by timestamp and get the most recent
        if sessions_with_timestamps:
            sessions_with_timestamps.sort(
                key=lambda x: x[1], reverse=True
            )  # Sort by timestamp, newest first
            return sessions_with_timestamps[0][0]  # Get session ID of most recent
        else:
            # Fallback to the last session in the list if no timestamps can be parsed
            return existing_sessions[-1]

    def display_sessions_table(self, sessions: List[str]):
        """Display sessions in a formatted table."""
        if not sessions:
            console.print("[yellow]No existing sessions found.[/yellow]")
            return

        table = Table(title="Available Sessions")
        table.add_column("Index", style="cyan", no_wrap=True)
        table.add_column("Session ID", style="green")
        table.add_column("Created", style="yellow")

        for idx, session_id in enumerate(sessions, 1):
            # Extract timestamp from session ID if possible
            try:
                parts = session_id.split("_")
                if len(parts) >= 3:
                    date_part = parts[1]
                    time_part = parts[2]
                    created = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]} {time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}"
                else:
                    created = "Unknown"
            except Exception as e:
                print(e)
                created = "Unknown"

            table.add_row(str(idx), session_id, created)

        console.print(table)


def persistent_session_example():
    """Example showing session persistence across script runs with PostgreSQL."""

    console.print("[bold green]Persistent Session Example (PostgreSQL)[/bold green]\n")

    try:
        username, user_uuid = get_user_id_from_argument()

        # Display user information
        console.print(f"[bold cyan]Username:[/bold cyan] {username}")
        console.print(f"[bold cyan]User UUID:[/bold cyan] {user_uuid}")

        # Create session manager using UUID as user_id
        session_manager = SessionManager(user_uuid)

        # Check for existing sessions
        existing_sessions = session_manager.get_available_sessions()

        if existing_sessions:
            console.print(
                f"Found {len(existing_sessions)} existing sessions in PostgreSQL for user '{username}'"
            )
            session_manager.display_sessions_table(existing_sessions)

            # Get the most recent session using the helper method
            most_recent_session = session_manager.get_most_recent_session()

            if is_session_expired(most_recent_session, max_age_minutes=30):
                session_time = get_session_timestamp(most_recent_session)
                age_minutes = (
                    (datetime.now() - session_time).total_seconds() / 60
                    if session_time
                    else "unknown"
                )

                console.print(
                    f"\n[yellow]Most recent session is {age_minutes:.1f} minutes old (>30 min limit)[/yellow]"
                )
                console.print(
                    "[cyan]Creating new session for better organization...[/cyan]"
                )

                session_id = session_manager.create_new_session()
                console.print(
                    f"[cyan]Starting new session for user '{username}': {session_id}[/cyan]"
                )
            else:
                session_time = get_session_timestamp(most_recent_session)
                age_minutes = (
                    (datetime.now() - session_time).total_seconds() / 60
                    if session_time
                    else "unknown"
                )

                console.print(
                    f"\n[green]Most recent session is {age_minutes:.1f} minutes old (<30 min limit)[/green]"
                )
                console.print(
                    f"[cyan]Continuing with recent session: {most_recent_session}[/cyan]"
                )
                session_id = most_recent_session
        else:
            session_id = session_manager.create_new_session()
            console.print(
                f"[cyan]Starting new session for user '{username}': {session_id}[/cyan]"
            )

        # Create agent with session using UUID as user_id
        agent = PersistentSessionAgent(user_id=user_uuid, session_id=session_id)

        # Show welcome message and instructions
        console.print("\n" + "=" * 60 + "\n")
        if existing_sessions:
            console.print(
                f"[bold yellow]Welcome back, {username}! What would you like to do today?[/bold yellow]"
            )
            console.print(
                "[italic]The agent will remember your previous interactions.[/italic]\n"
            )
        else:
            console.print(
                f"[bold yellow]Welcome, {username}! Starting a new conversation...[/bold yellow]"
            )
            console.print(
                "[italic]This is your first session. The agent will remember our conversation for next time.[/italic]\n"
            )

        # Show available commands
        console.print("[bold cyan]Available Commands:[/bold cyan]")
        console.print("• Type your message to chat with the agent")
        console.print("• 'memories' - Show stored memories")
        console.print("• 'sessions' - Show all available sessions")
        console.print("• 'summary' - Show current session summary")
        console.print("• 'quit' or 'exit' - End the conversation")
        console.print("\n" + "=" * 60 + "\n")

        # Interactive chat loop
        while True:
            try:
                # Get user input
                user_input = input(f"\n[{username}]: ").strip()

                # Handle empty input
                if not user_input:
                    continue

                # Handle special commands
                if user_input.lower() in ["quit", "exit", "bye", "goodbye"]:
                    console.print(
                        f"\n[green]Goodbye, {username}! Your conversation has been saved to PostgreSQL.[/green]"
                    )
                    console.print(
                        "[italic]You can continue this conversation next time you run the script with the same username.[/italic]"
                    )
                    break

                elif user_input.lower() == "memories":
                    agent.show_memories()
                    continue

                elif user_input.lower() == "sessions":
                    console.print(
                        f"\n[bold cyan]All Available Sessions for {username}:[/bold cyan]"
                    )
                    all_sessions = agent.list_sessions()
                    session_manager.display_sessions_table(all_sessions)
                    continue

                elif user_input.lower() == "summary":
                    agent.show_session_summary()
                    continue

                elif user_input.lower() in ["help", "commands"]:
                    console.print("\n[bold cyan]Available Commands:[/bold cyan]")
                    console.print("• Type your message to chat with the agent")
                    console.print("• 'memories' - Show stored memories")
                    console.print("• 'sessions' - Show all available sessions")
                    console.print("• 'summary' - Show current session summary")
                    console.print("• 'help' - Show this help message")
                    console.print("• 'quit' or 'exit' - End the conversation")
                    continue

                # Send message to agent
                console.print(f"\n[bold blue]{username}:[/bold blue] {user_input}")
                console.print("\n[bold green]Assistant:[/bold green]")
                agent.chat(user_input)

            except KeyboardInterrupt:
                console.print(
                    f"\n\n[yellow]Conversation interrupted for {username}.[/yellow]"
                )
                console.print(
                    "[green]Your session has been saved to PostgreSQL.[/green]"
                )
                break
            except EOFError:
                console.print(
                    f"\n\n[green]Session ended for {username}. Your conversation has been saved to PostgreSQL.[/green]"
                )
                break

        # Show final session summary
        console.print("\n" + "=" * 60 + "\n")
        console.print(f"[bold cyan]Session Summary for {username}:[/bold cyan]")
        agent.show_session_summary()

        # Show available sessions
        console.print(
            f"\n[bold cyan]All Available Sessions for {username}:[/bold cyan]"
        )
        all_sessions = agent.list_sessions()
        session_manager.display_sessions_table(all_sessions)

    except Exception as e:
        console.print(f"[red]Error in persistent session example: {str(e)}[/red]")


if __name__ == "__main__":
    # Show PostgreSQL configuration
    console.print("[bold cyan]PostgreSQL Configuration:[/bold cyan]")
    console.print(f"Database URL: {PostgresConfig.get_db_url()}")
    console.print()

    # Run the persistent session example
    console.print("\n" + "=" * 60 + "\n")
    persistent_session_example()
