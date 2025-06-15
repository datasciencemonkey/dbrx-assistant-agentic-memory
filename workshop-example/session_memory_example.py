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
- Automatic connection retry with exponential backoff
- Advanced session search and filtering capabilities

Connection Retry Logic:
- Automatically retries failed PostgreSQL connections up to 3 times
- Uses exponential backoff: delays of 2, 4, and 8 seconds
- Shows progress indicators during retry attempts
- Provides clear error messages and troubleshooting tips

Session Search & Filtering:
- Search conversations by keywords or content
- Filter sessions by date ranges (--from, --to)
- Advanced regex pattern matching support
- Search stored memories separately
- Interactive navigation between search results
- Highlighted search terms in results
- Paginated results display

Prerequisites:
- PostgreSQL database running
- Database created (e.g., CREATE DATABASE agno_memory;)
- Connection string in POSTGRES_URL environment variable

Environment Variables:
- POSTGRES_URL: Full connection URL (overrides all other settings)
- POSTGRES_HOST: Database host (default: localhost)
- POSTGRES_PORT: Database port (default: 5432)
- POSTGRES_USER: Database username (default: postgres)
- POSTGRES_PASSWORD: Database password
- POSTGRES_DB: Database name (default: agno_memory)

Usage:
- python persistent_session_example.py <username>
- Example: python persistent_session_example.py john_doe

Search Commands:
- search <keywords> - Find conversations containing keywords
- search --from 2024-01-01 --to 2024-12-31 - Date range search
- search --regex 'pattern' - Use regex pattern matching
- search --memories <keywords> - Search stored memories
- view <index> - View detailed session from search results
- continue <index> - Switch to a session from search results
"""

import os
import sys
import uuid
import time
import re
import psycopg2
from typing import Optional, List, TypeVar, Callable, Dict, Any
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

# Retry configuration constants
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 2  # Base delay in seconds for exponential backoff
CONNECTION_TIMEOUT = 10  # Connection timeout in seconds

# Type variable for generic connection return type
T = TypeVar('T')


def connect_with_retry(
    connection_func: Callable[[], T],
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: int = DEFAULT_BASE_DELAY,
    connection_name: str = "database"
) -> T:
    """
    Attempt to establish a connection with exponential backoff retry logic.
    
    Args:
        connection_func: A callable that attempts to establish a connection
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Base delay in seconds for exponential backoff (default: 2)
        connection_name: Name of the connection for logging purposes
        
    Returns:
        The connection object returned by connection_func
        
    Raises:
        Exception: If all retry attempts fail
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            # Attempt to establish connection
            console.print(f"[cyan]Attempting to connect to {connection_name}...[/cyan]")
            connection = connection_func()
            console.print(f"[green]✓ Successfully connected to {connection_name}[/green]")
            return connection
            
        except Exception as e:
            last_exception = e
            
            if attempt < max_retries - 1:
                # Calculate delay with exponential backoff
                delay = base_delay ** (attempt + 1)
                
                console.print(
                    f"[yellow]Connection attempt {attempt + 1} failed: {str(e)}[/yellow]"
                )
                console.print(
                    f"[yellow]Retrying in {delay} seconds... (attempt {attempt + 2}/{max_retries})[/yellow]"
                )
                
                # Show progress dots during wait
                for i in range(delay):
                    print(".", end="", flush=True)
                    time.sleep(1)
                print()  # New line after dots
                
            else:
                # Final attempt failed
                console.print(
                    f"[red]✗ All {max_retries} connection attempts failed[/red]"
                )
                console.print(
                    f"[red]Last error: {str(e)}[/red]"
                )
    
    # If we get here, all attempts failed
    raise last_exception if last_exception else Exception(
        f"Failed to connect to {connection_name} after {max_retries} attempts"
    )


class SessionSearcher:
    """
    Advanced session search functionality with full-text search, date filtering,
    and regex support for searching through conversation history.
    """
    
    def __init__(self, db_url: str, table_name: str = "persistent_agent_sessions"):
        self.db_url = db_url
        self.table_name = table_name
    
    def search_sessions(
        self,
        user_id: str,
        query: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        use_regex: bool = False,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Search sessions by content, date range, or regex patterns.
        
        Args:
            user_id: User ID to search sessions for
            query: Search query string (keywords or regex pattern)
            start_date: Start date for date range filtering
            end_date: End date for date range filtering
            use_regex: Whether to use regex pattern matching
            limit: Maximum number of results to return
            
        Returns:
            List of session dictionaries with metadata and snippets
        """
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cursor:
                    # Base query to get session data
                    base_query = f"""
                    SELECT session_id, user_id, runs, created_at, updated_at
                    FROM {self.table_name}
                    WHERE user_id = %s
                    """
                    params = [user_id]
                    
                    # Add date filtering
                    if start_date:
                        base_query += " AND created_at >= %s"
                        params.append(start_date)
                    
                    if end_date:
                        base_query += " AND created_at <= %s"
                        params.append(end_date)
                    
                    # Add content search
                    if query:
                        if use_regex:
                            # Use PostgreSQL regex matching
                            base_query += " AND runs::text ~* %s"
                            params.append(query)
                        else:
                            # Use full-text search with PostgreSQL's built-in capabilities
                            base_query += " AND runs::text ILIKE %s"
                            params.append(f"%{query}%")
                    
                    base_query += " ORDER BY created_at DESC LIMIT %s"
                    params.append(limit)
                    
                    cursor.execute(base_query, params)
                    results = cursor.fetchall()
                    
                    # Process results and extract snippets
                    processed_results = []
                    for row in results:
                        session_data = {
                            'session_id': row[0],
                            'user_id': row[1],
                            'runs_data': row[2],
                            'created_at': row[3],
                            'updated_at': row[4],
                            'snippet': self._extract_snippet(row[2], query, use_regex),
                            'match_count': self._count_matches(row[2], query, use_regex) if query else 0
                        }
                        processed_results.append(session_data)
                    
                    return processed_results
                    
        except Exception as e:
            console.print(f"[red]Error searching sessions: {str(e)}[/red]")
            return []
    
    def _extract_snippet(self, runs_data: Any, query: Optional[str], use_regex: bool = False) -> str:
        """
        Extract a relevant snippet from the conversation data.
        
        Args:
            runs_data: The conversation/runs data from the database
            query: Search query to highlight
            use_regex: Whether query is a regex pattern
            
        Returns:
            A snippet of the conversation with highlighted matches
        """
        try:
            # Convert runs data to string for searching
            content = str(runs_data) if runs_data else ""
            
            if not query or not content:
                # Return first 200 characters if no query
                return content[:200] + "..." if len(content) > 200 else content
            
            if use_regex:
                # Find regex matches
                matches = list(re.finditer(query, content, re.IGNORECASE))
                if matches:
                    match = matches[0]
                    start = max(0, match.start() - 100)
                    end = min(len(content), match.end() + 100)
                    snippet = content[start:end]
                    return f"...{snippet}..." if start > 0 or end < len(content) else snippet
            else:
                # Find keyword matches (case-insensitive)
                query_lower = query.lower()
                content_lower = content.lower()
                
                pos = content_lower.find(query_lower)
                if pos != -1:
                    start = max(0, pos - 100)
                    end = min(len(content), pos + len(query) + 100)
                    snippet = content[start:end]
                    return f"...{snippet}..." if start > 0 or end < len(content) else snippet
            
            # Fallback to beginning of content
            return content[:200] + "..." if len(content) > 200 else content
            
        except Exception:
            return "Error extracting snippet"
    
    def _count_matches(self, runs_data: Any, query: Optional[str], use_regex: bool = False) -> int:
        """
        Count the number of matches for the query in the conversation data.
        
        Args:
            runs_data: The conversation/runs data from the database
            query: Search query
            use_regex: Whether query is a regex pattern
            
        Returns:
            Number of matches found
        """
        try:
            content = str(runs_data) if runs_data else ""
            
            if not query or not content:
                return 0
            
            if use_regex:
                matches = re.findall(query, content, re.IGNORECASE)
                return len(matches)
            else:
                # Count case-insensitive keyword matches
                return content.lower().count(query.lower())
                
        except Exception:
            return 0
    
    def search_memories(
        self,
        user_id: str,
        query: str,
        use_regex: bool = False,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search through stored memories for the user.
        
        Args:
            user_id: User ID to search memories for
            query: Search query string
            use_regex: Whether to use regex pattern matching
            limit: Maximum number of results to return
            
        Returns:
            List of matching memories with metadata
        """
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cursor:
                    memory_table = "persistent_user_memories"
                    
                    if use_regex:
                        search_query = f"""
                        SELECT memory, created_at, updated_at
                        FROM {memory_table}
                        WHERE user_id = %s AND memory ~* %s
                        ORDER BY created_at DESC LIMIT %s
                        """
                        params = [user_id, query, limit]
                    else:
                        search_query = f"""
                        SELECT memory, created_at, updated_at
                        FROM {memory_table}
                        WHERE user_id = %s AND memory ILIKE %s
                        ORDER BY created_at DESC LIMIT %s
                        """
                        params = [user_id, f"%{query}%", limit]
                    
                    cursor.execute(search_query, params)
                    results = cursor.fetchall()
                    
                    return [
                        {
                            'memory': row[0],
                            'created_at': row[1],
                            'updated_at': row[2],
                            'match_count': self._count_matches(row[0], query, use_regex)
                        }
                        for row in results
                    ]
                    
        except Exception as e:
            console.print(f"[red]Error searching memories: {str(e)}[/red]")
            return []


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
        
        # Initialize search functionality
        self.searcher = SessionSearcher(db_url)

        console.print(
            "[cyan]Initializing PostgreSQL connections for persistent sessions...[/cyan]"
        )

        try:
            # Initialize memory database with PostgreSQL using retry logic
            self.memory_db = connect_with_retry(
                lambda: PostgresMemoryDb(
                    table_name="persistent_user_memories", db_url=db_url
                ),
                connection_name="PostgreSQL Memory Database"
            )

            # Initialize session storage with PostgreSQL using retry logic
            self.storage = connect_with_retry(
                lambda: PostgresStorage(
                    table_name="persistent_agent_sessions", db_url=db_url
                ),
                connection_name="PostgreSQL Session Storage"
            )

            # Create memory object
            self.memory = Memory(db=self.memory_db)

            # Initialize the agent
            self.agent = self._create_agent()

            console.print("[green]✓ All PostgreSQL connections established successfully[/green]")

        except Exception as e:
            console.print(f"[red]✗ Failed to establish PostgreSQL connections: {str(e)}[/red]")
            console.print(
                "[yellow]Troubleshooting tips:[/yellow]"
            )
            console.print(
                "[yellow]1. Ensure PostgreSQL is running[/yellow]"
            )
            console.print(
                "[yellow]2. Verify the database exists (CREATE DATABASE agno_memory;)[/yellow]"
            )
            console.print(
                "[yellow]3. Check your POSTGRES_URL environment variable[/yellow]"
            )
            console.print(
                f"[yellow]4. Current connection URL: {db_url}[/yellow]"
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
    
    def search_conversations(self, search_input: str):
        """
        Parse search command and execute search with various options.
        
        Supports:
        - search <query> - Basic keyword search
        - search --regex <pattern> - Regex pattern search
        - search --from YYYY-MM-DD --to YYYY-MM-DD - Date range search
        - search <query> --from YYYY-MM-DD - Combined search
        - search --memories <query> - Search memories instead of sessions
        """
        try:
            # Parse search command
            parts = search_input.strip().split()
            
            if len(parts) < 2:
                console.print("[yellow]Search usage examples:[/yellow]")
                console.print("• search <keywords> - Search for keywords in conversations")
                console.print("• search --regex <pattern> - Use regex pattern matching")
                console.print("• search --from 2024-01-01 --to 2024-12-31 - Date range search")
                console.print("• search <keywords> --from 2024-01-01 - Search with date filter")
                console.print("• search --memories <keywords> - Search stored memories")
                return
            
            # Initialize search parameters
            query = None
            start_date = None
            end_date = None
            use_regex = False
            search_memories = False
            
            i = 1  # Skip 'search' command
            while i < len(parts):
                arg = parts[i]
                
                if arg == "--regex":
                    use_regex = True
                    if i + 1 < len(parts) and not parts[i + 1].startswith("--"):
                        query = parts[i + 1]
                        i += 2
                    else:
                        i += 1
                elif arg == "--from":
                    if i + 1 < len(parts):
                        try:
                            start_date = datetime.strptime(parts[i + 1], "%Y-%m-%d")
                            i += 2
                        except ValueError:
                            console.print(f"[red]Invalid date format: {parts[i + 1]}. Use YYYY-MM-DD[/red]")
                            return
                    else:
                        console.print("[red]--from requires a date argument[/red]")
                        return
                elif arg == "--to":
                    if i + 1 < len(parts):
                        try:
                            end_date = datetime.strptime(parts[i + 1], "%Y-%m-%d")
                            i += 2
                        except ValueError:
                            console.print(f"[red]Invalid date format: {parts[i + 1]}. Use YYYY-MM-DD[/red]")
                            return
                    else:
                        console.print("[red]--to requires a date argument[/red]")
                        return
                elif arg == "--memories":
                    search_memories = True
                    if i + 1 < len(parts) and not parts[i + 1].startswith("--"):
                        query = parts[i + 1]
                        i += 2
                    else:
                        i += 1
                elif not arg.startswith("--"):
                    # Regular search query
                    if query is None:
                        query = arg
                    else:
                        query += " " + arg
                    i += 1
                else:
                    console.print(f"[red]Unknown option: {arg}[/red]")
                    return
            
            # Execute search
            if search_memories:
                if not query:
                    console.print("[red]Memory search requires a query[/red]")
                    return
                self._display_memory_search_results(query, use_regex)
            else:
                self._display_session_search_results(query, start_date, end_date, use_regex)
                
        except Exception as e:
            console.print(f"[red]Error processing search: {str(e)}[/red]")
    
    def _display_session_search_results(
        self,
        query: Optional[str],
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        use_regex: bool = False
    ):
        """Display session search results in a formatted table."""
        
        console.print(f"\n[bold cyan]Searching Sessions for User: {self.user_id}[/bold cyan]")
        
        # Build search description
        search_desc = []
        if query:
            search_desc.append(f"Query: '{query}'" + (" (regex)" if use_regex else ""))
        if start_date:
            search_desc.append(f"From: {start_date.strftime('%Y-%m-%d')}")
        if end_date:
            search_desc.append(f"To: {end_date.strftime('%Y-%m-%d')}")
        
        if search_desc:
            console.print(f"[yellow]Search criteria: {' | '.join(search_desc)}[/yellow]")
        else:
            console.print("[yellow]Showing all sessions[/yellow]")
        
        # Perform search
        results = self.searcher.search_sessions(
            user_id=self.user_id,
            query=query,
            start_date=start_date,
            end_date=end_date,
            use_regex=use_regex,
            limit=20
        )
        
        if not results:
            console.print("[yellow]No sessions found matching your criteria.[/yellow]")
            return
        
        # Create results table
        table = Table(title=f"Search Results ({len(results)} sessions)")
        table.add_column("Index", style="cyan", no_wrap=True, width=5)
        table.add_column("Session ID", style="green", width=20)
        table.add_column("Date", style="yellow", width=12)
        table.add_column("Matches", style="red", no_wrap=True, width=7)
        table.add_column("Snippet", style="white", width=60)
        
        for idx, result in enumerate(results, 1):
            # Format date
            created_date = result['created_at']
            if isinstance(created_date, datetime):
                date_str = created_date.strftime('%Y-%m-%d')
            else:
                date_str = str(created_date)[:10] if created_date else "Unknown"
            
            # Truncate session ID for display
            session_display = result['session_id'][-20:] if len(result['session_id']) > 20 else result['session_id']
            
            # Truncate snippet and highlight query terms
            snippet = result['snippet']
            if len(snippet) > 60:
                snippet = snippet[:57] + "..."
            
            # Highlight query terms in snippet (simple highlighting)
            if query and not use_regex:
                snippet = snippet.replace(
                    query, f"[bold red]{query}[/bold red]"
                ) if query.lower() in snippet.lower() else snippet
            
            table.add_row(
                str(idx),
                session_display,
                date_str,
                str(result['match_count']),
                snippet
            )
        
        console.print(table)
        
        # Store results for navigation
        self.set_search_results(results)
        
        # Show navigation options
        console.print(f"\n[bold cyan]Navigation Options:[/bold cyan]")
        console.print("• Type 'view <index>' to view full session details")
        console.print("• Type 'continue <index>' to switch to that session")
    
    def _display_memory_search_results(self, query: str, use_regex: bool = False):
        """Display memory search results."""
        
        console.print(f"\n[bold cyan]Searching Memories for: '{query}'[/bold cyan]")
        if use_regex:
            console.print("[yellow](Using regex pattern matching)[/yellow]")
        
        results = self.searcher.search_memories(
            user_id=self.user_id,
            query=query,
            use_regex=use_regex,
            limit=10
        )
        
        if not results:
            console.print("[yellow]No memories found matching your query.[/yellow]")
            return
        
        console.print(f"\n[bold green]Found {len(results)} matching memories:[/bold green]")
        
        for idx, result in enumerate(results, 1):
            memory_text = result['memory']
            match_count = result['match_count']
            
            # Highlight query terms in memory (simple highlighting)
            if not use_regex and query.lower() in memory_text.lower():
                highlighted = memory_text.replace(
                    query, f"[bold red]{query}[/bold red]"
                ) if query.lower() in memory_text.lower() else memory_text
            else:
                highlighted = memory_text
            
            console.print(
                Panel(
                    f"[yellow]Memory {idx} ({match_count} matches):[/yellow]\n{highlighted}",
                    expand=False,
                )
            )
    
    def set_search_results(self, results: List[Dict[str, Any]]):
        """Store search results for navigation purposes."""
        self.last_search_results = results
    
    def view_session_details(self, index: int):
        """View detailed information about a session from search results."""
        if not hasattr(self, 'last_search_results') or not self.last_search_results:
            console.print("[red]No search results available. Please run a search first.[/red]")
            return
        
        if index < 1 or index > len(self.last_search_results):
            console.print(f"[red]Invalid index. Please use a number between 1 and {len(self.last_search_results)}[/red]")
            return
        
        result = self.last_search_results[index - 1]
        
        console.print(f"\n[bold cyan]Session Details:[/bold cyan]")
        console.print(f"[yellow]Session ID:[/yellow] {result['session_id']}")
        console.print(f"[yellow]Created:[/yellow] {result['created_at']}")
        console.print(f"[yellow]Updated:[/yellow] {result.get('updated_at', 'N/A')}")
        console.print(f"[yellow]Match Count:[/yellow] {result['match_count']}")
        
        # Show more detailed snippet
        full_snippet = result.get('runs_data', '')
        if full_snippet:
            # Try to extract meaningful conversation content
            snippet_preview = str(full_snippet)[:1000]
            console.print(f"\n[yellow]Session Content Preview:[/yellow]")
            console.print(Panel(snippet_preview + ("..." if len(str(full_snippet)) > 1000 else ""), expand=False))
        
        console.print(f"\n[cyan]To switch to this session, type: continue {index}[/cyan]")
    
    def switch_to_session(self, index: int):
        """Switch the current session to a session from search results."""
        if not hasattr(self, 'last_search_results') or not self.last_search_results:
            console.print("[red]No search results available. Please run a search first.[/red]")
            return False
        
        if index < 1 or index > len(self.last_search_results):
            console.print(f"[red]Invalid index. Please use a number between 1 and {len(self.last_search_results)}[/red]")
            return False
        
        result = self.last_search_results[index - 1]
        new_session_id = result['session_id']
        
        console.print(f"[green]Switching to session: {new_session_id}[/green]")
        
        # Update the agent's session ID
        self.session_id = new_session_id
        self.agent.session_id = new_session_id
        
        console.print("[green]✓ Session switched successfully![/green]")
        console.print("[cyan]You can now continue the conversation from where it left off.[/cyan]")
        
        return True


class SessionManager:
    """Helper class for managing persistent sessions."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.db_url = PostgresConfig.get_db_url()

        # Initialize storage for session management with retry logic
        self.storage = connect_with_retry(
            lambda: PostgresStorage(
                table_name="persistent_agent_sessions", db_url=self.db_url
            ),
            connection_name="PostgreSQL Session Storage (SessionManager)"
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
        console.print("• 'search <query>' - Search conversations and memories")
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

                elif user_input.lower().startswith("search"):
                    agent.search_conversations(user_input)
                    continue

                elif user_input.lower().startswith("view "):
                    try:
                        index = int(user_input.split()[1])
                        agent.view_session_details(index)
                    except (IndexError, ValueError):
                        console.print("[red]Usage: view <index> (e.g., view 1)[/red]")
                    continue

                elif user_input.lower().startswith("continue "):
                    try:
                        index = int(user_input.split()[1])
                        if agent.switch_to_session(index):
                            console.print(f"[green]Now in session: {agent.session_id}[/green]")
                    except (IndexError, ValueError):
                        console.print("[red]Usage: continue <index> (e.g., continue 1)[/red]")
                    continue

                elif user_input.lower() in ["help", "commands"]:
                    console.print("\n[bold cyan]Available Commands:[/bold cyan]")
                    console.print("• Type your message to chat with the agent")
                    console.print("• 'memories' - Show stored memories")
                    console.print("• 'sessions' - Show all available sessions")
                    console.print("• 'summary' - Show current session summary")
                    console.print("• 'search <query>' - Search conversations and memories")
                    console.print("• 'help' - Show this help message")
                    console.print("• 'quit' or 'exit' - End the conversation")
                    console.print("\n[yellow]Search Examples:[/yellow]")
                    console.print("• search hello - Find conversations containing 'hello'")
                    console.print("• search --from 2024-01-01 - Sessions from a specific date")
                    console.print("• search --regex 'user.*name' - Use regex patterns")
                    console.print("• search --memories python - Search stored memories")
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
