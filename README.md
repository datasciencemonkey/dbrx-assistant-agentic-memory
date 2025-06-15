# DBRX Assistant Agentic Memory

A collection of examples demonstrating persistent memory capabilities for AI agents using the Agno framework with PostgreSQL backend. This repository showcases how to build AI assistants that remember conversations and user context across multiple sessions.

## Overview

This repository contains examples of implementing memory systems for AI agents, ranging from basic memory storage to advanced multi-agent systems with persistent PostgreSQL-backed memory. The main highlight is a session-based memory system that allows users to have continuous conversations with AI assistants that remember past interactions.

## Features

- **Persistent Memory**: Store and retrieve conversation history across application restarts
- **User-Specific Sessions**: Each user gets their own memory space with UUID-based identification
- **Session Management**: Automatic session continuation or creation based on time intervals
- **PostgreSQL Backend**: Reliable, scalable storage for long-term memory persistence
- **Connection Resilience**: Automatic retry logic with exponential backoff for database connections
- **Session Search & Filtering**: Search through conversation history by keywords, date ranges, and regex patterns
- **Multi-Agent Support**: Examples showing memory sharing between multiple agents
- **Interactive CLI**: User-friendly command-line interface with rich formatting

## Repository Structure

```
.
├── workshop-example/                 # ⭐ MAIN DEMO - Start here!
│   ├── session_memory_example.py    # Persistent session memory with PostgreSQL
│   ├── test_retry_connection.py     # Test suite for connection retry logic
│   └── test_session_search.py       # Test suite for session search functionality
├── other-examples/                  # Additional examples
│   ├── basic_memory_example.py      # Simple memory usage
│   ├── advanced_memory_example.py   # Advanced memory features
│   ├── memory_example.py            # Core memory concepts
│   └── multi_agent_memory_example.py # Multiple agents sharing memory
└── README.md
```

> **📍 Quick Start**: The main demonstration is in the `workshop-example/` directory. This example showcases the full capabilities of persistent memory with user sessions.

## Prerequisites

1. **Python 3.8+**
2. **PostgreSQL Database** (running and accessible)
3. **OpenAI API Key** (for GPT-4 access)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/datasciencemonkey/dbrx-assistant-agentic-memory.git
cd dbrx-assistant-agentic-memory
```

2. Install required packages:
```bash
pip install agno openai psycopg2-binary python-dotenv rich
```

3. Set up PostgreSQL database:
```sql
CREATE DATABASE agno_memory;
```

4. Create a `.env` file in the root directory:
```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# PostgreSQL Configuration (choose one method)
# Method 1: Full connection URL
POSTGRES_URL=postgresql://username:password@localhost:5432/agno_memory

# Method 2: Individual components
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_DB=agno_memory
```

## Quick Start

### Running the Session Memory Example

The main example demonstrates a persistent chat assistant that remembers users across sessions:

```bash
python workshop-example/session_memory_example.py <username>

# Example:
python workshop-example/session_memory_example.py john_doe
```

### Available Commands

During a chat session, you can use these commands:
- **Regular chat**: Just type your message
- `memories` - Display all stored memories for your user
- `sessions` - Show all available sessions
- `summary` - Display current session information
- `search <query>` - Search conversations and memories (see search examples below)
- `help` - Show available commands
- `quit` or `exit` - End the conversation

### Connection Resilience

The system includes automatic connection retry logic with exponential backoff:
- **Automatic Retries**: Up to 3 connection attempts with 2, 4, and 8 second delays
- **Progress Indicators**: Visual feedback during retry attempts  
- **Clear Error Messages**: Helpful troubleshooting tips when connections fail
- **Graceful Degradation**: Proper error handling for persistent connection issues

### Session Search & Filtering

The system provides powerful search capabilities to find past conversations:

#### Basic Search Commands
```bash
search hello                    # Find conversations containing "hello"
search machine learning         # Search for multiple keywords
search --memories python        # Search stored memories instead of sessions
```

#### Date Range Filtering
```bash
search --from 2024-01-01                    # Sessions from specific date onward
search --to 2024-12-31                      # Sessions up to specific date
search --from 2024-01-01 --to 2024-03-31   # Sessions within date range
search python --from 2024-01-01            # Combine keyword and date filtering
```

#### Advanced Pattern Matching
```bash
search --regex 'user.*name'                # Use regex patterns
search --regex '\b(python|javascript)\b'   # Advanced regex for specific terms
```

#### Navigation Commands
After searching, you can navigate through results:
```bash
view 1          # View detailed information about search result #1
continue 2      # Switch to session from search result #2
```

#### Search Features
- **Highlighted Results**: Search terms are highlighted in results
- **Match Counting**: Shows number of matches per session
- **Content Snippets**: Preview relevant parts of conversations
- **Date Sorting**: Results sorted by creation date (newest first)
- **Session Navigation**: Easy switching between found sessions

### Example Usage

```bash
$ python workshop-example/session_memory_example.py alice

Persistent Session Example (PostgreSQL)

Username: alice
User UUID: 5b1e3542-8d6f-5c9e-b7f2-123456789abc
Initializing PostgreSQL connections for persistent sessions...
Attempting to connect to PostgreSQL Memory Database...
✓ Successfully connected to PostgreSQL Memory Database
Attempting to connect to PostgreSQL Session Storage...
✓ Successfully connected to PostgreSQL Session Storage
✓ All PostgreSQL connections established successfully
Starting new session for user 'alice': alice_20250113_143022_a1b2c3d4

Welcome, alice! Starting a new conversation...
This is your first session. The agent will remember our conversation for next time.

[alice]: Hi! My favorite color is blue and I love hiking.