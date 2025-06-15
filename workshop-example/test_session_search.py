"""
Test suite for session search functionality
"""

import os
import sys
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from session_memory_example import SessionSearcher, PersistentSessionAgent
from rich.console import Console

console = Console()


def test_session_searcher_init():
    """Test SessionSearcher initialization"""
    console.print("\n[bold blue]Test 1: SessionSearcher initialization[/bold blue]")
    
    db_url = "postgresql://test:test@localhost:5432/test"
    searcher = SessionSearcher(db_url)
    
    assert searcher.db_url == db_url
    assert searcher.table_name == "persistent_agent_sessions"
    
    # Test custom table name
    searcher_custom = SessionSearcher(db_url, "custom_table")
    assert searcher_custom.table_name == "custom_table"
    
    console.print("[green]✓ Test passed: SessionSearcher initialized correctly[/green]")


def test_extract_snippet():
    """Test snippet extraction functionality"""
    console.print("\n[bold blue]Test 2: Snippet extraction[/bold blue]")
    
    searcher = SessionSearcher("postgresql://test:test@localhost:5432/test")
    
    # Test with query match
    content = "This is a long conversation about Python programming and data science. The user asked about machine learning algorithms."
    snippet = searcher._extract_snippet(content, "Python")
    
    assert "Python" in snippet
    assert len(snippet) <= 300  # Should be reasonably sized
    
    # Test with no query
    snippet_no_query = searcher._extract_snippet(content, None)
    assert len(snippet_no_query) <= 203  # 200 + "..."
    
    # Test with regex
    snippet_regex = searcher._extract_snippet(content, "data.*science", use_regex=True)
    assert "data science" in snippet_regex.lower()
    
    console.print("[green]✓ Test passed: Snippet extraction working correctly[/green]")


def test_count_matches():
    """Test match counting functionality"""
    console.print("\n[bold blue]Test 3: Match counting[/bold blue]")
    
    searcher = SessionSearcher("postgresql://test:test@localhost:5432/test")
    
    content = "Python is great. I love Python programming. Python Python Python!"
    
    # Test regular keyword counting
    count = searcher._count_matches(content, "Python")
    assert count == 5
    
    # Test case insensitive
    count_lower = searcher._count_matches(content, "python")
    assert count_lower == 5
    
    # Test regex counting
    count_regex = searcher._count_matches(content, "Python+", use_regex=True)
    assert count_regex == 5
    
    # Test no matches
    count_none = searcher._count_matches(content, "JavaScript")
    assert count_none == 0
    
    console.print("[green]✓ Test passed: Match counting working correctly[/green]")


def test_search_command_parsing():
    """Test search command parsing"""
    console.print("\n[bold blue]Test 4: Search command parsing[/bold blue]")
    
    # Mock PersistentSessionAgent
    with patch('session_memory_example.PostgresConfig'), \
         patch('session_memory_example.connect_with_retry'), \
         patch('session_memory_example.Memory'), \
         patch('session_memory_example.Agent'):
        
        agent = PersistentSessionAgent(user_id="test_user", session_id="test_session")
        
        # Mock the searcher methods
        agent.searcher = Mock()
        agent.searcher.search_sessions.return_value = []
        agent.searcher.search_memories.return_value = []
        
        # Test basic search
        with patch('session_memory_example.console'):
            agent.search_conversations("search hello")
            agent.searcher.search_sessions.assert_called_once()
            
        # Reset mock
        agent.searcher.reset_mock()
        
        # Test memory search
        with patch('session_memory_example.console'):
            agent.search_conversations("search --memories python")
            agent.searcher.search_memories.assert_called_once()
    
    console.print("[green]✓ Test passed: Search command parsing working correctly[/green]")


def test_search_results_navigation():
    """Test search results navigation functionality"""
    console.print("\n[bold blue]Test 5: Search results navigation[/bold blue]")
    
    # Mock PersistentSessionAgent
    with patch('session_memory_example.PostgresConfig'), \
         patch('session_memory_example.connect_with_retry'), \
         patch('session_memory_example.Memory'), \
         patch('session_memory_example.Agent'):
        
        agent = PersistentSessionAgent(user_id="test_user", session_id="test_session")
        
        # Mock search results
        mock_results = [
            {
                'session_id': 'test_session_1',
                'created_at': datetime.now(),
                'updated_at': datetime.now(),
                'match_count': 3,
                'runs_data': 'Sample conversation data'
            },
            {
                'session_id': 'test_session_2',
                'created_at': datetime.now(),
                'updated_at': datetime.now(),
                'match_count': 1,
                'runs_data': 'Another conversation'
            }
        ]
        
        agent.set_search_results(mock_results)
        
        # Test valid session switch
        with patch('session_memory_example.console'):
            result = agent.switch_to_session(1)
            assert result == True
            assert agent.session_id == 'test_session_1'
        
        # Test invalid index
        with patch('session_memory_example.console'):
            result = agent.switch_to_session(5)
            assert result == False
        
        # Test view session details
        with patch('session_memory_example.console'):
            agent.view_session_details(2)  # Should not raise exception
    
    console.print("[green]✓ Test passed: Search results navigation working correctly[/green]")


def test_date_filtering():
    """Test date range filtering in search"""
    console.print("\n[bold blue]Test 6: Date filtering[/bold blue]")
    
    # Test date parsing in search command
    with patch('session_memory_example.PostgresConfig'), \
         patch('session_memory_example.connect_with_retry'), \
         patch('session_memory_example.Memory'), \
         patch('session_memory_example.Agent'):
        
        agent = PersistentSessionAgent(user_id="test_user", session_id="test_session")
        
        # Mock the searcher
        agent.searcher = Mock()
        agent.searcher.search_sessions.return_value = []
        
        # Test date range search
        with patch('session_memory_example.console'):
            agent.search_conversations("search --from 2024-01-01 --to 2024-12-31")
            
            # Verify search was called with date parameters
            call_args = agent.searcher.search_sessions.call_args
            assert call_args[1]['start_date'] is not None
            assert call_args[1]['end_date'] is not None
            assert call_args[1]['start_date'].year == 2024
            assert call_args[1]['end_date'].year == 2024
    
    console.print("[green]✓ Test passed: Date filtering working correctly[/green]")


def test_regex_search():
    """Test regex search functionality"""
    console.print("\n[bold blue]Test 7: Regex search[/bold blue]")
    
    with patch('session_memory_example.PostgresConfig'), \
         patch('session_memory_example.connect_with_retry'), \
         patch('session_memory_example.Memory'), \
         patch('session_memory_example.Agent'):
        
        agent = PersistentSessionAgent(user_id="test_user", session_id="test_session")
        
        # Mock the searcher
        agent.searcher = Mock()
        agent.searcher.search_sessions.return_value = []
        
        # Test regex search
        with patch('session_memory_example.console'):
            agent.search_conversations("search --regex user.*name")
            
            # Verify regex flag was set
            call_args = agent.searcher.search_sessions.call_args
            assert call_args[1]['use_regex'] == True
            assert call_args[1]['query'] == 'user.*name'
    
    console.print("[green]✓ Test passed: Regex search working correctly[/green]")


def test_error_handling():
    """Test error handling in search functionality"""
    console.print("\n[bold blue]Test 8: Error handling[/bold blue]")
    
    searcher = SessionSearcher("postgresql://invalid:invalid@nonexistent:5432/test")
    
    # Test search with invalid database connection
    results = searcher.search_sessions("test_user", "test query")
    assert results == []  # Should return empty list on error
    
    # Test memory search with invalid connection
    memory_results = searcher.search_memories("test_user", "test query")
    assert memory_results == []  # Should return empty list on error
    
    console.print("[green]✓ Test passed: Error handling working correctly[/green]")


def test_integration_search_workflow():
    """Test complete search workflow integration"""
    console.print("\n[bold blue]Test 9: Integration - Complete search workflow[/bold blue]")
    
    # This test simulates a complete user workflow
    with patch('session_memory_example.PostgresConfig'), \
         patch('session_memory_example.connect_with_retry'), \
         patch('session_memory_example.Memory'), \
         patch('session_memory_example.Agent'):
        
        agent = PersistentSessionAgent(user_id="test_user", session_id="test_session")
        
        # Mock successful search results
        mock_results = [
            {
                'session_id': 'found_session_1',
                'created_at': datetime.now() - timedelta(days=1),
                'updated_at': datetime.now() - timedelta(days=1),
                'match_count': 5,
                'runs_data': 'This conversation contains the search term',
                'snippet': 'This conversation contains the search term...'
            }
        ]
        
        agent.searcher = Mock()
        agent.searcher.search_sessions.return_value = mock_results
        
        # Simulate complete workflow
        with patch('session_memory_example.console'):
            # 1. Perform search
            agent.search_conversations("search conversation")
            
            # 2. View details
            agent.view_session_details(1)
            
            # 3. Switch to session
            success = agent.switch_to_session(1)
            assert success == True
            assert agent.session_id == 'found_session_1'
    
    console.print("[green]✓ Test passed: Complete search workflow working correctly[/green]")


if __name__ == "__main__":
    console.print("[bold cyan]Running session search functionality tests...[/bold cyan]")
    
    try:
        test_session_searcher_init()
        test_extract_snippet()
        test_count_matches()
        test_search_command_parsing()
        test_search_results_navigation()
        test_date_filtering()
        test_regex_search()
        test_error_handling()
        test_integration_search_workflow()
        
        console.print("\n[bold green]All search functionality tests completed successfully![/bold green]")
        console.print("\n[yellow]Note: These tests use mocked database connections.[/yellow]")
        console.print("[yellow]For full integration testing, ensure PostgreSQL is running with test data.[/yellow]")
        
    except AssertionError as e:
        console.print(f"\n[red]Test failed: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Unexpected error: {e}[/red]")
        sys.exit(1)