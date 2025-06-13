"""
Test script for database connection retry logic
"""

import os
import sys
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from session_memory_example import connect_with_retry
from rich.console import Console

console = Console()


def test_successful_connection():
    """Test successful connection on first attempt"""
    console.print("\n[bold blue]Test 1: Successful connection on first attempt[/bold blue]")
    
    # Mock a successful connection
    mock_func = Mock(return_value="Successful Connection")
    
    result = connect_with_retry(mock_func, connection_name="Test Database")
    
    assert result == "Successful Connection"
    assert mock_func.call_count == 1
    console.print("[green]✓ Test passed: Connection succeeded on first attempt[/green]")


def test_connection_retry_succeeds():
    """Test connection that fails twice then succeeds"""
    console.print("\n[bold blue]Test 2: Connection fails twice, then succeeds[/bold blue]")
    
    # Mock a function that fails twice then succeeds
    attempt_count = 0
    
    def connection_func():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise Exception(f"Connection failed (attempt {attempt_count})")
        return "Successful Connection"
    
    result = connect_with_retry(connection_func, connection_name="Test Database", base_delay=1)
    
    assert result == "Successful Connection"
    assert attempt_count == 3
    console.print("[green]✓ Test passed: Connection succeeded after 2 retries[/green]")


def test_all_retries_fail():
    """Test when all retry attempts fail"""
    console.print("\n[bold blue]Test 3: All retry attempts fail[/bold blue]")
    
    # Mock a function that always fails
    mock_func = Mock(side_effect=Exception("Connection refused"))
    
    try:
        connect_with_retry(mock_func, max_retries=2, base_delay=1, connection_name="Test Database")
        assert False, "Should have raised an exception"
    except Exception as e:
        assert "Connection refused" in str(e)
        assert mock_func.call_count == 2
        console.print("[green]✓ Test passed: Exception raised after all retries failed[/green]")


def test_with_actual_postgres_simulation():
    """Test with simulated PostgreSQL connection scenarios"""
    console.print("\n[bold blue]Test 4: Simulated PostgreSQL connection scenarios[/bold blue]")
    
    # Test with invalid database URL (should fail all retries)
    from agno.memory.v2.db.postgres import PostgresMemoryDb
    
    console.print("\n[yellow]Testing with invalid database URL...[/yellow]")
    
    try:
        # Use an invalid URL that will definitely fail
        result = connect_with_retry(
            lambda: PostgresMemoryDb(
                table_name="test_table",
                db_url="postgresql://invalid:invalid@nonexistent:5432/test"
            ),
            max_retries=2,
            base_delay=1,
            connection_name="Invalid PostgreSQL"
        )
        console.print("[red]✗ Test failed: Should have raised an exception[/red]")
    except Exception as e:
        console.print(f"[green]✓ Test passed: Properly handled connection failure: {type(e).__name__}[/green]")


if __name__ == "__main__":
    console.print("[bold cyan]Running connection retry logic tests...[/bold cyan]")
    
    try:
        test_successful_connection()
        test_connection_retry_succeeds()
        test_all_retries_fail()
        test_with_actual_postgres_simulation()
        
        console.print("\n[bold green]All tests completed successfully![/bold green]")
        
    except AssertionError as e:
        console.print(f"\n[red]Test failed: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Unexpected error: {e}[/red]")
        sys.exit(1)