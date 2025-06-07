"""Async utility functions."""

import asyncio
from typing import Any, Awaitable, Callable, List, TypeVar

T = TypeVar('T')


async def run_with_timeout(
    coro: Awaitable[T],
    timeout_seconds: float,
    default_value: T = None,
) -> T:
    """Run a coroutine with a timeout, returning default value if timeout occurs."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        return default_value


async def gather_with_concurrency(
    tasks: List[Awaitable[T]],
    max_concurrency: int = 10,
) -> List[T]:
    """Run multiple coroutines with limited concurrency."""
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def _run_with_semaphore(task: Awaitable[T]) -> T:
        async with semaphore:
            return await task
    
    # Wrap all tasks with semaphore
    limited_tasks = [_run_with_semaphore(task) for task in tasks]
    
    return await asyncio.gather(*limited_tasks)


async def retry_with_backoff(
    func: Callable[[], Awaitable[T]],
    max_retries: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
) -> T:
    """Retry a function with exponential backoff."""
    delay = base_delay
    
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except Exception as e:
            if attempt == max_retries:
                raise e
            
            await asyncio.sleep(min(delay, max_delay))
            delay *= backoff_factor


class AsyncContextManager:
    """Base class for async context managers."""
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


def create_task_with_error_handling(
    coro: Awaitable[T],
    error_handler: Callable[[Exception], None] = None,
    task_name: str = None,
) -> asyncio.Task[T]:
    """Create a task with automatic error handling."""
    
    async def _wrapped_coro():
        try:
            return await coro
        except Exception as e:
            if error_handler:
                error_handler(e)
            else:
                # Log the error or handle it appropriately
                print(f"Task error {task_name or 'unnamed'}: {e}")
            raise
    
    task = asyncio.create_task(_wrapped_coro())
    if task_name:
        task.set_name(task_name)
    
    return task