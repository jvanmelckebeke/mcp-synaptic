"""Date and time utility functions."""

import re
from datetime import datetime, timedelta
from typing import Optional, Union

try:
    from croniter import croniter
except ImportError:
    croniter = None


def parse_ttl(ttl_str: str) -> int:
    """Parse a TTL string into seconds."""
    if not ttl_str:
        return 0
    
    # Handle numeric strings (assume seconds)
    if ttl_str.isdigit():
        return int(ttl_str)
    
    # Parse human-readable formats like "1h", "30m", "2d"
    pattern = r'^(\d+)([smhdw])$'
    match = re.match(pattern, ttl_str.lower())
    
    if not match:
        raise ValueError(f"Invalid TTL format: {ttl_str}")
    
    value, unit = match.groups()
    value = int(value)
    
    multipliers = {
        's': 1,           # seconds
        'm': 60,          # minutes
        'h': 3600,        # hours
        'd': 86400,       # days
        'w': 604800,      # weeks
    }
    
    return value * multipliers[unit]


def calculate_expiry(
    ttl_seconds: int,
    base_time: Optional[datetime] = None,
) -> datetime:
    """Calculate expiry time from TTL."""
    if base_time is None:
        base_time = datetime.utcnow()
    
    return base_time + timedelta(seconds=ttl_seconds)


def format_duration(seconds: int) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        if remaining_seconds == 0:
            return f"{minutes}m"
        else:
            return f"{minutes}m {remaining_seconds}s"
    elif seconds < 86400:
        hours = seconds // 3600
        remaining_minutes = (seconds % 3600) // 60
        if remaining_minutes == 0:
            return f"{hours}h"
        else:
            return f"{hours}h {remaining_minutes}m"
    else:
        days = seconds // 86400
        remaining_hours = (seconds % 86400) // 3600
        if remaining_hours == 0:
            return f"{days}d"
        else:
            return f"{days}d {remaining_hours}h"


def is_expired(
    expires_at: Optional[datetime],
    current_time: Optional[datetime] = None,
) -> bool:
    """Check if something has expired."""
    if expires_at is None:
        return False
    
    if current_time is None:
        current_time = datetime.utcnow()
    
    return current_time >= expires_at


def time_until_expiry(
    expires_at: Optional[datetime],
    current_time: Optional[datetime] = None,
) -> Optional[int]:
    """Get seconds until expiry, or None if no expiry."""
    if expires_at is None:
        return None
    
    if current_time is None:
        current_time = datetime.utcnow()
    
    delta = expires_at - current_time
    return max(0, int(delta.total_seconds()))


def next_cron_time(
    cron_expression: str,
    base_time: Optional[datetime] = None,
) -> Optional[datetime]:
    """Get the next execution time for a cron expression."""
    if croniter is None:
        raise ImportError("croniter not available. Install with: pip install croniter")
    
    if base_time is None:
        base_time = datetime.utcnow()
    
    try:
        cron = croniter(cron_expression, base_time)
        return cron.get_next(datetime)
    except Exception:
        return None


def format_timestamp(dt: datetime, include_microseconds: bool = False) -> str:
    """Format datetime to ISO string."""
    if include_microseconds:
        return dt.isoformat()
    else:
        return dt.replace(microsecond=0).isoformat()


def parse_timestamp(timestamp_str: str) -> datetime:
    """Parse ISO timestamp string to datetime."""
    try:
        return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
    except ValueError:
        # Try alternative formats
        formats = [
            '%Y-%m-%dT%H:%M:%S.%fZ',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%d %H:%M:%S.%f',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d',
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue
        
        raise ValueError(f"Unable to parse timestamp: {timestamp_str}")


def get_age_in_seconds(created_at: datetime, current_time: Optional[datetime] = None) -> int:
    """Get age of something in seconds."""
    if current_time is None:
        current_time = datetime.utcnow()
    
    delta = current_time - created_at
    return int(delta.total_seconds())


def is_recent(
    timestamp: datetime,
    threshold_seconds: int = 3600,
    current_time: Optional[datetime] = None,
) -> bool:
    """Check if a timestamp is recent (within threshold)."""
    age = get_age_in_seconds(timestamp, current_time)
    return age <= threshold_seconds