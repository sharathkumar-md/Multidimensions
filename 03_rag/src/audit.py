import json
import logging
from datetime import datetime
from pathlib import Path

from config.settings import settings

# Create log directory if it doesn't exist
settings.audit_log_file.parent.mkdir(parents=True, exist_ok=True)

def log_audit_event(
    user_id: str,
    session_id: str,
    question: str,
    route: str,
    response_time_ms: int,
    success: bool,
    error_message: str = "",
) -> None:
    """
    Appends a structured JSON audit log to the audit_log_file.
    This tracks usage, routing decisions, and errors for analytics.
    """
    event = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "user_id": user_id,
        "session_id": session_id,
        "question": question,
        "route": route,
        "response_time_ms": response_time_ms,
        "success": success,
        "error_message": error_message,
    }
    
    try:
        with open(settings.audit_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as e:
        # Fallback to standard logging if file write fails
        logging.getLogger(__name__).error(f"Failed to write audit log: {e}")
