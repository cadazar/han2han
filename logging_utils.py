"""Logging utilities for distributed training."""

def log_from_main_process(logger, level, message, *args, **kwargs):
    """Log only from main process to prevent console flooding."""
    try:
        import jax
        if jax.process_index() == 0:
            getattr(logger, level)(message, *args, **kwargs)
    except (NameError, ImportError):
        # jax not imported yet, just log normally
        getattr(logger, level)(message, *args, **kwargs)

def log_from_all_processes(logger, level, message, *args, **kwargs):
    """Log from all processes with process ID prefix for critical sync information."""
    try:
        import jax
        process_id = jax.process_index()
        prefixed_message = f"[Process {process_id}] {message}"
        getattr(logger, level)(prefixed_message, *args, **kwargs)
    except (NameError, ImportError):
        # jax not imported yet, just log normally
        getattr(logger, level)(message, *args, **kwargs)