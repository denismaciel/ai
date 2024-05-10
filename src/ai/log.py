import sys
import logging
import structlog
from structlog.typing import Processor


def configure_logging() -> None:
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt='iso'),
        # send_to_sentry,
    ]

    # Pretty print if running on a terminal
    if sys.stderr.isatty():
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        processors.extend(
            [
                structlog.processors.dict_tracebacks,
                structlog.processors.JSONRenderer(),
            ]
        )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(logging.NOTSET),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    if not structlog.is_configured():
        structlog.configure()

    if name is None:
        return structlog.stdlib.get_logger()

    return structlog.stdlib.get_logger(name)
