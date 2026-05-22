"""
SignalCollector abstract base class and SignalCollectionError.

Design decisions:

1. WHY ABC OVER PROTOCOL:
   abc.ABC + @abstractmethod enforces the contract at instantiation time with a
   clear TypeError. typing.Protocol is structural (duck typing) and would allow
   accidental satisfaction of the interface without explicit intent. With 6+
   concrete collector implementations, explicit opt-in via inheritance is safer.

2. WHY source IS AN ABSTRACT PROPERTY:
   Each collector must declare which SignalSource it represents. An abstract
   property forces this at class definition — the class literally cannot be
   instantiated without it. Concrete classes satisfy it with a class attribute:
       source = SignalSource.GOOGLE_TAKEOUT
   Python's ABCMeta allows class attributes to satisfy abstract property
   requirements (CPython docs, abc module). This is the canonical pattern.

3. WHY collect() IS ASYNC:
   All collectors do I/O: file reads, API calls, database queries. Even file
   collectors use asyncio.to_thread() to avoid blocking the event loop. Async
   interface here means the LangGraph pipeline can run multiple collectors
   concurrently via asyncio.gather() without adding concurrency logic later.

4. WHY ONE EXCEPTION TYPE (SignalCollectionError):
   Pipeline nodes catch SignalCollectionError to handle any collector failure
   gracefully (log, continue with other sources, emit AuditEntry). If we raised
   FileNotFoundError, JSONDecodeError, etc., every caller would need to catch
   multiple types. One domain exception simplifies pipeline error handling.
   Exception chaining (raise ... from cause) preserves the original traceback.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from trace.models import RawSignal, SignalSource


class SignalCollectionError(Exception):
    """
    Raised by any SignalCollector when collection fails at runtime.

    Covers: missing files, malformed data, API errors, network timeouts.
    NOT for programming errors (those remain as TypeError, AttributeError, etc).

    source attribute: which collector raised this, for structured logging.
    """

    def __init__(self, source: SignalSource, message: str) -> None:
        self.source = source
        super().__init__(f"[{source.value}] {message}")


class SignalCollector(ABC):
    """
    Abstract base for all signal collectors.

    Subclass contract:
      1. Declare `source = SignalSource.<VALUE>` as a class attribute.
         This value must be stamped onto every RawSignal the collector produces.
         Python ABCMeta: a class attribute satisfies an abstract property.

      2. Implement `async collect() -> list[RawSignal]`.
         Return empty list if the source has no signals (not an error).
         Raise SignalCollectionError on any runtime failure.
         Never raise other exception types for expected failure modes.

    Example minimal implementation:
        class MyCollector(SignalCollector):
            source = SignalSource.FILESYSTEM

            async def collect(self) -> list[RawSignal]:
                ...
    """

    @property
    @abstractmethod
    def source(self) -> SignalSource:
        """
        Identifies which data source this collector reads from.
        Override with a class attribute in each concrete subclass.
        """
        ...

    @abstractmethod
    async def collect(self) -> list[RawSignal]:
        """
        Collect all available signals from this source.

        Returns:
            List of RawSignal objects. May be empty; never None.

        Raises:
            SignalCollectionError: if the source is unreachable, malformed,
                                   or any other runtime failure occurs.
        """
        ...
