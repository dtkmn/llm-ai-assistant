from typing import Any, Dict, Protocol

from src.loop_engine import LoopReport, LoopSession


class LoopReportAdapter(Protocol):
    adapter_name: str
    adapter_schema_version: str

    def export_report(self, report: LoopReport, *, public: bool = True) -> Dict[str, Any]:
        ...

    def export_session(
        self, session: LoopSession, *, public: bool = True
    ) -> Dict[str, Any]:
        ...
