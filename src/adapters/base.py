from typing import Any, Dict, Optional, Protocol

from src.loop_engine import LoopReport, LoopSession


class LoopReportAdapter(Protocol):
    adapter_name: str
    adapter_schema_version: str

    def export_report(
        self,
        report: LoopReport,
        *,
        public: bool = True,
        session_id: Optional[str] = None,
        source_jsonl_line: Optional[int] = None,
    ) -> Dict[str, Any]:
        ...

    def export_session(
        self, session: LoopSession, *, public: bool = True
    ) -> Dict[str, Any]:
        ...
