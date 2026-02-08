from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import datetime
import uuid

@dataclass
class TagInstance:
    tag_type: str  # TAG01, TAG02, etc.
    description: str
    severity: float = 0.0  # 0.0 to 1.0
    time_range: tuple = (0.0, 0.0) # start, end
    reason_text: str = ""

@dataclass
class TestResult:
    test_id: str  # T1, T2...
    test_name: str
    audio_file_path: Optional[str] = None
    video_file_path: Optional[str] = None
    
    # Analysis Data (Time-series)
    pitch_track_time: List[float] = field(default_factory=list)
    pitch_track_hz: List[float] = field(default_factory=list)
    energy_track_time: List[float] = field(default_factory=list)
    energy_track_rms: List[float] = field(default_factory=list)
    
    # Derived Metrics (Scalar)
    pitch_accuracy_cents: float = 0.0 # Error from target
    pitch_stability_cents: float = 0.0 # Std dev
    pitch_drift_cents: float = 0.0 # Start vs End
    attack_overshoot_score: float = 0.0 
    
    tags: List[TagInstance] = field(default_factory=list)
    processed_at: Optional[datetime.datetime] = None

@dataclass
class StudentSession:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    
    # Student Info
    student_name: str = ""
    part: str = "Soprano"
    coach_name: str = ""
    
    # Configuration
    passaggio_info: Dict[str, str] = field(default_factory=dict)
    
    # Results
    results: Dict[str, TestResult] = field(default_factory=dict) # Key: test_id
    
    # Summary
    summary_tags: List[TagInstance] = field(default_factory=list) # Top 3
    coach_comment: str = ""
    routine_assignment: str = ""
    
    pdf_report_path: Optional[str] = None

    def add_result(self, result: TestResult):
        self.results[result.test_id] = result

    def get_result(self, test_id: str) -> Optional[TestResult]:
        return self.results.get(test_id)
