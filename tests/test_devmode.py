from clavenar_agent_sdk.devmode import render_deny_panel
from clavenar_agent_sdk.errors import ClavenarDenied


def test_renders_detector_breakdown_when_detail_present() -> None:
    denied = ClavenarDenied(
        tool_name="send_email",
        reasons=["indirect prompt injection"],
        review_reasons=[],
        intent_category="Exfiltration",
        layer="brain",
        correlation_id="abc-123",
        detail={
            "detectors": [
                {"detector": "persona_drift", "score": 0.12},
                {"detector": "injection", "score": 0.91, "flagged": True},
            ],
            "degraded": ["injection"],
        },
    )
    panel = render_deny_panel(denied)
    assert "send_email" in panel
    assert "layer=brain" in panel
    assert "correlation=abc-123" in panel
    assert "injection" in panel
    assert "0.91" in panel
    assert "⚠ flagged" in panel
    assert "degraded: injection" in panel


def test_hints_to_enable_verbose_verdicts_when_detail_absent() -> None:
    denied = ClavenarDenied(
        tool_name="wire_transfer",
        reasons=["policy denied"],
        review_reasons=[],
        intent_category="Direct Execution",
    )
    panel = render_deny_panel(denied)
    assert "wire_transfer" in panel
    assert "CLAVENAR_PROXY_VERBOSE_VERDICTS" in panel
