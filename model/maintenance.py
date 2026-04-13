import json
import os
from typing import Dict, List, Tuple
from .clap import NORMAL_LABEL, classify_audio_path_two_pass

try:
    from litellm import completion
except Exception:
    completion = None

_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini/gemini-1.5-flash")

_ISSUE_KEYWORDS = {
    "continuous low-frequency rumble": "Possible imbalance, misalignment, or loosened mounting.",
    "intermittent metallic clacking": "Possible loose component, coupling wear, or impact contact.",
    "repetitive tapping or knocking": "Possible bearing wear, shaft play, or cyclic mechanical strike.",
    "high-pitched squeal from friction": "Possible belt slip, dry bearing, or friction due to poor lubrication.",
    "air leak or hissing noise": "Possible pneumatic leak, valve leakage, or seal degradation.",
    "grinding noise from worn bearings": "Probable bearing deterioration requiring immediate inspection.",
}

_SEVERE_LABELS = {
    "grinding noise from worn bearings",
    "high-pitched squeal from friction",
    "intermittent metallic clacking",
    "repetitive tapping or knocking",
}


def _priority_from_label_score(label: str, score: float, issue_detected: bool) -> str:
    if not issue_detected:
        return "NONE"
    if label in _SEVERE_LABELS and score >= 0.35:
        return "HIGH"
    if score >= 0.20:
        return "MEDIUM"
    return "LOW"


def _detect_issue(top_label: str, top_score: float) -> bool:
    if top_label == "normal machine operating hum":
        return False
    if top_label in _ISSUE_KEYWORDS and top_score >= 0.10:
        return True
    return False


def _fallback_description(
    top_label: str,
    top_score: float,
    issue_detected: bool,
    priority: str,
) -> str:
    if not issue_detected:
        return (
            "The dominant acoustic signature appears consistent with normal machine operation. "
            "No immediate maintenance issue is indicated from this sample."
        )
    hypothesis = _ISSUE_KEYWORDS.get(top_label, "Potential abnormal mechanical condition detected.")
    return (
        f"The most likely issue is: {hypothesis} "
        f"Confidence signal from CLAP top class is {top_score:.3f}. "
        f"Recommended maintenance priority is {priority}."
    )


def _gemini_description(audio_path: str, ranked: List[Tuple[str, float]], issue_detected: bool, priority: str) -> str:
    if completion is None:
        return _fallback_description(ranked[0][0], ranked[0][1], issue_detected, priority)

    top_items = [{"label": label, "score": round(score, 4)} for label, score in ranked[:5]]
    prompt_payload = {
        "audio_path": audio_path,
        "top_clap_results": top_items,
        "issue_detected": issue_detected,
        "priority": priority,
        "instructions": (
            "Write 3-5 concise sentences explaining the most probable maintenance issue correlated "
            "with the sound, and include a practical next action."
        ),
    }
    try:
        response = completion(
            model=_GEMINI_MODEL,
            temperature=0,
            max_tokens=220,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a predictive-maintenance assistant. "
                        "Use the CLAP ranking as evidence and avoid overclaiming."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt_payload)},
            ],
        )
        content = response.choices[0].message.content
        return content.strip() if content else _fallback_description(ranked[0][0], ranked[0][1], issue_detected, priority)
    except Exception:
        return _fallback_description(ranked[0][0], ranked[0][1], issue_detected, priority)


def analyze_maintenance(audio_path: str) -> Dict:
    two_pass = classify_audio_path_two_pass(audio_path)
    ranked = two_pass["ranked_results"]
    if not ranked:
        raise ValueError("No analysis result produced for audio.")

    top_label, top_score = ranked[0]
    if not two_pass["is_abnormal"]:
        issue_detected = False
        priority = "NONE"
        description = (
            "FFNN gate classified this clip as normal machine operation. "
            "CLAP deep analysis was skipped."
        )
    else:
        issue_detected = _detect_issue(top_label, top_score)
        priority = _priority_from_label_score(top_label, top_score, issue_detected)
        description = _gemini_description(audio_path, ranked, issue_detected, priority)

    return {
        "audio_path": audio_path,
        "top_label": top_label if two_pass["is_abnormal"] else NORMAL_LABEL,
        "top_score": top_score if two_pass["is_abnormal"] else 1.0,
        "issue_detected": issue_detected,
        "maintenance_priority": priority,
        "ranked_results": ranked,
        "description": description,
        "analysis_stage": two_pass["stage"],
    }


def format_maintenance_report(report: Dict) -> str:
    lines = [
        f"Issue detected: {'YES' if report['issue_detected'] else 'NO'}",
        f"Maintenance priority: {report['maintenance_priority']}",
        f"Top CLAP signal: {report['top_label']} ({report['top_score']:.4f})",
        "",
        "CLAP ranking:",
    ]
    for label, score in report["ranked_results"]:
        lines.append(f"- {label}: {score:.4f}")
    lines.extend(["", "Gemini diagnosis:", report["description"]])
    return "\n".join(lines)
