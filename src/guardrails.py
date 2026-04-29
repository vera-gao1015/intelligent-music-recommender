"""
Guardrails module for the Intelligent Music Recommender.

Provides input validation, output verification, and logging to ensure
system reliability and prevent hallucinations or unsafe behavior.
"""

import re
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler("system.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("music-recommender")


# ---------------------------------------------------------------------------
# 1. Input Validation
# ---------------------------------------------------------------------------

MAX_QUERY_LENGTH = 500
MIN_QUERY_LENGTH = 2

BLOCKED_PATTERNS = [
    r"ignore\s+(previous|above|all)\s+(instructions|prompts)",
    r"you\s+are\s+now\s+a",
    r"act\s+as\s+(?:if|though)",
    r"system\s*prompt",
    r"<script",
]


def validate_input(query: str) -> Tuple[bool, str]:
    """
    Validate user input for safety and quality.

    Checks:
    - Not empty or too short
    - Not too long
    - No prompt injection attempts
    - Contains meaningful content

    Args:
        query: User's raw input string

    Returns:
        Tuple of (is_valid, message)
    """
    # Check empty or whitespace-only
    if not query or not query.strip():
        logger.warning("Input validation failed: empty query")
        return False, "Please enter a music request. For example: 'I want something chill to study to'"

    query_clean = query.strip()

    # Check length
    if len(query_clean) < MIN_QUERY_LENGTH:
        logger.warning(f"Input validation failed: too short ({len(query_clean)} chars)")
        return False, "Your request is too short. Please describe what kind of music you're looking for."

    if len(query_clean) > MAX_QUERY_LENGTH:
        logger.warning(f"Input validation failed: too long ({len(query_clean)} chars)")
        return False, f"Your request is too long (max {MAX_QUERY_LENGTH} characters). Please shorten it."

    # Check for prompt injection
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, query_clean, re.IGNORECASE):
            logger.warning(f"Input validation failed: blocked pattern detected")
            return False, "Your request contains unsupported content. Please describe what music you'd like."

    logger.info(f"Input validated: \"{query_clean[:50]}...\"")
    return True, "OK"


# ---------------------------------------------------------------------------
# 2. Output Verification (Anti-Hallucination)
# ---------------------------------------------------------------------------

def verify_output(
    recommendation: str,
    retrieved_songs: List[Tuple[Dict, float]],
) -> Tuple[bool, str, List[str]]:
    """
    Verify that the LLM's recommendation only references songs that were
    actually retrieved, preventing hallucinated song titles.

    Args:
        recommendation: LLM-generated recommendation text
        retrieved_songs: List of (song_dict, score) from RAG retrieval

    Returns:
        Tuple of (is_valid, cleaned_recommendation, warnings)
    """
    warnings = []

    # Get all retrieved song titles (lowercase for matching)
    valid_titles = {s["title"].lower() for s, _ in retrieved_songs}

    # Extract quoted song titles from the recommendation
    mentioned_titles = re.findall(r'"([^"]+)"', recommendation)

    for title in mentioned_titles:
        if title.lower() not in valid_titles:
            warnings.append(f"Hallucinated song detected: \"{title}\" was not in retrieved results")
            logger.warning(f"Output verification: hallucinated song \"{title}\"")

    if warnings:
        logger.warning(f"Output verification found {len(warnings)} issue(s)")
        return False, recommendation, warnings

    logger.info("Output verification passed: all songs are from retrieved results")
    return True, recommendation, warnings


# ---------------------------------------------------------------------------
# 3. Response Quality Check
# ---------------------------------------------------------------------------

def check_response_quality(recommendation: str, expected_count: int) -> Tuple[bool, List[str]]:
    """
    Check if the LLM response meets quality standards.

    Checks:
    - Response is not empty
    - Contains the expected number of recommendations
    - Each recommendation has an explanation (not just a title)

    Args:
        recommendation: LLM-generated text
        expected_count: Expected number of song recommendations

    Returns:
        Tuple of (passes_quality, issues)
    """
    issues = []

    # Check empty
    if not recommendation or len(recommendation.strip()) < 50:
        issues.append("Response is too short or empty")
        logger.warning("Quality check: response too short")
        return False, issues

    # Count numbered items (e.g., "1.", "2.", etc.)
    numbered_items = re.findall(r"^\d+\.", recommendation, re.MULTILINE)
    if len(numbered_items) < expected_count:
        issues.append(
            f"Expected {expected_count} recommendations but found {len(numbered_items)}"
        )
        logger.warning(f"Quality check: expected {expected_count}, found {len(numbered_items)}")

    # Check that explanations exist (each item should have more than just a title)
    lines = recommendation.strip().split("\n")
    short_items = 0
    for line in lines:
        if re.match(r"^\d+\.", line) and len(line) < 30:
            short_items += 1

    if short_items > 0:
        issues.append(f"{short_items} recommendation(s) lack detailed explanations")
        logger.warning(f"Quality check: {short_items} items lack explanations")

    if issues:
        return False, issues

    logger.info("Quality check passed")
    return True, issues


# ---------------------------------------------------------------------------
# 4. Full Guardrail Pipeline
# ---------------------------------------------------------------------------

def run_guardrails(
    query: str,
    recommendation: str,
    retrieved_songs: List[Tuple[Dict, float]],
    expected_count: int = 3,
) -> Dict:
    """
    Run all guardrail checks and return a comprehensive report.

    Args:
        query: Original user query
        recommendation: LLM-generated recommendation
        retrieved_songs: Songs retrieved by RAG
        expected_count: Expected number of recommendations

    Returns:
        Dict with validation results and overall pass/fail status
    """
    logger.info("Running guardrail checks...")

    # Input validation
    input_valid, input_msg = validate_input(query)

    # Output verification
    output_valid, cleaned_rec, output_warnings = verify_output(
        recommendation, retrieved_songs
    )

    # Quality check
    quality_pass, quality_issues = check_response_quality(
        recommendation, expected_count
    )

    # Overall status
    all_passed = input_valid and output_valid and quality_pass

    report = {
        "timestamp": datetime.now().isoformat(),
        "overall_pass": all_passed,
        "input_validation": {
            "passed": input_valid,
            "message": input_msg,
        },
        "output_verification": {
            "passed": output_valid,
            "warnings": output_warnings,
        },
        "quality_check": {
            "passed": quality_pass,
            "issues": quality_issues,
        },
    }

    if all_passed:
        logger.info("All guardrail checks PASSED")
    else:
        logger.warning("Some guardrail checks FAILED - see report for details")

    return report
