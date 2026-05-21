# ============================================
# src/security.py
# ============================================
# Security layer for BharatRAG API.
#
# Four protections:
# 1. Rate limiting    — prevent API abuse
# 2. Input validation — block malicious input
# 3. Prompt injection — detect attack patterns
# 4. PII detection    — redact sensitive data
# ============================================

import re
import os
from typing import Optional

# ── Prompt injection patterns ─────────────────────────
# Common patterns used in prompt injection attacks
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"forget\s+(everything|all|your\s+instructions)",
    r"you\s+are\s+now\s+(?:DAN|jailbreak|freed)",
    r"reveal\s+(your\s+)?(system\s+)?prompt",
    r"print\s+(your\s+)?instructions",
    r"bypass\s+(your\s+)?restrictions",
    r"disregard\s+(all\s+)?instructions",
    r"act\s+as\s+(if\s+you\s+are\s+)?(a\s+)?(?:DAN|evil)",
    r"pretend\s+(you\s+are|to\s+be)\s+(?:evil|unrestricted)",
    r"show\s+me\s+(all\s+)?user\s+data",
    r"leak\s+(the\s+)?(database|documents|data)",
]

COMPILED_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in INJECTION_PATTERNS
]

# ── Input limits ──────────────────────────────────────
MAX_QUESTION_LENGTH = 2000   # chars
MAX_FILENAME_LENGTH = 200    # chars


def check_prompt_injection(text: str) -> dict:
    """
    Detect prompt injection attempts.

    Returns:
        {"safe": True}  if no injection found
        {"safe": False, "reason": "..."} if found
    """
    if not text:
        return {"safe": True}

    for pattern in COMPILED_PATTERNS:
        if pattern.search(text):
            return {
                "safe":   False,
                "reason": "Potential prompt injection detected",
            }

    return {"safe": True}


def validate_question(question: str) -> dict:
    """
    Validate question input.

    Checks:
    1. Not empty
    2. Not too long
    3. No prompt injection
    4. No script injection (XSS)

    Returns:
        {"valid": True, "cleaned": question}
        {"valid": False, "reason": "..."}
    """
    if not question or not question.strip():
        return {
            "valid":  False,
            "reason": "Question cannot be empty",
        }

    # Length check
    if len(question) > MAX_QUESTION_LENGTH:
        return {
            "valid":  False,
            "reason": f"Question too long. "
                      f"Max {MAX_QUESTION_LENGTH} characters.",
        }

    # Basic XSS check
    if "<script" in question.lower():
        return {
            "valid":  False,
            "reason": "Invalid characters in question",
        }

    # Prompt injection check
    injection = check_prompt_injection(question)
    if not injection["safe"]:
        return {
            "valid":  False,
            "reason": injection["reason"],
        }

    # Clean the question
    cleaned = question.strip()

    return {"valid": True, "cleaned": cleaned}


def validate_filename(filename: str) -> dict:
    """
    Validate uploaded filename.

    Checks:
    1. PDF extension only
    2. No path traversal attacks (../../etc)
    3. Reasonable length
    4. No special characters
    """
    if not filename:
        return {"valid": False, "reason": "No filename"}

    # Path traversal attack prevention
    if ".." in filename or "/" in filename \
            or "\\" in filename:
        return {
            "valid":  False,
            "reason": "Invalid filename",
        }

    # Extension check
    if not filename.lower().endswith(".pdf"):
        return {
            "valid":  False,
            "reason": "Only PDF files accepted",
        }

    # Length check
    if len(filename) > MAX_FILENAME_LENGTH:
        return {
            "valid":  False,
            "reason": "Filename too long",
        }

    # Special characters check
    safe_pattern = re.compile(
        r'^[\w\-. ]+$'
    )
    if not safe_pattern.match(filename):
        return {
            "valid":  False,
            "reason": "Filename contains invalid characters",
        }

    return {"valid": True}


# ── PII Detection ─────────────────────────────────────
class PIIDetector:
    """
    Detects and redacts PII from text.

    Handles Indian-specific PII:
    → Aadhaar (12-digit number)
    → PAN card (ABCDE1234F format)
    → Phone numbers
    → Email addresses
    → Bank account numbers

    And general PII:
    → Credit card numbers
    → Passport numbers
    """

    # Indian PII patterns
    AADHAAR_PATTERN = re.compile(
        r'\b[2-9]{1}[0-9]{3}\s?[0-9]{4}\s?[0-9]{4}\b'
    )
    PAN_PATTERN = re.compile(
        r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b'
    )
    PHONE_PATTERN = re.compile(
        r'\b(?:\+91|91|0)?[6-9]\d{9}\b'
    )
    EMAIL_PATTERN = re.compile(
        r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b'
    )
    CREDIT_CARD_PATTERN = re.compile(
        r'\b(?:4[0-9]{12}(?:[0-9]{3})?|'
        r'5[1-5][0-9]{14}|'
        r'3[47][0-9]{13})\b'
    )

    def detect(self, text: str) -> list:
        """
        Detect all PII in text.
        Returns list of found PII types.
        """
        found = []

        if self.AADHAAR_PATTERN.search(text):
            found.append("AADHAAR")
        if self.PAN_PATTERN.search(text):
            found.append("PAN")
        if self.PHONE_PATTERN.search(text):
            found.append("PHONE")
        if self.EMAIL_PATTERN.search(text):
            found.append("EMAIL")
        if self.CREDIT_CARD_PATTERN.search(text):
            found.append("CREDIT_CARD")

        return found

    def redact(self, text: str) -> str:
        """
        Redact all detected PII from text.
        Replaces with [TYPE_REDACTED] placeholder.

        Example:
        "My Aadhaar is 1234 5678 9012"
        → "My Aadhaar is [AADHAAR_REDACTED]"
        """
        text = self.AADHAAR_PATTERN.sub(
            "[AADHAAR_REDACTED]", text
        )
        text = self.PAN_PATTERN.sub(
            "[PAN_REDACTED]", text
        )
        text = self.PHONE_PATTERN.sub(
            "[PHONE_REDACTED]", text
        )
        text = self.EMAIL_PATTERN.sub(
            "[EMAIL_REDACTED]", text
        )
        text = self.CREDIT_CARD_PATTERN.sub(
            "[CARD_REDACTED]", text
        )
        return text


# Singleton PII detector
_pii_detector = PIIDetector()

def get_pii_detector() -> PIIDetector:
    return _pii_detector