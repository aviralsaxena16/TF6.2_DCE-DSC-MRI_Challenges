"""
Core scoring metric utilities for OSIPI evaluation.

This module contains pure metric computation logic extracted
from challengeScoring.py to improve modularity and testability.
"""

from .metrics import compute_cov
from .validator import validate_submission