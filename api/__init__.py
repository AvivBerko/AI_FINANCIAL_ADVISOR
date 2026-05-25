"""FastAPI wrapper for the AI Financial Advisor backend.

Exposes the existing Python 4-stage ML pipeline (src/inference.py),
the yfinance live-price layer, and the ETF metadata/explanation
surfaces as HTTP endpoints consumable by the Next.js orchestrator.

The financial brain (src/) is untouched — this package is additive.
"""
