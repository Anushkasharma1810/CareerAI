"""
app.py
------
CareerAI Flask application entry point.

Usage:
    python backend/app.py

Or with flask CLI:
    FLASK_APP=backend/app.py flask run --port 5000
"""

import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flask import Flask, jsonify
from flask_cors import CORS

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def create_app() -> Flask:
    app = Flask(__name__)

    # CORS: allow React dev server
    CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "http://127.0.0.1:3000"]}})

    # ── Register blueprints ────────────────────────────────────────────────
    from backend.routes.api import api_bp
    app.register_blueprint(api_bp, url_prefix="/")

    # ── Error handlers ─────────────────────────────────────────────────────
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"status": "error", "error": "Endpoint not found"}), 404

    @app.errorhandler(405)
    def method_not_allowed(e):
        return jsonify({"status": "error", "error": "Method not allowed"}), 405

    @app.errorhandler(413)
    def file_too_large(e):
        return jsonify({"status": "error", "error": "File too large. Maximum 10 MB."}), 413

    @app.errorhandler(500)
    def internal_error(e):
        logger.exception("Unhandled 500 error")
        return jsonify({"status": "error", "error": "Internal server error"}), 500

    # ── Pre-load ML model at startup ───────────────────────────────────────
    with app.app_context():
        try:
            from src.inference.predictor import get_predictor
            get_predictor()
            logger.info("ML model loaded successfully.")
        except Exception as e:
            logger.warning(f"ML model not loaded at startup: {e}")
            logger.warning("Run `python src/training/train.py` to train the model.")

    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = False
    app = create_app()
    logger.info(f"Starting CareerAI backend on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
