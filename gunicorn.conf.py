"""
Gunicorn configuration file.
Automatically read by Gunicorn on startup to ensure proper port binding on Render/Heroku/Cloud platforms.
"""

import os

# Dynamic port binding from environment ($PORT on Render/Heroku, default 5000)
port = os.environ.get("PORT", "5000")
bind = f"0.0.0.0:{port}"

# Memory-safe configuration for Render Free Tier (512MB RAM limit)
workers = 1
threads = 2
timeout = 120
worker_class = "gthread"

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
