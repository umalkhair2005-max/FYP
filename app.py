# -*- coding: utf-8 -*-
"""
Run Pneumonia AI Lab from the project folder:

    python app.py

Same as:  python backend/app.py
Opens http://127.0.0.1:5000/ in your browser when the server is ready (unless NO_BROWSER=1).
"""
from __future__ import annotations

import os
import runpy

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_BACKEND_APP = os.path.join(_PROJECT_ROOT, "backend", "app.py")

if not os.path.isfile(_BACKEND_APP):
    raise SystemExit(f"Missing backend app: {_BACKEND_APP}")

runpy.run_path(_BACKEND_APP, run_name="__main__")
