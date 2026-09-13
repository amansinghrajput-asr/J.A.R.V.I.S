"""Views package for J.A.R.V.I.S PySide6 Desktop HUD.

Contains multi-view navigation screens:
- ActivityView (Index 1)
- SystemView (Index 2)
- SettingsView (Index 3)
"""

from app.ui.views.activity_view import ActivityView
from app.ui.views.settings_view import SettingsView
from app.ui.views.system_view import SystemView

__all__ = [
    "ActivityView",
    "SettingsView",
    "SystemView",
]
