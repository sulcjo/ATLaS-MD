"""Live-dashboard screen: context, panels, views, frame assembly."""

from .context import DashboardContext, build_context
from .screen import render_screen
from .sidecar import SidecarCache, SidecarSnapshot

__all__ = [
    "DashboardContext", "SidecarCache", "SidecarSnapshot", "build_context", "render_screen",
]
