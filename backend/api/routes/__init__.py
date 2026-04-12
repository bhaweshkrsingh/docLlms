"""Register all routes by importing them — FastAPI decorators fire on import."""
from backend.api.routes._shared import app  # noqa: F401
from backend.api.routes import route_health, route_chat  # noqa: F401
