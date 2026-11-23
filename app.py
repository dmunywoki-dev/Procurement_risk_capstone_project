"""Vercel entrypoint for the TGEE FastAPI service and project landing page."""
from pathlib import Path

from fastapi.responses import FileResponse, JSONResponse

from api_service import app

ROOT = Path(__file__).resolve().parent
LANDING_PAGE = ROOT / "public" / "index.html"


@app.get("/", include_in_schema=False)
def landing_page():
    if not LANDING_PAGE.is_file():
        return JSONResponse(
            {"detail": "Landing page asset is missing."}, status_code=404
        )
    return FileResponse(LANDING_PAGE, media_type="text/html")
