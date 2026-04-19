"""
URL configuration for backend project.
"""
from django.contrib import admin
from django.urls import path, include
from django.views.generic import TemplateView
from django.conf import settings
from django.conf.urls.static import static
import os

# Serve individual frontend files
from django.http import FileResponse, Http404

def serve_frontend_file(request, filename):
    """Serve static files from the frontend directory during development."""
    filepath = os.path.join(settings.PROJECT_ROOT, "frontend", filename)
    if os.path.isfile(filepath):
        # Determine content type
        content_types = {
            '.css': 'text/css',
            '.js': 'application/javascript',
            '.html': 'text/html',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.svg': 'image/svg+xml',
            '.ico': 'image/x-icon',
        }
        ext = os.path.splitext(filename)[1].lower()
        content_type = content_types.get(ext, 'application/octet-stream')
        return FileResponse(open(filepath, 'rb'), content_type=content_type)
    raise Http404(f"File not found: {filename}")

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("api.urls")),
    # Serve frontend static files (CSS, JS, etc.)
    path("<str:filename>", serve_frontend_file, name="frontend-file"),
    # Serve frontend index.html at root
    path("", TemplateView.as_view(template_name="index.html"), name="home"),
]
