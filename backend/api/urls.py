"""
API URL Configuration
"""
from django.urls import path
from . import views

urlpatterns = [
    path("predict-profile/", views.predict_profile, name="predict-profile"),
    path("detect-ai-text/", views.detect_ai_text_view, name="detect-ai-text"),
    path("analyze/", views.analyze, name="analyze"),
    path("analyze-url/", views.analyze_url, name="analyze-url"),
    path("analyze-posts/", views.analyze_posts, name="analyze-posts"),
]
