from django.urls import path
from .views import PredictView, HealthView, TeamsView, RosterView
from .views import index, teams_page, roster_page

urlpatterns = [
    # ── HTML pages ────────────────────────────────────────────────────────────
    path('',                            index,                  name='index'),
    path('teams/',                      teams_page,             name='teams'),
    path('teams/<str:team_code>/',      roster_page,            name='roster'),

    # ── API endpoints ─────────────────────────────────────────────────────────
    path('api/health/',                 HealthView.as_view(),   name='health'),
    path('api/predict/',                PredictView.as_view(),  name='predict'),
    path('api/teams/',                  TeamsView.as_view(),    name='api-teams'),
    path('api/teams/<str:team_code>/',  RosterView.as_view(),   name='api-roster'),
]