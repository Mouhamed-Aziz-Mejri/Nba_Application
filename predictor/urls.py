from django.urls import path
from .views import PredictView, HealthView, TeamsView, RosterView, SimilarPlayersView, PlayersSearchView, similar_page
from .views import index, teams_page, roster_page, login_page

urlpatterns = [
    # ── HTML pages ────────────────────────────────────────────────────────────
    path('',                            index,                  name='index'),
    path('teams/',                      teams_page,             name='teams'),
    path('teams/<str:team_code>/',      roster_page,            name='roster'),
    path('similar/',                       similar_page,           name='similar'),
    path('login/',                         login_page,             name='login'),

    # ── API endpoints ─────────────────────────────────────────────────────────
    path('api/health/',                 HealthView.as_view(),   name='health'),
    path('api/predict/',                PredictView.as_view(),  name='predict'),
    path('api/teams/',                  TeamsView.as_view(),    name='api-teams'),
    path('api/teams/<str:team_code>/',          RosterView.as_view(),         name='api-roster'),
    path('api/similar/<str:player_name>/',      SimilarPlayersView.as_view(), name='api-similar'),
    path('api/players/',                        PlayersSearchView.as_view(),  name='api-players'),
]