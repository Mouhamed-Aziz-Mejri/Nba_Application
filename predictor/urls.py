from django.urls import path
from .views import (
    # HTML pages
    index, login_page, teams_page, roster_page, similar_page,
    playoff_page, fantasy_page, fantasy_debug,
    # Auth endpoints
    login_view, register_view, logout_view, me_view,
    # API endpoints
    PredictView, HealthView, TeamsView, RosterView,
    SimilarPlayersView, PlayersSearchView,
    PlayoffSimulateView, PlayoffSeriesView,
    # Fantasy endpoints
    FantasyPlayersView, FantasyMyTeamView,
    FantasyAddPlayerView, FantasyDropPlayerView,
    FantasyLeaderboardView,
)

urlpatterns = [
    # ── HTML pages ──────────────────────────────────────────────────────────
    path('',                            index,           name='index'),
    path('login/',                      login_page,      name='login'),
    path('teams/',                      teams_page,      name='teams'),
    path('teams/<str:team_code>/',      roster_page,     name='roster'),
    path('similar/',                    similar_page,    name='similar'),
    path('playoffs/',                   playoff_page,    name='playoffs'),
    path('fantasy/',                    fantasy_page,    name='fantasy'),
    path('api/fantasy/debug/',              fantasy_debug,   name='api-fantasy-debug'),

    # ── Auth endpoints ───────────────────────────────────────────────────────
    path('auth/login/',                 login_view,      name='auth-login'),
    path('auth/register/',              register_view,   name='auth-register'),
    path('auth/logout/',                logout_view,     name='auth-logout'),
    path('auth/me/',                    me_view,         name='auth-me'),

    # ── Core API ─────────────────────────────────────────────────────────────
    path('api/health/',                                     HealthView.as_view(),          name='health'),
    path('api/predict/',                                    PredictView.as_view(),         name='predict'),
    path('api/teams/',                                      TeamsView.as_view(),           name='api-teams'),
    path('api/teams/<str:team_code>/',                      RosterView.as_view(),          name='api-roster'),
    path('api/similar/<str:player_name>/',                  SimilarPlayersView.as_view(),  name='api-similar'),
    path('api/players/',                                    PlayersSearchView.as_view(),   name='api-players'),

    # ── Playoff API ───────────────────────────────────────────────────────────
    path('api/playoff/simulate/',                           PlayoffSimulateView.as_view(), name='api-playoff-simulate'),
    path('api/playoff/series/<str:home_code>/<str:away_code>/', PlayoffSeriesView.as_view(), name='api-playoff-series'),

    # ── Fantasy API ───────────────────────────────────────────────────────────
    path('api/fantasy/players/',        FantasyPlayersView.as_view(),     name='api-fantasy-players'),
    path('api/fantasy/my-team/',        FantasyMyTeamView.as_view(),      name='api-fantasy-myteam'),
    path('api/fantasy/add/',            FantasyAddPlayerView.as_view(),   name='api-fantasy-add'),
    path('api/fantasy/drop/',           FantasyDropPlayerView.as_view(),  name='api-fantasy-drop'),
    path('api/fantasy/leaderboard/',    FantasyLeaderboardView.as_view(), name='api-fantasy-leaderboard'),
]