from django.db import models
from django.contrib.auth.models import User


class FantasyLeague(models.Model):
    """A season-long fantasy league."""
    name        = models.CharField(max_length=100, default="NBA Fantasy 2024-25")
    season      = models.CharField(max_length=10, default="2024-25")
    budget_cap  = models.IntegerField(default=100_000_000)   # $100M salary cap
    max_players = models.IntegerField(default=8)              # roster size
    created_at  = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


class FantasyTeam(models.Model):
    """One user's fantasy team in a league."""
    user        = models.OneToOneField(User, on_delete=models.CASCADE, related_name="fantasy_team")
    league      = models.ForeignKey(FantasyLeague, on_delete=models.CASCADE, related_name="teams")
    name        = models.CharField(max_length=80)
    total_score = models.FloatField(default=0.0)
    total_spent = models.IntegerField(default=0)
    created_at  = models.DateTimeField(auto_now_add=True)
    updated_at  = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-total_score"]

    def __str__(self):
        return f"{self.name} ({self.user.username})"

    def recalculate_score(self):
        """Recompute total_score and total_spent from roster entries."""
        entries = self.roster.all()
        self.total_score = round(sum(e.fantasy_score for e in entries), 2)
        self.total_spent = sum(e.salary for e in entries)
        self.save()


class FantasyRoster(models.Model):
    """A player slot on a fantasy team."""
    POSITIONS = [
        ("PG", "Point Guard"),
        ("SG", "Shooting Guard"),
        ("SF", "Small Forward"),
        ("PF", "Power Forward"),
        ("C",  "Center"),
        ("BENCH", "Bench"),
    ]

    team          = models.ForeignKey(FantasyTeam, on_delete=models.CASCADE, related_name="roster")
    player_name   = models.CharField(max_length=100)
    team_code     = models.CharField(max_length=5)
    position      = models.CharField(max_length=6, choices=POSITIONS, default="BENCH")
    salary        = models.IntegerField(default=0)
    fantasy_score = models.FloatField(default=0.0)

    # raw stats snapshot
    pts  = models.FloatField(default=0)
    ast  = models.FloatField(default=0)
    reb  = models.FloatField(default=0)
    stl  = models.FloatField(default=0)
    blk  = models.FloatField(default=0)
    tov  = models.FloatField(default=0)
    fg_pct  = models.FloatField(default=0)
    three_pct = models.FloatField(default=0)
    ft_pct  = models.FloatField(default=0)
    mp   = models.FloatField(default=0)

    added_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("team", "player_name")
        ordering = ["-fantasy_score"]

    def __str__(self):
        return f"{self.player_name} → {self.team.name}"