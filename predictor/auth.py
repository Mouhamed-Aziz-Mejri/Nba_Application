"""
predictor/auth.py
-----------------
Custom DRF authentication class that uses Django sessions
but skips the CSRF enforcement that SessionAuthentication
applies by default — even when CsrfViewMiddleware is removed.
"""

from rest_framework.authentication import SessionAuthentication


class CsrfExemptSessionAuthentication(SessionAuthentication):
    """Session auth without CSRF enforcement."""

    def enforce_csrf(self, request):
        # Skip CSRF check entirely — our views handle auth manually
        return