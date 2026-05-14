from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = 'django-insecure-nba-project-change-this-in-production'

DEBUG = True

ALLOWED_HOSTS = ['*']

# ── Add these to your existing nba_project/settings.py ──────────────────────
#
# 1. Make sure 'django.contrib.auth' and 'django.contrib.sessions'
#    are in INSTALLED_APPS (they usually are by default):
#
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',          # ← required
    'django.contrib.contenttypes',  # ← required by auth
    'django.contrib.sessions',      # ← required for login sessions
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'corsheaders',
    'predictor',
]

# 2. Make sure SessionMiddleware and AuthenticationMiddleware are in MIDDLEWARE:
MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',   # ← required
    'django.middleware.common.CommonMiddleware',
    # 'django.middleware.csrf.CsrfViewMiddleware',            # ← keep disabled
    'django.contrib.auth.middleware.AuthenticationMiddleware', # ← required
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

# 3. Session settings (add these):
SESSION_ENGINE        = 'django.contrib.sessions.backends.db'
SESSION_COOKIE_AGE    = 60 * 60 * 24 * 30   # 30 days
SESSION_COOKIE_NAME   = 'nba_sessionid'
LOGIN_URL             = '/login/'
LOGIN_REDIRECT_URL    = '/'



ROOT_URLCONF = 'nba_project.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        # ↓ tells Django to look inside each app's templates/ folder
        'DIRS': [BASE_DIR / 'predictor' / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'nba_project.wsgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

STATIC_URL = 'static/'
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

REST_FRAMEWORK = {
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
    ],
}

# ── CORS — allow requests from the HTML test page ────────────────────────────
CORS_ALLOW_ALL_ORIGINS = True
CORS_ALLOW_HEADERS = [
    'content-type',
    'accept',
    'authorization',
    'x-requested-with',
]
# ── ADD THESE LINES TO nba_project/settings.py ──────────────────────────────

# Session configuration
SESSION_COOKIE_SAMESITE = 'Lax'
SESSION_COOKIE_HTTPONLY = False
SESSION_COOKIE_SECURE   = False        # False for local dev
SESSION_COOKIE_AGE      = 60 * 60 * 24 * 30

# CORS — must allow credentials so fetch() sends session cookie
CORS_ALLOW_CREDENTIALS   = True

# Django REST Framework — AllowAny globally, views handle auth manually
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.SessionAuthentication',
        'rest_framework.authentication.BasicAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.AllowAny',
    ],
}