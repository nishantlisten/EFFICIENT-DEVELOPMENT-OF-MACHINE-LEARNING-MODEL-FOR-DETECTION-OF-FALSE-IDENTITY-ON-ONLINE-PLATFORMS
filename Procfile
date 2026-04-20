web: python backend/manage.py collectstatic --noinput && gunicorn --chdir backend backend.wsgi --log-file - --bind 0.0.0.0:$PORT
