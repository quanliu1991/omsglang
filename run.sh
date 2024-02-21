export PYTHONPATH=/app/python
echo "this is /app/run.sh"
python python/sglang/launch_server.py &
python wsgi.py
