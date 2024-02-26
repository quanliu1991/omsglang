export PYTHONPATH=/app/python
echo "this is /app/run.sh"
python3 python/sglang/launch_server.py &
python3 wsgi.py
