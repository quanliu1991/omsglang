FROM registry.linker.cc/linker/sglang_base_line:v0.0.0
WORKDIR /app
COPY . .
ENTRYPOINT ["bash", "/app/run.sh"]
