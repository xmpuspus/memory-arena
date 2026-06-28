FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY memory_arena/ memory_arena/
COPY cypher/ cypher/
# The smoke corpus + result snapshot ship inside memory_arena/data/ (bundled
# above); the loaders resolve to them via memory_arena/paths.py. The top-level
# datasets/ tree is gitignored (only placeholder question YAMLs are tracked),
# so it is intentionally not copied into the image.

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "memory_arena.chatbot.api:app", "--host", "0.0.0.0", "--port", "8000"]
