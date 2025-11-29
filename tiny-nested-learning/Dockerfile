FROM tensorflow/tensorflow:2.15.0

RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

ENV POETRY_VERSION=1.7.1
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

COPY . .

RUN poetry config virtualenvs.in-project true && \
    poetry install --no-interaction --no-ansi

CMD ["poetry", "run", "python", "main.py"]
