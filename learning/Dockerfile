FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    libpq5 \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get update && apt-get install -y \
    wget \
    gnupg2 \
    curl \
    && wget -qO - http://repo.radeon.com/rocm/rocm.gpg.key | apt-key add - \
    && echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/5.3/ ubuntu main' | tee /etc/apt/sources.list.d/rocm.list \
    && apt-get update && apt-get install -y \
    rocm-libs rocm-dev \
    && apt-get clean

# Add the ROCm libraries to the library path
ENV LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "simplex.py", "/app/goodware", "/app/malware", "/app/goodware-opcodes", "/app/malware-opcodes"]