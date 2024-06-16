FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

RUN pip install -U pip

RUN pip install pandas

RUN pip install numpy

RUN pip install scikit-learn==1.5.0

RUN pip install pyarrow

WORKDIR /app
COPY [ "starter.py", "./" ]

RUN [ "python", "starter.py", "2023", "05" ]
