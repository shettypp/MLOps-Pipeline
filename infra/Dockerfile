# Lambda container image (zip limit of 250 MB is too small for sklearn + lightgbm + xgboost)
FROM public.ecr.aws/lambda/python:3.12
RUN dnf install -y libgomp && dnf clean all   # OpenMP runtime for LightGBM/XGBoost
COPY requirements-lambda.txt .
RUN pip install --no-cache-dir -r requirements-lambda.txt
COPY src/ ${LAMBDA_TASK_ROOT}/src/
COPY data/ ${LAMBDA_TASK_ROOT}/data/
COPY infra/__init__.py infra/lambda_handlers.py ${LAMBDA_TASK_ROOT}/infra/
CMD ["infra.lambda_handlers.api_handler"]
