FROM public.ecr.aws/lambda/python:3.10

RUN pip install keras-image-helper

RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl

COPY inception_v3_brain_tumor_classifier.tflite .

COPY src/predict.py .

CMD [ "predict.lambda_handler" ]
