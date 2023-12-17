import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor


MODEL_PATH = 'inception_v3_brain_tumor_classifier.tflite'
CLASSES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
INDEX_KEY = 'index'

preprocessor = create_preprocessor('inception_v3', target_size=(256, 256))


def setup_tflite_interpreter(model_path):
    try:
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    except Exception as e:
        raise RuntimeError(f"Error setting up TFLite interpreter: {e}")

    input_index = interpreter.get_input_details()[0][INDEX_KEY]
    output_index = interpreter.get_output_details()[0][INDEX_KEY]

    return interpreter, input_index, output_index


def predict(url, model_path, preprocessor, classes):
    X = preprocessor.from_url(url)
    interpreter, input_index, output_index = setup_tflite_interpreter(model_path)
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_index)
    return dict(zip(classes, predictions[0].tolist()))


def lambda_handler(event, context):
    url = event['url']
    result = predict(url, model_path=MODEL_PATH, preprocessor=preprocessor, classes=CLASSES)
    return result
