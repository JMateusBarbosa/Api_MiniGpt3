from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.models import load_model

app = Flask(__name__)

# Carregando o modelo pré-treinado
modelo = load_model('miniGpt3.h5')

@app.route('/generate_text', methods=['POST'])
def generate_text():
    input_text = request.json['input_text']
    num_tokens_generated = request.json.get('num_tokens_generated', 40)

    
    input_tokens = tokenize_input(input_text, vectorize_layer)
    start_tokens = input_tokens

    num_tokens_generated = max(1, num_tokens_generated)

    num_tokens_generated += len(input_tokens)

    text_generated = []

    for _ in range(num_tokens_generated):
        x = np.array([start_tokens])
        y, _ = modelo.predict(x)
        predicted_token = text_gen_callback.sample_from(y[0][-1])
        start_tokens.append(predicted_token)
        text_generated.append(predicted_token)

    generated_text = " ".join([text_gen_callback.detokenize(_) for _ in input_tokens + text_generated])
    
    return jsonify({"generated_text": generated_text})

if __name__ == '__main__':
    app.run(debug=True)
