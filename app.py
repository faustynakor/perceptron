from flask import Flask, render_template, request, jsonify
import random
from data import digit_variants

app = Flask(__name__)

class Perceptron:
    def __init__(self, n_inputs, learning_rate=0.1):
        self.weights = [random.uniform(-1, 1) for _ in range(n_inputs)] 
        self.bias = random.uniform(-1, 1)
        self.lr = learning_rate

    def predict(self, x):
        activation = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias  
        return 1 if activation >= 0 else 0

    def fit(self, training_data, epochs=100):
        for _ in range(epochs):
            correct_predictions = 0
            for x, target in training_data:
                prediction = self.predict(x)
                error = target - prediction
                if error == 0:
                    correct_predictions += 1
                else:
                    self.weights = [w + self.lr * error * xi for w, xi in zip(self.weights, x)] 
                    self.bias += self.lr * error

def flatten(matrix):
    return [cell for row in matrix for cell in row]

digit_training_data = {}
digit_test_data = {}

for digit, variants in digit_variants.items():   
    variants_copy = variants.copy()
    random.shuffle(variants_copy)
    split_index = int(0.7 * len(variants_copy))
    digit_training_data[digit] = variants_copy[:split_index]
    digit_test_data[digit] = variants_copy[split_index:]

perceptrons = {digit: Perceptron(n_inputs=35) for digit in digit_variants.keys()}

for digit, perceptron in perceptrons.items():  
    training_data = []     
    for d, variants in digit_training_data.items():  
        target = 1 if d == digit else 0
        for variant in variants:
            training_data.append((flatten(variant), target)) 
    perceptron.fit(training_data, epochs=100) 

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    vector = data.get('vector')
    print("Otrzymany wektor:", vector)
    if not vector or len(vector) != 35:
        return jsonify({'error': 'Niepoprawny format wektora'}), 400
    results = {digit: perceptron.predict(vector) for digit, perceptron in perceptrons.items()}
    return jsonify({'predictions': results})

@app.route('/')
def index():
    return render_template('index.html', perceptrons=list(digit_variants.keys()))

if __name__ == '__main__':
    app.run(debug=True)
