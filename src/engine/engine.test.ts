import test from 'node:test';
import assert from 'node:assert/strict';
import { MatrixOps } from './matrix';
import { Activation_ReLU, Activation_Sigmoid, Activation_Tanh, Activation_LeakyReLU } from './activations';
import { Activation_Custom } from './customActivation';
import { MathInspectorLogger } from './inspectorLogger';
import { DenseLayer } from './layers';
import { Loss_MSE, Loss_BinaryCrossEntropy } from './loss';
import { Optimizer_Adam, Optimizer_SGD } from './optimizers';
import { NeuralNetwork } from './model';

test('MatrixOps.dot computes correct matrix multiplication', () => {
  const a = [
    [1, 2],
    [3, 4],
  ];
  const b = [
    [2, 0],
    [1, 2],
  ];
  // [1*2 + 2*1, 1*0 + 2*2] = [4, 4]
  // [3*2 + 4*1, 3*0 + 4*2] = [10, 8]
  const c = MatrixOps.dot(a, b);
  assert.deepEqual(c, [
    [4, 4],
    [10, 8],
  ]);
});

test('MatrixOps.transpose correctly swaps rows and columns', () => {
  const a = [
    [1, 2, 3],
    [4, 5, 6],
  ];
  const t = MatrixOps.transpose(a);
  assert.deepEqual(t, [
    [1, 4],
    [2, 5],
    [3, 6],
  ]);
});

test('Activation_ReLU forward and backward derivative', () => {
  const relu = new Activation_ReLU();
  const input = [[-2, 0, 3]];
  const out = relu.forward(input);
  assert.deepEqual(out, [[0, 0, 3]]);

  const dvalues = [[1, 1, 1]];
  const dinputs = relu.backward(dvalues);
  assert.deepEqual(dinputs, [[0, 0, 1]]);
});

test('Activation_Sigmoid forward and backward derivative', () => {
  const sig = new Activation_Sigmoid();
  const input = [[0]];
  const out = sig.forward(input);
  assert.ok(Math.abs(out[0][0] - 0.5) < 1e-5);

  const dvalues = [[1]];
  const dinputs = sig.backward(dvalues);
  // Derivative at 0: 0.5 * (1 - 0.5) = 0.25
  assert.ok(Math.abs(dinputs[0][0] - 0.25) < 1e-5);
});

test('DenseLayer forward pass computes z = W · a + b', () => {
  const layer = new DenseLayer(2, 2);
  layer.weights = [
    [1, 2],
    [3, 4],
  ];
  layer.biases = [[0.5, -0.5]];

  const X = [[1, 1]];
  const z = layer.forward(X);
  // z = [1*1 + 1*3 + 0.5, 1*2 + 1*4 - 0.5] = [4.5, 5.5]
  assert.equal(z[0][0], 4.5);
  assert.equal(z[0][1], 5.5);
});

test('Loss_MSE calculates correct forward loss and backward gradients', () => {
  const lossFn = new Loss_MSE();
  const pred = [[2], [4]];
  const actual = [[1], [3]];
  // diff = [1, 1], sq = [1, 1], mean = 1
  const loss = lossFn.forward(pred, actual);
  assert.equal(loss, 1);

  const dvalues = lossFn.backward(pred, actual);
  // 2 * (pred - actual) / (2 * 1) = 2 * [1, 1] / 2 = [1, 1]
  assert.equal(dvalues[0][0], 1);
  assert.equal(dvalues[1][0], 1);
});

test('Optimizer_Adam updates weights and reduces loss over steps', () => {
  const model = new NeuralNetwork();
  const dense = new DenseLayer(2, 1, 0.5);
  model.add(dense);
  model.set(new Loss_MSE(), new Optimizer_Adam(0.1));

  const X = [
    [1, 0],
    [0, 1],
  ];
  const y = [[2], [3]];

  const initialLoss = model.loss_function!.forward(model.forward(X), y);

  // Train 100 steps
  for (let i = 0; i < 100; i++) {
    model.step(X, y);
  }

  const finalLoss = model.loss_function!.forward(model.forward(X), y);
  assert.ok(finalLoss < initialLoss, `Final loss ${finalLoss} should be lower than initial ${initialLoss}`);
});

test('NeuralNetwork successfully learns XOR non-linear mapping', () => {
  const model = new NeuralNetwork();
  // 2 inputs -> 8 hidden neurons (ReLU) -> 1 output (Sigmoid)
  model.add(new DenseLayer(2, 8, 0.5));
  model.add(new Activation_ReLU());
  model.add(new DenseLayer(8, 1, 0.5));
  model.add(new Activation_Sigmoid());

  model.set(new Loss_MSE(), new Optimizer_Adam(0.08));

  const X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
  ];
  const y = [[0], [1], [1], [0]];

  for (let epoch = 0; epoch < 600; epoch++) {
    model.step(X, y);
  }

  const preds = model.forward(X);
  // Check that XOR outputs are correctly separated (<0.4 for 0, >0.6 for 1)
  const isXorLearned =
    preds[0][0] < 0.4 &&
    preds[1][0] > 0.6 &&
    preds[2][0] > 0.6 &&
    preds[3][0] < 0.4;

  assert.ok(isXorLearned, `XOR should be separated: got ${preds.map(p => p[0].toFixed(2)).join(', ')}`);
});

test('MathInspectorLogger records backprop gradients and updates', () => {
  const logger = new MathInspectorLogger();
  const model = new NeuralNetwork();
  model.add(new DenseLayer(2, 1));
  model.set(new Loss_MSE(), new Optimizer_SGD(0.05));
  model.setLogger(logger);

  const X = [[1.0, 0.5]];
  const y = [[2.0]];

  model.step(X, y);

  const latest = logger.getLatest();
  assert.ok(latest !== null, 'Logger should have recorded the epoch');
  assert.equal(latest.epoch, 1);
  assert.ok(latest.formattedSummary.includes('[1] Forward Pass'), 'Summary should contain Forward Pass section');
  assert.ok(latest.formattedSummary.includes('[2] Loss Gradient'), 'Summary should contain Backpropagation section');
  assert.ok(latest.formattedSummary.includes('dWeights = Xᵀ • dZ'), 'Summary should detail Chain Rule');
});

test('Activation_Custom (Swish & Siren) computes correct forward and backward derivatives', () => {
  const swish = new Activation_Custom('swish');
  const out = swish.forward([[0.0]]);
  // Swish(0) = 0 * sig(0) = 0
  assert.equal(out[0][0], 0);

  // Swish derivative at 0: sig(0) + 0 = 0.5
  const dinputs = swish.backward([[1.0]]);
  assert.ok(Math.abs(dinputs[0][0] - 0.5) < 1e-4);

  // Siren: sin(0) = 0, derivative cos(0) = 1
  const siren = new Activation_Custom('siren');
  const sirenOut = siren.forward([[0.0]]);
  assert.equal(sirenOut[0][0], 0);
  const sirenGrad = siren.backward([[1.0]]);
  assert.ok(Math.abs(sirenGrad[0][0] - 1.0) < 1e-4);
});

