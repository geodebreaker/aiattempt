var FPS = 30;
var nn;
var td;

function start() {
  nn = new NeuralNetwork([1, 2, 3]);
}

function loop(dt) {

}

class NeuralNetwork {
  constructor(layers) {
    this.layersSizes = layers;
    this.layers = [];
    this.bpact = [];
    this.learnRate = 0.5;

    for (var i = 1; i < layers.length; i++) {
      this.layers.push(
        new Array(layers[i]).fill(0).map(x => new Array(layers[i - 1] + 1).fill(0).map(x => Math.random() * 2 - 1))
      )
    }
  }

  run(fninput) {
    var input = fninput;
    this.bpact = [];
    for (var l of this.layers) {
      input.push(1);
      this.bpact.push(input);
      input = l.map(n => activateFn(dotarr(input, n)));
    }
    this.bpact.push(input);
    return input;
  }

  error(input) {
    return this.run(input.x).map((x, i) => input.y[i] - x);
  }

  train(input) {
    var error = [this.error(input)];
    var grad = [];

    for (let i = this.layers.length - 1; i >= 0; i--) {
      grad.unshift(error[0].map((e, j) => e * aFnDeriv(this.bpact[i + 1][j])));
      if (i > 0) {
        var herror = [];
        for (var j = 0; j < this.layers[i]; j++) {
          var sum = 0;
          for (var k = 0; k < this.layers[i][j].length; k++) {
            sum += grad[0][k] * this.layers[i][j][k];
          }
          herror.push(sum);
        }
        error.unshift(herror);
      }
    }

    for (let i = 0; i < this.layers.length; i++) {
      var inputs = this.bpact[i];
      for (var j = 0; j < this.layers[i]; j++) {
        for (var k = 0; k < this.layers[i][j].length; k++) {
          this.layers[i][j][k] += inputs[j] * grad[i] * this.learnRate;
        }
      }
    }
  }
}

function activateFn(x) {
  return 1 / (1 + Math.exp(-x))
}

function aFnDeriv(x) {
  return x * (1 - x)
}

function dotarr(a, b) {
  return a.map((x, y) => x * b[y]).reduce((x, y) => x + y)
}

ge.start();