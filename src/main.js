var FPS = 30;
var nn;
var td = [
  { x: [1, 0], y: [0, 0, 1] },
  { x: [0, 1], y: [0, 1, 1] },
  { x: [1, 1], y: [1, 0, 0] },
];

function start() {
  nn = new NeuralNetwork([2, 2, 2, 3]);
}

function loop(dt) {
  __.text(
    td.map(d => nn.error(d).map(x => x * x).reduce((x, y) => x + y)).reduce((x, y) => x + y),
    100, 60, 'white', 30
  );
  td.map((d, i) =>
    __.text(
      d.x.join(', ') + ': ' + nn.run(d.x).map(x=>Math.floor(x*10000)/10000).join(', ') + ' / ' + d.y.join(', '),
      100, 100 + i * 40, 'white', 30
    )
  );
  td.map(d => nn.train(d));
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
    var input = structuredClone(fninput);
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

    for (var i = this.layers.length - 1; i >= 0; i--) {
      grad.unshift(error[0].map((e, j) => e * (this.bpact[i + 1][j])));
      if (i > 0) {
        var herror = [];
        for (var j = 0; j < this.layers[i].length; j++) {
          var sum = 0;
          for (var k = 0; k < this.layers[i][j].length; k++) {
            sum += grad[0][k] * this.layers[i][j][k];
          }
          herror.push(sum);
        }
        error.unshift(herror);
      }
    }

    for (var i = 0; i < this.layers.length; i++) {
      var inputs = this.bpact[i];
      for (var j = 0; j < this.layers[i].length; j++) {
        for (var k = 0; k < this.layers[i][j].length; k++) {
          this.layers[i][j][k] += inputs[j] * grad[i][k] * this.learnRate;
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
  return a.map((x, y) => x * b[y]).reduce((x, y) => (x||0) + (y||0))
}

ge.start();