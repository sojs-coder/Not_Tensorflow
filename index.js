const { Network, evaluate, examineParameter } = require("./neuron");

const net = new Network({
    inputs: 1,
    outputs: 1,
    hiddenLayers: [5],
    activation: "linear",
    ignoreNaN: false
});
const inputFunction = (x) => 1.5 * (x);
const trainingOptions = {
    inputFunction,
    inputRange: [-50, 50],
    learningRate: 0.05,
    epochs: 10000,
    batchSize: 100,
    numInputs: 1000,
    verbose: false,
}
const evaluationParameters = {
    inputRange: [-1,1],
    numTrials: 100,
    numInputs: 1,
    model: net,
    inputFunction
}
const studyOptions = {
    inputRange: [-1,1],
    inputStep: 0.1,
    aspects: ["model-output-n1","expected"],
    graph: true,
    inputFunction
}
// examineParameter({
//     net,
//     baseOptions: trainingOptions,
//     parameters: ["epochs","learningRate"],
//     ranges: [[0, 250]],
//     log: true,
//     specificRanges: [,[0.5]],
//     steps: [1],
//     graph: true,
//     exportcsv: true,
//     aspects: ["model-error"], // only one aspect currently supported
//     evaluationParameters
// });
// net.reinitialize()
net.train({
    inputFunction,
    inputRange: [-1, 1],
    learningRate: 0.005,
    epochs: 1000,
    batchSize: 5,
    numInputs: 1000,
    verbose: false,
    graph: true,
    aspects: ["error-mse","weight-l1-n1-w1","bais-l1-n1"]
    // aspects are
    // weight-lx-ni-wj
    //    - x is layer number (input layer does not have weights, dont try l0)
    //    - i is node number
    //    - j is the preceding node number (the weight that connects node I to node J);
    // bias-lx-bn:
    //    - x is the layer number (same as weight)
    //    - n is the node on that layer
    // error-<denom>
    //    - <denom> can be one of ["mse"]
});
net.studyOutput(studyOptions);
// you can study the following aspects
// "model-output-nx", where x is the number of the node whose output you want
// "node-<denom>"
// - denom is "output" or "error" for the neuron specified in "neuron"
// neuron is specified "lx-ni", layer x, node i

console.log(evaluate(evaluationParameters).meanerror);
console.log(net.forwardPass([2]))
console.log(net.summary())
console.log("Done")