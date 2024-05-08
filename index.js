const { Network, evaluate, examineParameter } = require("./neuron");

const net = new Network({
    inputs: 1,
    outputs: 1,
    activation: "linear",
    ignoreNaN: true
});
const inputFunction = (x) => x;
const trainingOptions = {
    inputFunction,
    inputRange: [-1,1],
    learningRate: 0.05,
    epochs: 500,
    verbose: false,
    graph: false,
}
const evaluationParameters = {
    inputRange: [-50,50],
    numTrials: 100,
    numInputs: 1,
    model: net,
    inputFunction
}
const studyOptions = {
    inputRange: [-50,50],
    inputStep: 0.01,
    aspects: ["model-output-n1","expected"],
    graph: true,
    inputFunction
}
examineParameter({
    net,
    baseOptions: trainingOptions,
    parameters: ["epochs","learningRate"],
    ranges: [[0, 250]],
    log: true,
    specificRanges: [,[0.5,1,2]],
    steps: [1],
    graph: true,
    exportcsv: true,
    aspects: ["model-error"], // only one aspect currently supported
    evaluationParameters
});
// net.reinitialize()
// net.train({
//     inputFunction,
//     inputRange: [-3, 3],
//     learningRate: 0,
//     epochs: 1500,
//     verbose: false,
//     graph: true,
//     aspects: ["error-mse"]
//     // aspects are
//     // weight-lx-ni-wj
//     //    - x is layer number (input layer does not have weights, dont try l0)
//     //    - i is node number
//     //    - j is the preceding node number (the weight that connects node I to node J);
//     // bias-lx-bn:
//     //    - x is the layer number (same as weight)
//     //    - n is the node on that layer
//     // error-<denom>
//     //    - <denom> can be one of ["mse"]
// });
net.studyOutput(studyOptions);
// you can study the following aspects
// "model-output-nx", where x is the number of the node whose output you want
// "node-<denom>"
// - denom is "output" or "error" for the neuron specified in "neuron"
// neuron is specified "lx-ni", layer x, node i

console.log(evaluate(evaluationParameters).meanerror);
console.log(net.summary())
console.log("Done")