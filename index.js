const { Network, evaluate, examineParameter } = require("./neuron");

const net = new Network({
    inputs: 1,
    outputs: 1,
    activation: "linear",
});
const inputFunction = (x) => x;
examineParameter({
    net,
    baseOptions: {
        inputFunction,
        inputRange: [-1,1],
        learningRate: 0.01,
        epochs: 500,
        verbose: false,
        graph: false
    },
    parameter: "learningRate",
    range: [0, 0.5],
    step: 0.05,
    graph: true,
    aspects: ["model-error"],
    evaluationParameters: {
        inputRange: [-50,50],
        numTrials: 100,
        numInputs: 1
    }
})
// net.train({
//     inputFunction,
//     inputRange: [-1, 1],
//     learningRate: 0.5,
//     epochs: 150000,
//     verbose: false,
//     graph: true,
//     aspects: ["weight-l1-n1-w1","bias-l1-b1","error-mse"]
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

// net.studyOutput({
//     inputRange: [-50,50],
//     inputStep: 1,
//     aspects: ["model-output-n1","expected"],
//     neuron: "l1-n1",
//     graph: true,
//     inputFunction
// });
// you can study the following aspects
// "model-output-nx", where x is the number of the node whose output you want
// "node-<denom>"
// - denom is "output" or "error" for the neuron specified in "neuron"
// neuron is specified "lx-ni", layer x, node i

console.log(evaluate({
    model: net,
    numInputs: 1,
    inputRange: [-50,50],
    inputFunction,
    numTrials: 100
}).meanerror);
console.log(net.summary())