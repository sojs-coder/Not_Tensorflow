const { ChartJSNodeCanvas } = require('chartjs-node-canvas');
const fs = require("fs");
var bigDecimal = require('js-big-decimal').default;
class Network {
    constructor({ inputs, outputs, hiddenLayers = [], activation, ignoreNaN=false }) {
        this.activationFunction = activation;
        this.numInputs = inputs;
        this.numOutputs = outputs;
        this.network = [];
        this.ignoreNaN = ignoreNaN
        this.hiddenSpec = hiddenLayers;
        // first layer is input layer
        this.network.push(new Layer(inputs));
        for (var i = 0; i < hiddenLayers.length; i++) {
            this.network.push(
                new Layer(hiddenLayers[i], this.network[this.network.length - 1]),
            );
        }
        // output layer
        this.network.push(
            new Layer(outputs, this.network[this.network.length - 1]),
        );
    }
    reinitialize() {
        this.network = [];
        this.network.push(new Layer(this.numInputs));
        for (var i = 0; i < this.hiddenSpec.length; i++) {
            this.network.push(
                new Layer(this.hiddenSpec[i], this.network[this.network.length - 1]),
            );
        }
        // output layer
        this.network.push(
            new Layer(this.numOutputs, this.network[this.network.length - 1]),
        );
    }
    summary() {
        return `Model Summary:\n - ${this.numInputs} inputs, ${this.network[this.network.length - 1].numNodes} outputs\n - Hidden layers: [${this.hiddenSpec.join(", ")}]`
    }
    activation(x) {
        switch (this.activationFunction) {
            case "relu":
                return relu(x);
            case "sigmoid":
                return sigmoid(x);
            case "linear":
                return x;
        }
    }
    _activation(x) {
        switch (this.activationFunction) {
            case "relu":
                return _relu(x);
            case "sigmoid":
                return _sigmoid(x);
            case "linear":
                return 1;
        }
    }
    mse(actual, expected) {
        if (actual.length != expected.length) throw new Error("@Network.mse: actual and expected sets must be the same length");

        var sum = 0;
        for (var i = 0; i < actual.length; i++) {
            sum += Math.pow(actual[i] - expected[i], 2);
        }
        return sum / actual.length;
    }
    // format for inputRane is [min,max].
    // for multi-dimensional inputs
    // [[node1_min,node1_max], [node2_min,node2_max]]
    // etc
    studyOutput({ inputRange, inputStep, neuron, aspects, graph, inputFunction }) {
        var dataPoints = {
            labels: []
        }
        for (const a of aspects) {
            dataPoints[a] = []
        }
        if (inputRange[0].length == undefined) {

            for (var i = inputRange[0]; i < inputRange[1]; i += inputStep) {
                var output = this.forwardPass([i]);
                dataPoints.labels.push(i);
                for (const aspect of aspects) {
                    var aParts = aspect.split("-");
                    var aspectClass = aParts[0];
                    var aspectDenom = aParts[1];

                    switch (aspectClass) {
                        case "model":
                            if (aspectDenom == "output") {
                                var onNode = parseInt(aParts[2].substring(1)) - 1;
                                dataPoints[aspect].push(output[onNode]);
                                if (dataPoints[aspect] !== undefined) {
                                    var actual = inputFunction([i]);
                                    if (typeof actual == "number") {
                                        dataPoints.expected.push(actual);
                                    } else if (actual.length !== undefined) {
                                        dataPoints.expected.push(actual[onNode]);
                                    }
                                }
                            }
                            break;
                        case "node":
                            var parts = neuron.split("-");
                            var layerP = parts[0].substring(1);
                            var nodeP = parts[1].substring(1);
                    
                            var layerNum = parseInt(layerP);
                            var nodeNum = parseInt(nodeP) - 1;
                    
                            var layer = this.network[layerNum];
                            var node = layer.neurons[nodeNum];
                    
                            if (aspectDenom == "error") {
                                dataPoints[aspect].push(node.error);
                            } else if (aspectDenom = "output") {
                                dataPoints[aspect].push(node.output);
                            }
                            break;
                    }
                }
            }
        } else {
            throw new Error("You can only currently study single-input models. ")
        }
        if (graph) {
            graphData({
                dataPoints,
                compare: true,
                filename: "study",
                xAxisLabel: "Input",
                yAxisLabel: "Output"
            })
        }
        return dataPoints
    }
    // input function requires a range defined. Input function accepts only 1 parameter, which mean your model must only accept one input.
    train({
        inputSet,
        inputFunction,
        inputRange = [-1, 1],
        outSet,
        learningRate,
        epochs,
        verbose = true,
        graph = false,
        aspects = ["error"],
        compare = true,
    }) {
        var gValues = {
            labels: [],
        };
        if (graph) {
            if (!aspects) {
                throw new Error("Provide aspects to graph");
            }
            for (var i = 0; i < aspects.length; i++) {
                gValues[aspects[i]] = [];
            }
        }
        if (verbose)
            console.log(
                `Training, ${learningRate} learning rate, ${epochs} epochs`,
            );
        for (var i = 0; i < epochs; i++) {
            var inputs = [];
            var expected = [];
            if (!inputFunction) {
                var index = Math.floor(Math.random() * inputSet.length);
                inputs = inputSet[index];
                expected = outSet[index];
            } else {
                var inputs = [];
                for (var numIx = 0; numIx < this.numInputs; numIx++) {
                    inputs.push(randRange(inputRange[0], inputRange[1]));
                }
                var realOutput = inputFunction(...inputs);
                if (typeof realOutput == "number") {
                    expected = [realOutput]
                } else if (realOutput.length !== undefined) {
                    expected = realOutput;
                }
            }

            var output = this.forwardPass(inputs);
            if (verbose)
                console.log(`${i}/${epochs} ===== mse = ${this.mse(output, expected)}`);
            if (graph) {
                for (const aspect of aspects) {
                    var parts = aspect.split("-");
                    var denom = parts[0];
                    switch (denom) {
                        case "error":
                            var errorCase = parts[1];
                            if (errorCase == "mse") {
                                gValues[aspect].push(this.mse(output, expected));
                            } else {
                                gValues[aspect].push(this.mse(output, expected));
                            }
                            break;
                        case "weight":
                            var layer = parts[1];
                            var node = parts[2];
                            var weight = parts[3];

                            var layerNum = parseInt(layer.substring(1));
                            var nodeNum = parseInt(node.substring(1));
                            var weightNum = parseInt(weight.substring(1));

                            // already offset by 1 b/c the input is also treated as a layer
                            var layer = this.network[layerNum];
                            var node = layer.neurons[nodeNum - 1]
                            var weight = node.weights[weightNum - 1];

                            gValues[aspect].push(weight);
                            break;
                        case "bias":
                            var layer = parts[1];
                            var bias = parts[2];

                            var layerNum = parseInt(layer.substring(1));
                            var biasNum = parseInt(bias.substring(1));

                            var layer = this.network[layerNum];
                            var node = layer.neurons[biasNum - 1];
                            var bias = node.bias;
                            gValues[aspect].push(bias);
                            break;
                        default:
                            var errorCase = parts[1];
                            if (errorCase == "mse") {
                                gValues[aspect].push(this.mse(output, expected));
                            } else {
                                gValues[aspect].push(this.mse(output, expected));
                            }
                            break;

                    }
                }
                gValues.labels.push(i);
            }
            this.backwardPass(inputs, expected);
            this.modifyWeights(learningRate);
        }
        if (graph) {
            graphData({
                dataPoints: gValues,
                compare,
                filename: "training"
            })
        }
    }
    modifyWeights(learningRate) {
        // Iterate over each layer (excluding the input layer)
        for (var i = 1; i < this.network.length; i++) {
            var layer = this.network[i];
            var prevLayerOutputNeurons = this.network[i - 1].neurons;
            // Iterate over each neuron in the layer
            for (var j = 0; j < layer.neurons.length; j++) {
                var neuron = layer.neurons[j];
                // Get the error for the current neuron
                var error = neuron.error;
                // Iterate over each weight connected to the current neuron
                for (var k = 0; k < prevLayerOutputNeurons.length; k++) {
                    neuron.weights[k] -=
                        learningRate * error * prevLayerOutputNeurons[k].output;
                    if(isNaN(neuron.weights[k]) && !this.ignoreNaN) throw new Error(`NaN detected, learning rate ${learningRate}, error: ${error}, prevLayerOut: ${prevLayerOutputNeurons[k].output}`)
                }
                // Update the bias for the current neuron
                neuron.bias -= learningRate * error;
                if(isNaN(neuron.bias) && !this.ignoreNaN) throw new Error(`NaN detected, learning rate: ${learningRate}, error: ${error}`)
            }
        }
    }
    backwardPass(inputs, expected) {
        // go backwards from the output layer to the input
        for (var i = this.network.length - 1; i >= 0; i--) {
            var layer = this.network[i];
            // if the layer is the output layer, calculate error from expected
            if (i == this.network.length - 1) {
                // go through each node and find the individual error
                for (var j = 0; j < layer.neurons.length; j++) {
                    var neuron = layer.neurons[j];
                    var output = neuron.output;
                    neuron.error = (output - expected[j]) * this._activation(output);
                    if(isNaN(neuron.error) && !this.ignoreNaN) throw new Error(`NaN detected, calculating output neuron error\n Output: ${output}. Expected: ${expected[j]}. _activation: ${this._activation(output)}`)
                }
            } else {
                // calculate error for the hidden layers
                for (var j = 0; j < layer.neurons.length; j++) {
                    var errorSum = 0;
                    // accumulate error from the forward nodes, corresponding with their weights
                    // (if the weight is small, modifications will not change much)
                    var forwardLayer = this.network[i + 1];
                    var neuron = layer.neurons[j];
                    for (var k = 0; k < forwardLayer.neurons.length; k++) {
                        var fneuron = forwardLayer.neurons[k];
                        var weightToCurrent = fneuron.weights[j];
                        var forwardError = fneuron.error;
                        errorSum += weightToCurrent * forwardError;
                    }
                    // add the error for the current node (j)
                    neuron.error = errorSum * this._activation(neuron.output);
                }
            }
        }
    }
    forwardPass(inputs, verbose = false) {
        for (var i = 0; i < this.network.length; i++) {
            var layer = this.network[i];
            if (i == 0) {
                // this is this input layer
                // the input neurons already have their ouput configured as input
                // jk they dont
                layer.neurons.forEach((n, ni) => {
                    n.output = inputs[ni];
                    if(isNaN(n.output)) throw new Error(`NaN detected, not a number passed: input: ${inputs[ni]}`)
                });
                if (verbose) console.log(`Feeding: [${inputs.join(",")}]`);
                continue;
            }
            for (var b = 0; b < layer.neurons.length; b++) {
                // for each node
                var sum = 0;
                // node is..
                var node = layer.neurons[b];
                // get the preceding inputs
                var precedingInputNeurons = this.network[i - 1].neurons;
                // mult each input with its corresponding weight and add to sum
                for (var inp = 0; inp < precedingInputNeurons.length; inp++) {
                    if (verbose)
                        console.log(
                            "Prev node: " +precedingInputNeurons[inp].id +
                            ": " +
                            precedingInputNeurons[inp].output,
                        );
                    var prevoutput = precedingInputNeurons[inp].output;
                    var weight = node.weights[inp];
                    if(verbose) console.log("Weight is: ",weight)
                    sum += weight * prevoutput;
                }
                // add the bias
                sum += node.bias;
                // set as the output for the corresponding node `b`
                if (verbose)
                    console.log(
                        `Node ${node.id}\n\t(bias): ${node.bias}\n\t(sum-bias), sum, a(sum)\n\t1. ${sum - node.bias}\n\t2. ${sum}\n\t3. ${this.activation(sum)}`,
                    );
                node.output = this.activation(sum);
            }
        }
        if(verbose) console.log("Output:"+this.network[this.network.length - 1].getOutput())
        return this.network[this.network.length - 1].getOutput();
    }
}

class Layer {
    constructor(n_nodes, precedingLayer) {
        this.layerNum = precedingLayer ? precedingLayer.layerNum + 1 : 0;
        this.id = "layer" + this.layerNum;
        this.numNodes = n_nodes;
        this.neurons = [];
        if (precedingLayer) {
            for (var i = 0; i < n_nodes; i++) {
                var neuron = new Neuron();
                neuron.id = "l-" + this.layerNum + "-n-" + i;
                neuron.bias = Math.random() * 2 - 1;
                for (var j = 0; j < precedingLayer.numNodes; j++) {
                    neuron.weights[j] = Math.random() * 2 - 1;
                }
                this.neurons.push(neuron);
            }
        } else {
            this.isInput = true;
            for (var i = 0; i < n_nodes; i++) {
                this.neurons.push(new Input());
            }
        }
    }
    getOutput() {
        return this.neurons.map((n) => n.output);
    }
}
class Node {
    constructor() {
        this.output = 0;
        this.id = Math.random().toString(16).substring(2);
    }
}
class Input extends Node {
    constructor() {
        super();
        this.id = "input-" + this.id;
    }
}
class Neuron extends Node {
    // exists purely to be able to.... idk.... just nice way to format data
    // if this were ts this'd be a type or an interfact idk really its not ts so I havent throught that far
    constructor() {
        super();
        this.id = "neuron-" + this.id;
        this.output;
        this.bias;
        this.weights = []; // (from previous layer)
        this.error;
    }
}
function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}
function _sigmoid(x) {
    return x * (1 - x);
}
function relu(x) {
    return Math.max(0, x);
}
function _relu(x) {
    if (x > 0) {
        return 1;
    } else {
        return 0;
    }
}

function linear(x) {
    return x;
}
function _linear(x) {
    return 1;
}
function randRange(min, max) {
    return Math.random() * (max - min) + min;
}
function exportToCSV({
    dataPoints,
    labelAlt,
    filename
}) {
    dataPoints[labelAlt] = dataPoints.labels;
    var header = [labelAlt, ...Object.keys(dataPoints).filter((col) => col !== "labels" && col !== labelAlt)]
    var finalString = `${header.join(",")}`;
    for (var i = 0; i < dataPoints.labels.length; i++) {
        var record = [];
        for (const col of Object.keys(dataPoints)) {
            if (col == "labels") continue;
            record[header.indexOf(col)] = (dataPoints[col][i]);
        }
        finalString += `\n${record.join(",")}`;
    }
    fs.writeFile("graphs/" + filename + ".csv", finalString, "utf-8", (err) => {
        if (err) throw err;
    })
}
function graphData({
    dataPoints,
    compare,
    filename,
    xAxisLabel = "Epochs",
    yAxisLabel,
    logY = false
}) {
    if (!fs.existsSync("graphs")) {
        fs.mkdirSync("graphs")
    }
    const aspects = Object.keys(dataPoints);
    aspects.splice(aspects.indexOf('labels'), 1);
    var colors = ["#36a2eb", "#ff6384", "#4bc0c0", "#ff9f40", "#9966ff", "#ffcd56", "#c9cbcf"]
    const chartJSNodeCanvas = new ChartJSNodeCanvas({
        width: 800,
        height: 600,
        backgroundColour: "white"
    });
    var configuration = {};
    if (compare) {
        configuration = {
            type: "line",
            data: {
                labels: dataPoints.labels,
                datasets: [],
            },
            options: {
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: xAxisLabel
                        },
                    },
                    y: {
                        title: {
                            display: (yAxisLabel) ? true : ((aspects.length == 1) ? true : false),
                            text: (yAxisLabel) ? yAxisLabel : ((aspects.length == 1) ? aspects[0] : "")
                        },
                        type: (logY) ? "logarithmic" : "linear"
                    }
                }
            }
        };
        for (const aspect of aspects) {
            if (compare) {
                configuration.data.datasets.push({
                    data: dataPoints[aspect],
                    label: aspect,
                    borderColor: colors[aspects.indexOf(aspect) % colors.length]
                });
            } else {
                var configuration = {
                    type: "line",
                    data: {
                        labels: dataPoints.labels,
                        datasets: [
                            {
                                label: aspect,
                                data: dataPoints[aspect],
                                borderColor: colors[aspects.indexOf(aspect) % colors.length]
                            },
                        ],
                    },
                };
                chartJSNodeCanvas.renderToDataURL(configuration).then((dataUrl) => {
                    const base64Image = dataUrl;
                    var base64Data = base64Image.replace(
                        /^data:image\/png;base64,/,
                        "",
                    );
                    fs.writeFile("graphs/" + filename + "-" + aspect + ".png", base64Data, "base64", function (err) {
                        if (err) {
                            console.log(err);
                        }
                    });
                });
            }
        }
        if (compare) {
            chartJSNodeCanvas.renderToDataURL(configuration).then((dataUrl) => {
                const base64Image = dataUrl;
                var base64Data = base64Image.replace(
                    /^data:image\/png;base64,/,
                    "",
                );
                fs.writeFile("graphs/" + filename + ".png", base64Data, "base64", function (err) {
                    if (err) {
                        console.log(err);
                    }
                });
            });
        }
    }
}
function evaluate({ model, numInputs, inputRange, inputFunction, numTrials }) {
    var errors = [];
    for (var i = 0; i < numTrials; i++) {
        var inputs = new Array(numInputs).fill(0).map(() => {
            return randRange(inputRange[0], inputRange[1]);
        });
        var outputs = model.forwardPass(inputs);
        var expected = [];
        var realOutput = inputFunction(...inputs);
        if (typeof realOutput == "number") {
            expected = [realOutput]
        } else if (realOutput.length !== undefined) {
            expected = realOutput;
        }
        errors.push(model.mse(outputs, expected));
    }
    var meanerror = errors.reduce((acc, val) => {
        return acc + val
    }, 0) / errors.length;
    return { meanerror, errors }
}
function examineParameter({ net, baseOptions, log=true, parameters, ranges, specificRanges = [], steps, aspects, evaluationParameters, exportcsv = false, graph = false }) {
    var dataPoints = {
        "labels": []
    }
    var p = 0;
    const step0 = steps[p];
    const range0 = ranges[p];
    const specificRange0 = specificRanges[0];
    const parameter0 = parameters[p];
    var stepNum0 = new bigDecimal(step0);
    // this is the first paremeter, will be on X axis
    var i = 0;
    if (!specificRange0) {
        i = new bigDecimal(range0[0]);
    }
    while (
        (!specificRange0) ? parseFloat(i.getValue()) <= range0[1] :
            (i < specificRange0.length)
    ) {
        const currentXValue = (!specificRange0) ? parseFloat(i.getValue()) : specificRange0[i];
        var options = { ...baseOptions };
        options[parameter0] = currentXValue;
        dataPoints.labels.push(currentXValue);
        if (parameters.length == 1) {
            var specificRangeP = specificRange0
            net.reinitialize();
            options[parameters[p]] = (!specificRangeP) ? parseFloat(i.getValue()) : specificRangeP[i];
            net.train(options);
            var { meanerror: mse } = evaluate({
                model: net,
                numInputs: evaluationParameters.numInputs,
                inputRange: evaluationParameters.inputRange,
                inputFunction: baseOptions.inputFunction,
                numTrials: evaluationParameters.numTrials
            });
            for (var a = 0; a < aspects.length; a++) {
                var labelString = `${aspects[a]}`;
                if (dataPoints[labelString] == undefined) dataPoints[labelString] = [];
                var dataPoint = (aspects[a] == "model-error") ? mse : 1
                dataPoints[labelString].push(dataPoint);
            }
        }
        // then, we find all of the values for the other parameters at this point on the X axis and add them as a series
        for (p = 1; p < parameters.length; p++) {
            const range = ranges[p];
            const step = steps[p];
            const stepNum = new bigDecimal(step);
            const specificRangeP = specificRanges[p];
            var np = (specificRangeP) ? 0 : new bigDecimal(range[0])
            while (
                (specificRangeP) ? np < specificRangeP.length :
                    parseFloat(np.getValue()) <= range[1]
            ) {
                net.reinitialize();
                options[parameters[p]] = (!specificRangeP) ? parseFloat(np.getValue()) : specificRangeP[np];
                net.train(options);
                var { meanerror: mse } = evaluate({
                    model: net,
                    numInputs: evaluationParameters.numInputs,
                    inputRange: evaluationParameters.inputRange,
                    inputFunction: baseOptions.inputFunction,
                    numTrials: evaluationParameters.numTrials
                });
                for (var a = 0; a < aspects.length; a++) {
                    var labelString = `${aspects[a]}_${parameters[p]}-${(!specificRangeP) ? parseFloat(np.getValue()) : specificRangeP[np]}`;
                    if (dataPoints[labelString] == undefined) dataPoints[labelString] = [];
                    var dataPoint = (aspects[a] == "model-error") ? mse : 1
                    dataPoints[labelString].push(dataPoint);
                }
                if (!specificRangeP) {
                    np = np.add(stepNum);
                } else {
                    np++;
                }
            }
        }
        if (!specificRange0) {
            i = i.add(stepNum0);
        } else {
            i++
        }
    }
    if (graph) {
        graphData({
            dataPoints,
            compare: true,
            filename: "examineParameter-" + parameters.join("_"),
            logY: log,
            xAxisLabel: parameters[0],
            yAxisLabel: aspects[0]

        })
    }
    if (exportcsv) {
        exportToCSV({
            dataPoints,
            filename: "examineParameter-" + parameters.join("_"),
            labelAlt: parameters[0]
        })
    }
    return dataPoints;
}
module.exports = {
    Network,
    Layer,
    Neuron,

    evaluate,
    examineParameter,

    sigmoid,
    _sigmoid,
    relu,
    _relu,
    linear,
    _linear,

    randRange
};
