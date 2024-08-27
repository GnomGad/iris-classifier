const fs = require("fs");

function loadIrisData(fileName) {
    const file = fs.readFileSync(fileName, "utf-8");
    const lines = file.trim().split("\r\n");
    const headers = lines[0].split(",");

    return lines.slice(1).map((line) => {
        const values = line.split(",");
        let record = {};
        headers.forEach((val, i) => {
            record[val.trim()] = values[i].trim();
        });
        return {
            input: [
                parseFloat(record.SepalLength),
                parseFloat(record.SepalWidth),
                parseFloat(record.PetalLength),
                parseFloat(record.PetalWidth),
            ],
            output: ConverSpeciesToVector(record.Species),
        };
    });
}

function ConverSpeciesToVector(species) {
    if (species === "setosa") return [1, 0, 0];
    if (species === "versicolor") return [0, 1, 0];
    if (species === "virginica") return [0, 0, 1];
}
function createModel() {
    return {
        hw: Array.from({ length: 4 }, () => Math.random()),
        ow: Array.from({ length: 3 }, () => Math.random()),
        hb: Math.random(),
        ob: Array.from({ length: 3 }, () => Math.random()),
    };
}

function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

function sigmoidDerivative(x) {
    return x * (1 - x);
}

function forwardPropagation(inputs, model) {
    let hsum = inputs.reduce((sum, input, i) => {
        return sum + input * model.hw[i];
    }, model.hb);
    let ho = sigmoid(hsum);

    return model.ow.map((w, i) => sigmoid(ho * w + model.ob[i]));
}

function train(model, trainingData, iterations, learningRate) {
    for (let i = 0; i < iterations; i++) {
        trainingData.forEach((data) => {
            let outputs = forwardPropagation(data.input, model);
            let outputErrors = data.output.map((expected, i) => expected - outputs[i]);

            model.ow = model.ow.map((w, i) => {
                return w + outputErrors[i] * sigmoidDerivative(outputs[i]) * learningRate;
            });

            model.ob = model.ob.map((b, i) => {
                return b + outputErrors[i] * learningRate;
            });

            let hError = model.ow.reduce((sum, w, i) => {
                return sum + w * outputErrors[i];
            }, 0);

            model.hw = model.hw.map((w, i) => {
                return w + hError * sigmoidDerivative(hError) * data.input[i] * learningRate;
            });

            model.hb += hError * learningRate;
        });
    }
}

const irisData = loadIrisData("./data/iris.csv");
const model = createModel();
train(model, irisData, 100000, 0.1);

console.log("Predictions:");
irisData.forEach((data) => {
    let outputs = forwardPropagation(data.input, model);
    console.log(`Input: ${data.input}, Predicted: ${outputs.map((o) => o.toFixed(4))}, Expected: ${data.output}`);
});
