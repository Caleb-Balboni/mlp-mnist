#pragma once

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <cstdint>
#include <stdexcept>

using namespace std;

class Network
{
public:
    // Your original toy default inputs
    vector<vector<double>> inputs =
    {
        {1,0},
        {1,1},
        {1,2},
        {1,3},
        {1,4},
        {1,5},
        {1,6},
        {1,7},
        {1,8},
        {1,9}
    };

    // Your original toy outputs (kept)
    double outputs[10]{ 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1 };

    // Added: MNIST-style labels aligned with inputs
    vector<uint8_t> labels;

    // Added: learning rate
    double learningRate = 0.05;

    bool changeinputs(vector<vector<double>> inp);

    // Added: set labels (does not rename anything you already had)
    void changelabels(const vector<uint8_t>& lab);

    double random_num(double start, double end);
    double calcderivitive(double output);
    double sigmoid(double x);

    struct Neuron
    {
        vector <double> Weights;
        double output = 0;
        double error = 0;
    };

    struct Layer
    {
        vector <Neuron> Neurons;
        double Bias = 1;
        double BiasWeight;
        Layer();
    };

    Neuron CreateNeuron(int WA);
    vector <Layer> Layers;

    void CreateLayer(int NeuronN);

    // Updated: Backpropegate now supports multi-output final layer
    void Backpropegate(int z);

    void ReturnWeights();

    // Updated: TrainNetwork uses labels if present (MNIST-mode)
    void TrainNetwork(int epochs, bool dbg);

    void outputnetwork(int epochs, bool dbg);
};
