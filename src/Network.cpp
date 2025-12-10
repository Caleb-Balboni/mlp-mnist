#include "Network.h"

Network::Layer::Layer()
{
    Network net;
    cout << "Bias ";
    BiasWeight = net.random_num(-1.75, 1.75);
}

bool Network::changeinputs(vector<vector<double>> inp)
{
    inputs.clear();
    for (int i = 0; i < (int)inp.size(); i++)
        inputs.push_back(inp[i]);

    if (!inputs.empty() && inputs[0].size() >= 2 &&
        (inputs[0][0] == 1) && (inputs[0][1] == 5))
        return true;

    return false;
}

void Network::changelabels(const vector<uint8_t>& lab)
{
    labels = lab;
}

double Network::random_num(double start, double end)
{
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<double> distribution(start, end);
    double random = distribution(generator);
    return random;
}

double Network::calcderivitive(double output)
{
    return output * (1.0 - output);
}

double Network::sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

Network::Neuron Network::CreateNeuron(int WA)
{
    Neuron TempN;
    for (int i = 0; i < WA; i++)
        TempN.Weights.push_back(random_num(-1.75, 1.75));
    TempN.output = 0;
    TempN.error = 0;
    return TempN;
}

void Network::CreateLayer(int NeuronN)
{
    if (NeuronN <= 0)
        throw invalid_argument("CreateLayer: NeuronN must be > 0");

    if (inputs.empty() || inputs[0].empty())
        throw runtime_error("CreateLayer: inputs must be initialized first");

    Layer templayer;
    for (int i = 0; i < NeuronN; i++)
    {
        if (Layers.size() == 0)
            templayer.Neurons.push_back(CreateNeuron((int)inputs[0].size()));
        else
            templayer.Neurons.push_back(CreateNeuron((int)Layers[Layers.size() - 1].Neurons.size()));
    }
    Layers.push_back(templayer);
}

static vector<double> MakeOneHot(uint8_t label)
{
    vector<double> t(10, 0.0);
    if (label < 10) t[label] = 1.0;
    return t;
}

void Network::Backpropegate(int z)
{
    for (int i = (int)Layers.size() - 1; i >= 0; i--)
    {
    if (i == Layers.size() - 1)
{
    // If we have 10 outputs, treat this as classification for your 10 toy inputs.
    // Target is one-hot: neuron z is correct.
    if (Layers[i].Neurons.size() == 10)
    {
        for (int n = 0; n < 10; n++)
        {
            double target = (n == z) ? 1.0 : 0.0;

            Layers[i].Neurons[n].error =
                (target - Layers[i].Neurons[n].output) *
                calcderivitive(Layers[i].Neurons[n].output);

            for (int x = 0; x < Layers[i].Neurons[n].Weights.size(); x++)
            {
                Layers[i].Neurons[n].Weights[x] +=
                    Layers[i].Neurons[n].error *
                    Layers[i - 1].Neurons[x].output;
            }
        }

        // Keep your layer-level shared bias style
        Layers[i].BiasWeight += Layers[i].Bias * Layers[i].Neurons[z].error;
    }
    else
    {
        // Legacy behavior for 1-output networks
        Layers[i].Neurons[0].error =
            ((outputs[z] - Layers[i].Neurons[0].output) *
             calcderivitive(Layers[i].Neurons[0].output));

        for (int x = 0; x < Layers[i].Neurons[0].Weights.size(); x++)
            Layers[i].Neurons[0].Weights[x] +=
                Layers[i].Neurons[0].error * Layers[i - 1].Neurons[x].output;

        Layers[i].BiasWeight += Layers[i].Bias * Layers[i].Neurons[0].error;
    }
}

    }
}

void Network::ReturnWeights()
{
    for (int i = 0; i < (int)Layers.size(); i++)
    {
        cout << "Layer: " << i << endl;
        for (int x = 0; x < (int)Layers[i].Neurons.size(); x++)
        {
            for (int y = 0; y < (int)Layers[i].Neurons[x].Weights.size(); y++)
            {
                if (y == 0) cout << Layers[i].Neurons[x].Weights[y] << "x + ";
                if (y == 1) cout << Layers[i].Neurons[x].Weights[y] << "y + ";
            }
            cout << Layers[i].BiasWeight << " = 0\n";
        }
    }
}

void Network::TrainNetwork(int epochs, bool dbg)
{
    if (epochs <= 0)
        throw invalid_argument("TrainNetwork: epochs must be > 0");

    if (Layers.empty())
        throw runtime_error("TrainNetwork: must CreateLayer first");

    for (int z = 0; z < epochs; z++)
    {
        for (int c = 0; c < (int)inputs.size(); c++)
        {
            vector<double> Inputs = inputs[c];

            for (int i = 0; i < (int)Layers.size(); i++)
            {
                for (int x = 0; x < (int)Layers[i].Neurons.size(); x++)
                {
                    double tempoutput = 0;
                    for (int y = 0; y < (int)Inputs.size(); y++)
                        tempoutput += Inputs[y] * Layers[i].Neurons[x].Weights[y];

                    Layers[i].Neurons[x].output =
                        sigmoid(tempoutput + Layers[i].Bias * Layers[i].BiasWeight);
                }

                Inputs.clear();
                for (int x = 0; x < (int)Layers[i].Neurons.size(); x++)
                    Inputs.push_back(Layers[i].Neurons[x].output);
            }
            Backpropegate(c);
        }
    }
}

void Network::outputnetwork(int epochs, bool dbg)
{
    for (int z = 0; z < epochs; z++)
    {
        for (int c = 0; c < (int)inputs.size(); c++)
        {
            vector<double> Inputs = inputs[c];
            for (int i = 0; i < (int)Layers.size(); i++)
            {
                for (int x = 0; x < (int)Layers[i].Neurons.size(); x++)
                {
                    double tempoutput = 0;
                    for (int y = 0; y < (int)Inputs.size(); y++)
                        tempoutput += Inputs[y] * Layers[i].Neurons[x].Weights[y];

                    Layers[i].Neurons[x].output =
                        sigmoid(tempoutput + Layers[i].Bias * Layers[i].BiasWeight);
                }

                Inputs.clear();
                for (int x = 0; x < (int)Layers[i].Neurons.size(); x++)
                    Inputs.push_back(Layers[i].Neurons[x].output);
            }
            if (dbg) {
            for (int i = 0; i < Inputs.size(); i++) {
              cout << "output: " << i << " " << Inputs[i] << endl; 
            }
          }
        }
    }
}
