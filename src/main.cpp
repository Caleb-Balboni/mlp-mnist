#include "Network.h"
#include "MNIST.h"

static int ArgInt(int argc, char** argv, const string& key, int def)
{
    for (int i = 1; i < argc - 1; i++)
        if (key == argv[i]) return stoi(argv[i + 1]);
    return def;
}

static string ArgStr(int argc, char** argv, const string& key, const string& def = "")
{
    for (int i = 1; i < argc - 1; i++)
        if (key == argv[i]) return argv[i + 1];
    return def;
}

int main(int argc, char** argv)
{
    try
    {
        string trainImages = ArgStr(argc, argv, "--train-images");
        string trainLabels = ArgStr(argc, argv, "--train-labels");

        int epochs = ArgInt(argc, argv, "--epochs", 1);
        double lr = 0.05;
        {
            string lrStr = ArgStr(argc, argv, "--lr");
            if (!lrStr.empty()) lr = stod(lrStr);
        }

        Network testnet;
        testnet.learningRate = lr;

        if (!trainImages.empty() && !trainLabels.empty())
        {
            MNISTData data = MNIST::Load(trainImages, trainLabels);
            testnet.changeinputs(data.images);
            testnet.changelabels(data.labels);

            testnet.CreateLayer(128);
            testnet.CreateLayer(64);
            testnet.CreateLayer(10);

            testnet.TrainNetwork(epochs, false);

            vector<vector<double>> one =
            {
                data.images[0]
            };
            testnet.changeinputs(one);
            testnet.outputnetwork(1, true);

            cout << "First label was: " << int(data.labels[0]) << endl;
        }
        else
        {
            // Legacy toy-mode (your original behavior)
            testnet.CreateLayer(4);
            testnet.CreateLayer(3);
            testnet.CreateLayer(1);
            testnet.TrainNetwork(100000,false);
            testnet.TrainNetwork(10, true);

            vector<vector<double>> inp =
            {
                {1,5}
            };

            cout << testnet.changeinputs(inp);
            testnet.outputnetwork(5, true);
            testnet.ReturnWeights();
        }
    }
    catch (const invalid_argument& e)
    {
        cerr << "Invalid argument: " << e.what() << endl;
        return 1;
    }
    catch (const runtime_error& e)
    {
        cerr << "Runtime error: " << e.what() << endl;
        return 2;
    }
    catch (const exception& e)
    {
        cerr << "Error: " << e.what() << endl;
        return 3;
    }

    return 0;
}
