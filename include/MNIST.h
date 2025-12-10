#pragma once
#include <vector>
#include <string>
#include <cstdint>

struct MNISTData
{
    std::vector<std::vector<double>> images; // each is size 784
    std::vector<uint8_t> labels;             // 0..9
};

class MNIST
{
public:
    static MNISTData Load(const std::string& imagesPath, const std::string& labelsPath);

private:
    static uint32_t ReadBE32(std::ifstream& in);
};

