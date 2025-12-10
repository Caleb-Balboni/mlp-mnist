#include "MNIST.h"
#include <fstream>
#include <stdexcept>

uint32_t MNIST::ReadBE32(std::ifstream& in)
{
    uint8_t b[4];
    in.read(reinterpret_cast<char*>(b), 4);
    if (!in) throw std::runtime_error("Failed to read BE32");
    return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) | (uint32_t(b[2]) << 8) | uint32_t(b[3]);
}

MNISTData MNIST::Load(const std::string& imagesPath, const std::string& labelsPath)
{
    std::ifstream img(imagesPath, std::ios::binary);
    std::ifstream lab(labelsPath, std::ios::binary);

    if (!img) throw std::runtime_error("Could not open images file");
    if (!lab) throw std::runtime_error("Could not open labels file");

    uint32_t imgMagic = ReadBE32(img);
    uint32_t imgCount = ReadBE32(img);
    uint32_t rows = ReadBE32(img);
    uint32_t cols = ReadBE32(img);

    uint32_t labMagic = ReadBE32(lab);
    uint32_t labCount = ReadBE32(lab);

    if (imgMagic != 2051) throw std::runtime_error("Invalid MNIST image magic");
    if (labMagic != 2049) throw std::runtime_error("Invalid MNIST label magic");
    if (imgCount != labCount) throw std::runtime_error("Image/label count mismatch");
    if (rows != 28 || cols != 28) throw std::runtime_error("Expected 28x28 images");

    MNISTData data;
    data.images.reserve(imgCount);
    data.labels.reserve(labCount);

    for (uint32_t i = 0; i < imgCount; ++i)
    {
        uint8_t label;
        lab.read(reinterpret_cast<char*>(&label), 1);
        if (!lab) throw std::runtime_error("Failed reading label");
        data.labels.push_back(label);

        std::vector<double> vec(rows * cols);
        for (uint32_t p = 0; p < rows * cols; ++p)
        {
            uint8_t px;
            img.read(reinterpret_cast<char*>(&px), 1);
            if (!img) throw std::runtime_error("Failed reading pixel");
            vec[p] = double(px) / 255.0;
        }

        data.images.push_back(std::move(vec));
    }

    return data;
}
