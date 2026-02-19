#include <torch/torch.h>
#include <iostream>
#include <vector>
using std::vector;

struct AlexNetParams
{
    int kernel_size_1;
    int kernel_size_2;
    int kernel_size_3;
    int kernel_size_4;
    int kernel_size_5;

    int stride;

    int pool_size;

    int conv_channels_1;
    int conv_channels_2;
    int conv_channels_3;
    int conv_channels_4;
    int conv_channels_5;

    int fc_size;
}

struct AlexNetModel : torch::nn::Module
{
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, conv4{nullptr}, conv5{nullptr};

    torch::nn::Linear fc1{nullptr}, fc2{nullptr};

    AlexNetModel(int64_t in_dim, int64_t in_channels, int64_t out_dim, AlexNetParams model_params)
    {
        conv1 = register_module(
            "conv1",
            torch::nn::Conv2d(
                in_channels,
                model_params.conv_channels_1,
                model_params.kernel_size_1, stride = model_params.stride));

        conv2 = register_module(
            "conv2",
            torch::nn::Conv2d(
                model_params.conv_channels_1,
                model_params.conv_channels_2,
                model_params.kernel_size_2, stride = model_params.stride));

        conv3 = register_module(
            "conv3",
            torch::nn::Conv2d(
                model_params.conv_channels_2,
                model_params.conv_channels_3,
                model_params.kernel_size_3, stride = model_params.stride));

        conv4 = register_module(
            "conv4",
            torch::nn::Conv2d(
                model_params.conv_channels_3,
                model_params.conv_channels_4,
                model_params.kernel_size_4, stride = model_params.stride));

        conv5 = register_module(
            "conv5",
            torch::nn::Conv2d(
                model_params.conv_channels_4,
                model_params.conv_channels_5,
                model_params.kernel_size_5, stride = model_params.stride));

        pool = register_module(
            "pool",
            torch::nn::MaxPool2d(model_params.pool_size));

        // Use dynamic sizing for first dimension of FC1
        // Thus don't define it in the constructor and only in the forward pass

        fc1 = register_module(
            "fc1",
            torch::nn::Linear(
                model_params.fc_size,
                out_dim));
    }

    torch::Tensor forward(torch::Tensor x, AlexNetParams model_params)
    {
        //... fill in initial convolutions and pooling ...
        x = conv1(x);

        // Pool after first convolution
        x = pool(x);
        x = conv2(x);

        x = pool(x); // pool twice

        // last 3 convolutions directly adjoining

        x = conv3(x);
        x = conv4(x);
        x = conv5(x);

        flattened_x = torch::flatten(x);

        const int64_t flat_dim = flattened_x.size(1);
        fc1 = register_module("fc1", torch::nn::Linear(flat_dim, model_params.stride));

        x = fc1(x);
        x = fc2(x);

        logits = torch::nn::softmax(x);
        return logits; // logits for each of 1000 ImageNet classes
    }
}

vector<double>
train(
    torch::n::Module &model,
    torch::optim::Optimizer &optimizer,
    torch::nn::Module &criterion,
    torch::data::DataLoader &data,
    int epochs = 10,
    int lr = 0.2)
{
    vector<double> losses;
    vector<double> cur_losses;

    model.train(); // first model in train mode

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        for (auto const &batch : *data)
        {
            optimizer.zero_grad();
            auto data = batch.data;
            auto target = batch.target;

            torch::nn::tensor model_preds = model.forward(data);
            torch::nn::float loss = criterion(model_preds, target);
            cur_losses.push_back(loss.item<double>());

            loss.backward();
            optimizer.step();
        }

        double sum = std::accumulate(cur_losses.begin(), cur_losses.end(), 0.0);
        double avg_loss = sum / cur_losses.size();
        std::cout
            << "Epoch " << epoch << " loss: " << last << std::endl;

        losses.insert(std::end(losses), std::begin(cur_losses), std::end(cur_losses));

        cur_losses.clear();
    }
    return losses
}

void evaluate()
{
}