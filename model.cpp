#include <torch/torch.h>
#include <iostream>

struct AlexNetParams
{
  int kernel_size_1;
  int kernel_size_2;
  int kernel_size_3;
  int kernel_size_4;
  int kernel_size_5;

  int stride;

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

  AlexNetModel(int64_t in_dim, int64_t in_channels, int64_t out_dim, AlexNetModel model_params)
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

    // Use dynamic sizing for first dimension of FC1
    // Thus don't define it in the constructor and only in the forward pass

    fc1 = register_module(
        "fc1",
        torch::nn::Linear(
            model_params.fc_size,
            out_dim));
  }

  torch::Tensor forward(torch::Tensor x)
  {
    //... fill in initial convolutions and pooling ...
    // Input is a 256 x 256 x 3 image
    if (!fc1)
    {
      const int64_t flat_dim = x.size(1);
      fc1 = register_module("fc1", torch::nn::Linear(flat_dim, fc_size));
    }
    return x; // logits for each of 1000 ImageNet classes
  }
}