/*
  C++ from Python source code translations using LibTorch 
  of "Make Your Own Neural Network" book by Tariq Rashid 
  https://github.com/makeyourownneuralnetwork
  code for a 3-layer neural network, and code for learning the MNIST dataset
  (c) Ole Roel, 2018
  license is GPLv3
*/

#include <torch/torch.h>
#include <iostream>
#include <cmath>
#include <chrono>

// using namespace torch;

// Hyper-parameters 
constexpr int64_t kInputSize = 784LL;
constexpr int64_t kHiddenSize = 200LL;
constexpr int64_t kNumClasses = 10LL;
constexpr float kLearningRate = 0.1f;

// The batch size for training.
constexpr int64_t kBatchSize = 64;

// The number of epochs to train.
constexpr int64_t kNumberOfEpochs = 5;

typedef torch::data::transforms::Normalize<> NORMALIZE;
typedef torch::data::transforms::Stack<> STACK;

template <class Dataset>
void run_train(torch::nn::Sequential& model, Dataset& ds, torch::Device& device) {
    auto train_loader =
        torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(ds), kBatchSize);
    torch::Tensor fake_labels = torch::nn::init::constant_(torch::empty(10, device), 0.01f);

    torch::optim::SGD model_optimizer(model->parameters(),
                      torch::optim::SGDOptions(kLearningRate));

    for (int64_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
        for (torch::data::Example<>& batch : *train_loader) {
            torch::Tensor real_images = batch.data.to(device);
            torch::Tensor targets = batch.target;

            int64_t batch_size = targets.sizes()[0];

            for (int64_t i = 0; i <  batch_size; ++i) {
                torch::Tensor output = model->forward(real_images[i].reshape(kInputSize));
                
                fake_labels[targets[i]] = 0.99;
        		torch::Tensor loss = torch::mse_loss(output, fake_labels, Reduction::None);
                model_optimizer.zero_grad();
                loss.backward();
                model_optimizer.step();
                fake_labels[targets[i]] = 0.01;
            }
        }
    }
}

template <class Dataset>
void run_test(torch::nn::Sequential& model, Dataset& ds, torch::Device& device) {
    auto test_loader =
        torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(ds), kBatchSize);

    std::vector<int> scorecard;
    for (torch::data::Example<>& batch : *test_loader) {
        torch::Tensor real_images = batch.data.to(device);
        torch::Tensor targets = batch.target;
        int64_t batch_size = targets.sizes()[0];

        for (int64_t i = 0; i <  batch_size; ++i) {
            torch::Tensor output = model->forward(real_images[i].reshape(784));
            int64_t numel = output.numel();
            int64_t expected = targets[i].data<int64_t>()[0];
            int64_t label = std::distance(output.data<float>(), std::max_element(output.data<float>(), output.data<float>()+numel));
            // std::cout << "numel: " << numel << std::endl;
            // std::cout << "expected: " << expected << std::endl;
            // std::cout << "label: " << label << std::endl;
            // std::cout << "output.data<float>(): " << output.data<float>() << std::endl;
            // for (int64_t i = 0; i < numel; ++i) {
            //     std::cout << "output.data<float>()[" << i << "]: " << output.data<float>()[i] << std::endl;
            // }

            if (label == expected) {
                // std::cout << "found correct label: " << correct_label << std::endl;
                scorecard.push_back(1);
            } else {
                // std::cout << ":-( label: " << label << " should be: " << correct_label << std::endl;
                scorecard.push_back(0);
            }
        }
    }

    int sum = std::accumulate(scorecard.begin(), scorecard.end(), 0);
    std::cout << "performance = " << std::setw(6) << float(sum)/float(scorecard.size()) << std::endl;
}

int main() {
  torch::manual_seed(1);

  // Create the device we pass around based on whether CUDA is available.
  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available()) {
    std::cout << "CUDA is available! Training on GPU." << std::endl;
    device = torch::Device(torch::kCUDA);
  }

  torch::nn::Sequential model(
      torch::nn::Linear(torch::nn::LinearOptions(kInputSize, kHiddenSize).with_bias(false)),
      torch::nn::Functional(torch::sigmoid),
      torch::nn::Linear(torch::nn::LinearOptions(kHiddenSize, kNumClasses).with_bias(false)),
      torch::nn::Functional(torch::sigmoid)
      );

  for (auto&& layer : *model) {
    std::shared_ptr<torch::nn::Module> module_ptr = layer.ptr();
    module_ptr->apply([](torch::nn::Module& module) {
       torch::NoGradGuard no_grad;
       if (auto* linear = module.as<torch::nn::Linear>()) {
            std::cout << "i.sizes():\n" << linear->weight.sizes() << std::endl;
            float N = linear->weight.sizes()[1];
            float sigma{std::pow(N, -0.5F)};
            std::cout << "sigma: " << sigma << std::endl;
            linear->weight.normal_(0.0, sigma);
       }
    });
  }

  model->to(device);
#if 1
    auto start_time = std::chrono::high_resolution_clock::now();
#if 1
    {
        auto ds_mnist = torch::data::datasets::MNIST("./mnist")
                .map(NORMALIZE(0.5, 0.5))
                .map(STACK());
        auto& t1 = std::chrono::high_resolution_clock::now();
        run_train(model, ds_mnist, device);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "training took "
                << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
                << " milliseconds" << std::endl;
    }
#endif
    {
        auto ds_mnist = torch::data::datasets::MNIST("./mnist")
                .map(NORMALIZE(0.5, 0.5))
                .map(STACK());
        auto t1 = std::chrono::high_resolution_clock::now();
        run_test(model, ds_mnist, device);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "test took "
                << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
                << " milliseconds" << std::endl;
    }

    auto end_time = std::chrono::high_resolution_clock::now();

    std::cout << "total run-time "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end_time-start_time).count()
            << " milliseconds" << std::endl;    
#endif
}