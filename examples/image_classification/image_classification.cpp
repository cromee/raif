#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>

int main() {
    try {
        // Load the traced PyTorch model
        torch::jit::script::Module module = torch::jit::load("simple_resnet.pt");
        module.eval();

        // Prepare a dummy input tensor
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(torch::randn({1, 3, 32, 32}));

        // Run the model
        at::Tensor output = module.forward(inputs).toTensor();
        std::cout << output.sizes() << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "Error loading or running the model: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
