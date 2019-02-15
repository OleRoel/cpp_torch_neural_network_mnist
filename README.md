# cpp_torch_neural_network_mnist
This is a coding experiment to compare speed of a TorchLib C++ implementation for training a MNIST network against C++ without using a lib.

The repository with the "lib-less" C++ MNIST training code can be found here: https://github.com/OleRoel/cpp_neural_network_mnist

The original Python code can be found in the "Code for the Make Your Own Neural Network book" in Tariq Rashid's repository here: https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork

## Data preparation
```sh
python download_mnist.py -d=mnist
```

The MNIST download script has been copied from here: https://github.com/pytorch/pytorch/blob/master/tools/download_mnist.py

## Building

```sh
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=</path/to/libtorch> ..
make
```

## Running

```sh
./mnist
```

## Performance
|              | Performance | Train Time [s] | Test Time [s] |
| ------------ |------------:| ---------------:|-------------:|
| **cblas**    |      0.9668 |          37.192 |        0.200 |
| **blis**     |      0.9667 |          17.471 |        0.122 |
| **MKL**      |      0.9664 |          16.406 |        0.098 |
| **cuBLAS**   |      0.9624 |          66.196 |        0.735 |
| **Python**   |      0.9668 |         260.706 |        1.362 |
| **LibTorch** |      0.8743 |         787,088 |       52,422 |

The LibTorch trained only 5 epochs instead of 10.
My code is in an early stage, I need to look into compile flags and there is very likely something
to improve in my code.

Hardware used:<br>
MacBook Pro (15-inch, 2018), 2,9 GHz Intel Core i9<br>

