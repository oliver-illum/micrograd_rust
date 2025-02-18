# Micrograd in Rust

This repository is an experimental Rust implementation of Andrej Karpathy’s [Micrograd tutorial](https://www.youtube.com/watch?v=VMj-3S1tku0&t=4698s). It explores the core concepts of a tiny autograd engine, reimagined in Rust.

## Watch the Tutorial

[![Watch on YouTube](https://img.shields.io/badge/Watch%20on-YouTube-red?style=for-the-badge)](https://www.youtube.com/watch?v=VMj-3S1tku0&t=4698s)

[![YouTube Thumbnail](https://img.youtube.com/vi/VMj-3S1tku0/maxresdefault.jpg)](https://www.youtube.com/watch?v=VMj-3S1tku0&t=4698s)

## Introduction

In this project, I aimed to follow the micrograd tutorial as closely as possible while adapting it to Rust. Due to Rust’s strict ownership and borrow rules, and the dynamic nature of an automatic differentiation engine—which requires shared, mutable references to nodes—I had to make creative design decisions to allow multiple parts of the computation graph to share and update values during backpropagation.

In this repository, you will find the complete implementation of micrograd in Rust, along with a simple multi-layer perceptron (MLP). The only external crate used is `rand`, which is utilized for generating random weights during the neural network’s initialization.

## Implementation Challenges

During development, several challenges arose:

1. **Shared Ownership and Mutability:**  
   Rust’s ownership rules meant that I had to wrap the core `Value` struct in an `Rc<RefCell<...>>`. This pattern is common in Rust for cases where you need both shared ownership and interior mutability, but it required careful management to avoid runtime borrow errors.

2. **Operator Overloading:**  
   Since Rust doesn’t have magic methods like Python, every operation (addition, multiplication, power, etc.) had to be implemented via the corresponding trait. Special care was needed for operations like exponentiation (`.pow()`) because Rust doesn’t have a built-in operator for it. I implemented `.pow()` as a method, and for some operations (like power), I even resorted to parsing parameters from a string in the `_op` field—though a better design might store such parameters explicitly.

3. **Backward Pass and Borrowing:**  
   A significant challenge was ensuring that during backpropagation no conflicting mutable borrows occur. In Rust, you can only have one mutable borrow at a time, so when calling the backward functions, I had to ensure that temporary immutable borrows were dropped before new mutable borrows were created. To work around these issues, I chose to store the backward functions as function pointers (inside an `Option`) rather than closures that capture variables.

## Usage

To run the project, simply execute:

```bash
cargo build
```

```bash
cargo run
```

## Results

The main file tests the simple MLP using the same small dataset as shown in the tutorial. Initially, the predictions are off. The network is trained for 100 iterations, with the mean squared loss printed every 10 iterations, and finally, the prediction after the 101st backpropagation is displayed.

Below is an example of the output:

```plaintext
First predictions:
Sample 0: Prediction = 0.00814962088529803
Sample 1: Prediction = 0.5640571294882423
Sample 2: Prediction = 0.6283985267256972
Sample 3: Prediction = 0.25408125509888685

Epoch 0: Loss = 6.6381184146902505
Epoch 10: Loss = 0.06953021047692366
Epoch 20: Loss = 0.03587764387922449
Epoch 30: Loss = 0.023916247335598453
Epoch 40: Loss = 0.017854165083438733
Epoch 50: Loss = 0.014208895414006356
Epoch 60: Loss = 0.011782189119982193
Epoch 70: Loss = 0.010053569722137914
Epoch 80: Loss = 0.008761229037721184
Epoch 90: Loss = 0.007759361662150957
Epoch 100: Loss = 0.006960435621501898

Final predictions:
Sample 0: Prediction = 0.9676195017562665
Sample 1: Prediction = -0.9700001802282829
Sample 2: Prediction = -0.9502409179751493
Sample 3: Prediction = 0.9503520611992682
```
