Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder for Unsupervised Anomaly Detection
=====

TensorFlow implementation of Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder for Unsupervised Anomaly Detection.

The official source code implemented with PyTorch is provided in <a href="https://github.com/donggong1/memae-anomaly-detection">donggong1/memae-anomaly-detection</a>.

## Architecture
<div align="center">
  <img src="./figures/xxx.png" width="400">  
  <p>Architecture of MemAE.</p>
</div>

## Graph in TensorBoard
<div align="center">
  <img src="./figures/graph.png" width="500">  
  <p>Graph of ConAD.</p>
</div>

## Problem Definition
<div align="center">
  <img src="./figures/definition.png" width="600">  
  <p>'Class-1' is defined as normal and the others are defined as abnormal.</p>
</div>

## Results
<div align="center">
  <img src="./figures/restoring.png" width="800">  
  <p>Restoration result by CondAD.</p>
</div>

<div align="center">
  <img src="./figures/test-box.png" width="350"><img src="./figures/histogram-test.png" width="390">
  <p>Box plot and histogram of restoration loss in test procedure.</p>
</div>

## Environment
* Python 3.7.4  
* Tensorflow 1.14.0  
* Numpy 1.17.1  
* Matplotlib 3.1.1  
* Scikit Learn (sklearn) 0.21.3  

## Reference
[1] Dong Gong et al. (2019). <a href="https://arxiv.org/abs/1904.02639">Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder for Unsupervised Anomaly Detection</a>. arXiv preprint arXiv:1904.02639.
