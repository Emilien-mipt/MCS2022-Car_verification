# My experiments
1. Baseline with default parameters: resnet18 with SGD
    
  * Epoch 52; val loss: 0.2108; val acc: 0.95; public ROC-AUC: 0.78567
2. Baseline with resnet50 and default parameters

  * Epoch 64; val loss: 0.1434; val acc: 0.96; public ROC-AUC: 0.79488
3. Baseline with efficientnet_b3 and default parameters

  * Epoch 83; val loss: 0.117; val acc: 0.9711; public ROC-AUC: 0.80404

### Final experiment
* Model: efficientnetv2_rw_s
* Losses: circle+contrast
* Training details: batch size=32; input size=224; 
Adam optimizer with learning rate=0.00005; StepLR scheduler with step size=9 epochs

With these parameters I managed to get ~0.89 on public, and ~0.87 on private leaderboard.