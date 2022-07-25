# Machines can see 2022 - Car Model Verification
The following pipeline scored **~0.89** on public (**top 15**) and **~0.87** on private (**top 5**). 
Though the pipeline was intended to solve the car verification problem it can also be used to solve any verification 
or identification problem.

## Competition overview
In this competition, participants needed to train a model to verify car models 
(models are the same, not the same car).

The idea of the solution is to remove the classification layer from the classifier (backbone model), 
add the losses that are used for metric learning (such as arcface, cosface and e.t.c)
and train the resulting model. After that the embeddings are extracted to measure the proximity between two images.

## Steps for working with baseline
### 0. Download CompCars dataset
To train the model in this offline the CompCars dataset is used. You can download it [here](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html).

If you have problems getting box labels in datasets, you can use a duplicate of the labels that we posted here.
### 1. Prepare data for classification
Launch `prepare_data.py` to crop images on bboxes and generate lists for training and validation phases.
```bash
python prepare_data.py --data_path ../CompCars/data/ --annotation_path ../CompCars/annotation/
```

### 2. Run model training
```bash                         
CUDA_VISIBLE_DEVICES=0 python main.py --cfg config/baseline_mcs.yml
```
### 3. Create a submission file

```bash
CUDA_VISIBLE_DEVICES=0 python create_submission.py --exp_cfg config/baseline_mcs.yml \
                                                   --checkpoint_path experiments/baseline_mcs/model_0077.pth \
                                                   --inference_cfg config/inference_config.yml
```