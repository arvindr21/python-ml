#### TF Training PY Script

```shell
IMAGE_SIZE=224
ARCHITECTURE="mobilenet_0.50_${IMAGE_SIZE}"
python scripts/retrain.py \
  --bottleneck_dir=training_data/bottlenecks \
  --how_many_training_steps=500 \
  --model_dir=training_data/models/ \
  --summaries_dir=training_data/training_summaries/"${ARCHITECTURE}" \
  --output_graph=training_data/retrained_graph.pb \
  --output_labels=training_data/retrained_labels.txt \
  --architecture="${ARCHITECTURE}" \
  --image_dir=training_data/flower_photos
```


#### Test Script
Test Data Source: https://www.pexels.com/search/roses/
```shell
python tensor_flow_NN.py \
    --graph=training_data/retrained_graph.pb  \
    --image=testing_data/flower-roses-red-roses-bloom.jpg
```

##### Results
```shell
2018-02-23 16:22:31.779427: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.2 AVX AVX2 FMA

Evaluation time (1-image): 0.431s

roses 0.9996132
tulips 0.00023157008
sunflowers 0.00012243475
daisy 3.0275001e-05
dandelion 2.6256848e-06
```
