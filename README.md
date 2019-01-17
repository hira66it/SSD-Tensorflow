# SSD: Single Shot MultiBox Detector in TensorFlow

SSD is an unified framework for object detection with a single network. It has been originally introduced in this research [article](http://arxiv.org/abs/1512.02325).

This repository contains a TensorFlow re-implementation of the original [Caffe code](https://github.com/weiliu89/caffe/tree/ssd). At present, it only implements VGG-based SSD networks (with 300 and 512 inputs), but the architecture of the project is modular, and should make easy the implementation and training of other SSD variants (ResNet or Inception based for instance). Present TF checkpoints have been directly converted from SSD Caffe models.
>이 리포지토리에는 원래 Caffe 코드의 TensorFlow 재실행 기능이 있습니다. 현재 VGG 기반 SSD 네트워크(입력 300 및 512개)만 구현하고 있지만, 프로젝트의 아키텍처는 모듈식이므로 다른 SSD 변종(예: ResNet 또는 Inception 기반)의 구현과 교육을 용이하게 해야 합니다. 현재 TF 체크포인트는 SSD Caffe 모델에서 직접 변환되었습니다.


The organisation is inspired by the TF-Slim models repository containing the implementation of popular architectures (ResNet, Inception and VGG). Hence, it is separated in three main parts:
>조직은 인기 있는 아키텍처(ResNet, Inception 및 VGG)의 구현을 포함하는 TF-Slim 모델 저장소에서 영감을 얻습니다. 따라서 세 가지 주요 부분으로 구분됩니다.

* datasets: interface to popular datasets (Pascal VOC, COCO, ...) and scripts to convert the former to TF-Records;
* networks: definition of SSD networks, and common encoding and decoding methods (we refer to the paper on this precise topic);
* pre-processing: pre-processing and data augmentation routines, inspired by original VGG and Inception implementations.

## SSD minimal example

The [SSD Notebook](notebooks/ssd_notebook.ipynb) contains a minimal example of the SSD TensorFlow pipeline. Shortly, the detection is made of two main steps: running the SSD network on the image and post-processing the output using common algorithms (top-k filtering and Non-Maximum Suppression algorithm).

Here are two examples of successful detection outputs:
>SSD 노트북에는 SSD TensorFlow 파이프라인의 최소 예가 포함되어 있습니다. 곧 검출은 이미지에서 SSD 네트워크를 실행하고 공통 알고리즘(top-k 필터링 및 Non-Maximum Suppressment 알고리즘)을 사용하여 출력을 후 처리한다는 두 가지 주요 단계로 이루어집니다.

>다음은 성공적인 감지 출력의 두 가지 예입니다.
![](pictures/ex1.png "SSD anchors")
![](pictures/ex2.png "SSD anchors")

To run the notebook you first have to unzip the checkpoint files in ./checkpoint
```bash
unzip ssd_300_vgg.ckpt.zip
```
and then start a jupyter notebook with
```bash
jupyter notebook notebooks/ssd_notebook.ipynb
```


## Datasets

The current version only supports Pascal VOC datasets (2007 and 2012). In order to be used for training a SSD model, the former need to be converted to TF-Records using the `tf_convert_data.py` script:
>현재 버전은 Pascal VOC 데이터 세트만 지원합니다(2007 및 2012). SSD 모델을 교육하는 데 사용하려면 tf_convert_data.py 스크립트를 사용하여 전자를 TF-Records로 변환해야 합니다.

```bash
DATASET_DIR=./VOC2007/test/
OUTPUT_DIR=./tfrecords
python tf_convert_data.py \
    --dataset_name=pascalvoc \
    --dataset_dir=${DATASET_DIR} \
    --output_name=voc_2007_train \
    --output_dir=${OUTPUT_DIR}
```
Note the previous command generated a collection of TF-Records instead of a single file in order to ease shuffling during training.
>이전 명령에서 단일 파일 대신 TF-Records 컬렉션을 생성했으므로 교육 중에 셔플링이 용이해집니다.
## Evaluation on Pascal VOC 2007

The present TensorFlow implementation of SSD models have the following performances:

| Model | Training data  | Testing data | mAP | FPS  |
|--------|:---------:|:------:|:------:|:------:|
| [SSD-300 VGG-based](https://drive.google.com/open?id=0B0qPCUZ-3YwWZlJaRTRRQWRFYXM) | VOC07+12 trainval | VOC07 test | 0.778 | - |
| [SSD-300 VGG-based](https://drive.google.com/file/d/0B0qPCUZ-3YwWUXh4UHJrd1RDM3c/view?usp=sharing) | VOC07+12+COCO trainval | VOC07 test | 0.817 | - |
| [SSD-512 VGG-based](https://drive.google.com/open?id=0B0qPCUZ-3YwWT1RCLVZNN3RTVEU) | VOC07+12+COCO trainval | VOC07 test | 0.837 | - |

We are working hard at reproducing the same performance as the original [Caffe implementation](https://github.com/weiliu89/caffe/tree/ssd)!

After downloading and extracting the previous checkpoints, the evaluation metrics should be reproducible by running the following command:
>우리는 원래 카페인을 구현한 것과 동일한 성능을 재현하기 위해 열심히 노력하고 있습니다!

>이전 체크포인트를 다운로드하고 추출한 후 다음 명령을 실행하여 평가 메트릭을 재현할 수 있어야 합니다.
```bash
EVAL_DIR=./logs/
CHECKPOINT_PATH=./checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt
python eval_ssd_network.py \
    --eval_dir=${EVAL_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=pascalvoc_2007 \
    --dataset_split_name=test \
    --model_name=ssd_300_vgg \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --batch_size=1
```
The evaluation script provides estimates on the recall-precision curve and compute the mAP metrics following the Pascal VOC 2007 and 2012 guidelines.

In addition, if one wants to experiment/test a different Caffe SSD checkpoint, the former can be converted to TensorFlow checkpoints as following:
```sh
CAFFE_MODEL=./ckpts/SSD_300x300_ft_VOC0712/VGG_VOC0712_SSD_300x300_ft_iter_120000.caffemodel
python caffe_to_tensorflow.py \
    --model_name=ssd_300_vgg \
    --num_classes=21 \
    --caffemodel_path=${CAFFE_MODEL}
```

## Training

The script `train_ssd_network.py` is in charged of training the network. Similarly to TF-Slim models, one can pass numerous options to the training process (dataset, optimiser, hyper-parameters, model, ...). In particular, it is possible to provide a checkpoint file which can be use as starting point in order to fine-tune a network.
>스크립트 train_ssd_network입니다.py는 네트워크 훈련을 담당합니다. TF-Slim 모델과 마찬가지로 교육 프로세스에 다양한 옵션(데이터셋, 최적화자, 하이퍼 파라미터, 모델 등)을 전달할 수 있습니다. 특히 네트워크를 미세 조정하기 위해 시작 지점으로 사용할 수 있는 체크포인트 파일을 제공할 수 있습니다.
### Fine-tuning existing SSD checkpoints

The easiest way to fine the SSD model is to use as pre-trained SSD network (VGG-300 or VGG-512). For instance, one can fine a model starting from the former as following:
>SSD 모델을 미세 조정하는 가장 쉬운 방법은 사전 교육을 받은 SSD 네트워크(VGG-300 또는 VGG-512)로 사용하는 것입니다. 예를 들어, 다음과 같이 모델부터 시작하여 모델을 미세 조정할 수 있습니다.
```bash
DATASET_DIR=./tfrecords
TRAIN_DIR=./logs/
CHECKPOINT_PATH=./checkpoints/ssd_300_vgg.ckpt
python train_ssd_network.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=pascalvoc_2012 \
    --dataset_split_name=train \
    --model_name=ssd_300_vgg \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --save_summaries_secs=60 \
    --save_interval_secs=600 \
    --weight_decay=0.0005 \
    --optimizer=adam \
    --learning_rate=0.001 \
    --batch_size=32
```
Note that in addition to the training script flags, one may also want to experiment with data augmentation parameters (random cropping, resolution, ...) in `ssd_vgg_preprocessing.py` or/and network parameters (feature layers, anchors boxes, ...) in `ssd_vgg_300/512.py`
>교육 스크립트 플래그 외에도 ssd_vgg_preprocessing.py의 데이터 증가 매개 변수(랜덤 자르기, 해상도 등)와 ssd_vgg_300/512.py의 네트워크 매개 변수(기능 계층, 앵커 상자 등)를 사용하여 테스트할 수도 있습니다.

Furthermore, the training script can be combined with the evaluation routine in order to monitor the performance of saved checkpoints on a validation dataset. For that purpose, one can pass to training and validation scripts a GPU memory upper limit such that both can run in parallel on the same device. If some GPU memory is available for the evaluation script, the former can be run in parallel as follows:

>또한 교육 스크립트를 평가 루틴과 결합하여 검증 데이터 세트에 저장된 체크포인트의 성능을 모니터링할 수 있습니다. 이를 위해 교육 및 검증 스크립트에 GPU 메모리 상한을 전달하여 두 메모리 모두 동일한 장치에서 병렬로 실행할 수 있습니다. 평가 스크립트에 일부 GPU 메모리를 사용할 수 있는 경우 다음과 같이 전자를 병렬로 실행할 수 있습니다.

```bash
EVAL_DIR=${TRAIN_DIR}/eval
python eval_ssd_network.py \
    --eval_dir=${EVAL_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=pascalvoc_2007 \
    --dataset_split_name=test \
    --model_name=ssd_300_vgg \
    --checkpoint_path=${TRAIN_DIR} \
    --wait_for_checkpoints=True \
    --batch_size=1 \
    --max_num_batches=500
```

### Fine-tuning a network trained on ImageNet

One can also try to build a new SSD model based on standard architecture (VGG, ResNet, Inception, ...) and set up on top of it the `multibox` layers (with specific anchors, ratios, ...). For that purpose, you can fine-tune a network by only loading the weights of the original architecture, and initialize randomly the rest of network. For instance, in the case of the [VGG-16 architecture](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz), one can train a new model as following:
>또한 표준 아키텍처(VGG, ResNet, Inception 등)를 기반으로 새로운 SSD 모델을 구축하고 멀티박스 계층(특정 앵커, 비율 등)을 그 위에 설정할 수 있습니다. 이를 위해 원래 아키텍처의 가중치만 로드하여 네트워크를 미세 조정하고 나머지 네트워크를 임의로 초기화할 수 있습니다. 예를 들어 VGG-16 아키텍처의 경우 다음과 같이 새 모델을 교육할 수 있습니다.
```bash
DATASET_DIR=./tfrecords
TRAIN_DIR=./log/
CHECKPOINT_PATH=./checkpoints/vgg_16.ckpt
python train_ssd_network.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=pascalvoc_2007 \
    --dataset_split_name=train \
    --model_name=ssd_300_vgg \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_model_scope=vgg_16 \
    --checkpoint_exclude_scopes=ssd_300_vgg/conv6,ssd_300_vgg/conv7,ssd_300_vgg/block8,ssd_300_vgg/block9,ssd_300_vgg/block10,ssd_300_vgg/block11,ssd_300_vgg/block4_box,ssd_300_vgg/block7_box,ssd_300_vgg/block8_box,ssd_300_vgg/block9_box,ssd_300_vgg/block10_box,ssd_300_vgg/block11_box \
    --trainable_scopes=ssd_300_vgg/conv6,ssd_300_vgg/conv7,ssd_300_vgg/block8,ssd_300_vgg/block9,ssd_300_vgg/block10,ssd_300_vgg/block11,ssd_300_vgg/block4_box,ssd_300_vgg/block7_box,ssd_300_vgg/block8_box,ssd_300_vgg/block9_box,ssd_300_vgg/block10_box,ssd_300_vgg/block11_box \
    --save_summaries_secs=60 \
    --save_interval_secs=600 \
    --weight_decay=0.0005 \
    --optimizer=adam \
    --learning_rate=0.001 \
    --learning_rate_decay_factor=0.94 \
    --batch_size=32
```
Hence, in the former command, the training script randomly initializes the weights belonging to the `checkpoint_exclude_scopes` and load from the checkpoint file `vgg_16.ckpt` the remaining part of the network. Note that we also specify with the `trainable_scopes` parameter to first only train the new SSD components and left the rest of VGG network unchanged. Once the network has converged to a good first result (~0.5 mAP for instance), you can fine-tuned the complete network as following:
>따라서 이전 명령에서 교육 스크립트는 checkpoint_excope에 속한 가중치를 임의로 초기화하고 네트워크의 나머지 부분 vg_16.ckpt에서 로드합니다. 또한 Trainable_scopes 매개 변수를 지정하여 먼저 새 SSD 구성 요소만 교육하고 나머지 VGG 네트워크를 변경하지 않도록 합니다. 네트워크가 양호한 첫 번째 결과(예: ~0.5mAP)로 수렴되면 다음과 같이 전체 네트워크를 미세 조정할 수 있습니다.
```bash
DATASET_DIR=./tfrecords
TRAIN_DIR=./log_finetune/
CHECKPOINT_PATH=./log/model.ckpt-N
python train_ssd_network.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=pascalvoc_2007 \
    --dataset_split_name=train \
    --model_name=ssd_300_vgg \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_model_scope=vgg_16 \
    --save_summaries_secs=60 \
    --save_interval_secs=600 \
    --weight_decay=0.0005 \
    --optimizer=adam \
    --learning_rate=0.00001 \
    --learning_rate_decay_factor=0.94 \
    --batch_size=32
```

A number of pre-trained weights of popular deep architectures can be found on [TF-Slim models page](https://github.com/tensorflow/models/tree/master/slim).
>TF-Slim 모델 페이지에서 널리 사용되는 심층 아키텍처에 대한 사전 교육된 여러 가중치를 확인할 수 있습니다.
