# README

using python3 and keras, tensorflow(1.15)

cal_mAP.py is used for test KNN precision

other file is the train file

#### Example

```bash
python extract_feature_inception_v3.py -database ./pet/train_all -embedding inception_v3_train -json inception_v3_train.json

python cal_mAP.py -train_embedding vgg16_train.npz -query_embedding vgg16_query.npz -train_json vgg16_train.json -query_json vgg16_query.json

python cal_mAP.py -train_embedding vgg16_train.npz -query_embedding vgg16_query.npz -train_json vgg16_train.json -query_json vgg16_query.json

```

