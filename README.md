# iPet - Image Retrieval & Visualizatin System Based on Deep Learning

肥胖可怜又无助队，11号

#### Introduction

We design an web-based visualization system for image retrieval and exploration. The users can choose
an query image, retrieve similar images from a large database in real time. We use pet dataset and get image feature using different netural networks(e.g. ADSH, VGG16, ResNet50, Inception_v3..). Meanwhile we evalutaion the KNN map precision through our experiment.

<img src="./assets/teaser.png">

#### Architecture

the file architecture is

```bash
.
├── assets
├── backend
│   ├── data
│   │   ├── hash_label
│   │   ├── inception_label
│   │   ├── resnet_label
│   │   ├── vgg_label
│   │   └── vgg_raw
│   └── src
├── deeplearning
│   ├── ADSH
│   │   ├── data
│   │   ├── log
│   │   └── utils
│   │       └── __pycache__
│   └── network_traninig_withlabel
├── frontend
│   └── src
└── image
    └── file
```

our system is using react, flask, d3, ant design and so on. the system code is in frontend and backend folder. the deeplearning training process is in deeplearning folder.

#### Install and Run

```bash
# new terminal
cd backend
pip install -r requirements.txt
python router.py

# new terminal
cd frontend
npm install
npm run start
```
then go to [localhost:8080](http://localhost:8080)



