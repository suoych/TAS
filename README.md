# TAS
<<<<<<< HEAD

#### Introduction

Implementation of the paper Text Augmented Spatial-aware Zero-shot Referring Image Segmentation (EMNLP Findings 2023)

#### Preparation

1. Download Dataset (RefCOCO, RefCOCO+, RefCOCOg) and put in "../refer"

2. Prepare SAM-H, CLIP and BLIP-2 model

3. Prepare captions for images (Using BLIP-2)

4. Install the environment requirements (pip install -r requirements.txt). For syntactic parsing tools, you need to manually install some extension (en-core-web-trf in spacy, wordnet in nltk) 

#### Usage
```
python tas_main.py --config config/refcoco/refcoco_val.json
```

#### Acknowledgements

1. The repo is derived from the Grounded Segment Anything project.

2. If you have question, feel free to drop me an [e-mail](suoych@zju.edu.cn)
=======
Implementation of the paper Text Augmented Spatial Aware Zero-shot Referring Image Segmentation (Findings of EMNLP 2023)
>>>>>>> aef0f2447da1624b4efdb1be35ab1152cdd05e20
