# Memotion Multi-Modal Model (M4 Model)
The growing ubiquity of Internet memes on social media platforms, such as Facebook, Twitter, Instagram and Reddit have become a unstoppable trend. Comprising of visual and textual information, memes typically convey various emotion (e.g. humor, sarcasm, offensive, motivational, and sentiment). Nevertheless, there is not much research attention toward meme emotion analysis. The main objective of this project is to bring the attention of the research community toward meme studying. In this project, we propose the **M**e**m**otion **M**ulti**m**odal Model (M4 Model) newtork for humor detection on [memotion dataset](https://arxiv.org/pdf/2008.03781.pdf). Inspired by [ConcatBERT model](https://github.com/IsaacRodgz/ConcatBERT), we propose our model while the core "Gated Multimodal Layer" (GML) are proposed by this [arXiv paper](https://arxiv.org/pdf/1702.01992.pdf)

<p align="center">
  <img src="https://github.com/terenceylchow124/Meme-MultiModal/blob/main/Project.png" width="550" height="300">
</p>

# Dataset Preparation 
In this project, memotion Dataset are mainly used while we apply transfer learning using Reddit Dataset.
1. Download offical Memotion Dataset from [kaggle](https://www.kaggle.com/williamscott701/memotion-dataset-7k)
2. Download and prepare sampled Reddit Dataset from [this repository](https://github.com/orionw/RedditHumorDetection)
3. Put your Reddit and Memotion Dataset to ./data/reddit and ./data/memotion accordingly. 

# Procedures
1. Training the ALBERT model using Reddit Dataset.  
  > - go to utils/util_args/.  
  > - set "train_reddit" to 1
  > - set "dataset" to "reddit" 
  > - set "model" to "RedditBert"
  > - set "bert_model" to "bert-base-uncased"
  > - feel free to tune any hyper-parameter, i.e max_iter, batch,...
  > 
  > - run "python main.py"
2. After training, the trained model and corresponding log file should be stored under ./pre_trained_models/.  
3. Training the MultiModal model using Memotion Dataset. 
  > - set "train_reddit" to 0, "dataset" to "memotion" and "model" to "GatedAverageBERT" in utils/util_args. 
  > 
  > - load the pretrained bert model by modify the model name in initiate() function of main.py.
  > 
  > - run "python main.py"

# Pretrained Model 
Resulted log files can be found under pre_trained_models/. 
| Variant | Model            | Task          | Test Acc % | Macro-F1-Score %  | Benchmark % | Download Link |
| ------- | ---------------- | ------------- | ---------- | ----------------- | ----------- | ----- |
| A2      | ALBERT+FC        | Reddit        | 60.79      | 55.96             | [72.40 (Acc)](https://arxiv.org/pdf/1909.00252.pdf) | [:white_check_mark:](https://drive.google.com/file/d/16ArUFaJG6tfkyQEsq7unxg9u8nmni-q-/view?usp=sharing) |
| A21     | ALBERT+FC+VGG16  | Memotion      | 68.32      | 54.57             | [52.99 (F1)](https://arxiv.org/pdf/2008.03781.pdf)  | [:white_check_mark:](https://drive.google.com/file/d/1Y78Kto6axhWLxf0mucsEW0WVCCHSjh-8/view?usp=sharing) |
 
# Acknowledgment
This code is partial borrowed from [ConcatBERT model](https://github.com/IsaacRodgz/ConcatBERT).





