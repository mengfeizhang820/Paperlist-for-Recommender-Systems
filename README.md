# Recommendation_system_paperlist 

## Survey paper
* Deep Learning based Recommender System: A Survey and New Perspectives [2017][[__PDF__]](https://arxiv.org/pdf/1707.07435.pdf)
* 基于深度学习的推荐系统研究综述 [2018] [[__PDF__](http://cjc.ict.ac.cn/online/bfpub/hlww-2018124152810.pdf)]
* Explainable Recommendation: A Survey and New Perspectives [2018] [[__PDF__]](https://arxiv.org/pdf/1804.11192.pdf)
* Sequence-Aware Recommender Systems [2018] [[__PDF__]](https://arxiv.org/pdf/1802.08452.pdf)


## Recommendation Systems with Text Information
  ### review-based approach
  * Convolutional Matrix Factorization for Document Context-Aware Recommendation [RecSys 2016] [[__PDF__]](http://delivery.acm.org/10.1145/2960000/2959165/p233-kim.pdf?ip=159.226.43.46&id=2959165&acc=ACTIVE%20SERVICE&key=33E289E220520BFB%2ED25FD1BB8C28ADF7%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1568101423_225c8120374449cc9c20766fe8a2911c) 
    - code : https://github.com/cartopy/ConvMF
  * Joint Deep Modeling of Users and Items Using Reviews for Recommendation [WSDM 2017][[__PDF__]](https://arxiv.org/pdf/1701.04783.pdf)
    - code : https://github.com/chenchongthu/DeepCoNN
  * Multi-Pointer Co-Attention Networks for Recommendation [KDD 2018][[__PDF__]](https://arxiv.org/pdf/1801.09251)
    - code : https://github.com/vanzytay/KDD2018_MPCN
  * Gated attentive-autoencoder for content-aware recommendation [WSDM 2019][[__PDF__]](https://arxiv.org/pdf/1812.02869)
    - code : https://github.com/allenjack/GATE


## collaborative filtering approach
  * Neural Collaborative Filtering [2017][[__PDF__]](https://arxiv.org/pdf/1708.05031.pdf)
    - code :https://paperswithcode.com/paper/neural-collaborative-filtering-1#code
  * Outer Product-based Neural Collaborative Filtering [IJCAI 2018][[__PDF__]](https://arxiv.org/pdf/1808.03912v1.pdf)
    - code :https://github.com/duxy-me/ConvNCF
  * DeepCF : A Unified Framework of Representation Learning and Matching Function Learning in Recommender System [AAAI 2019][[__PDF__]](https://arxiv.org/pdf/1901.04704v1.pdf)
  * Neural Graph Collaborative Filtering [SIGIR 2019] [[__PDF__]](https://arxiv.org/pdf/1905.08108v1.pdf)
    - code : https://paperswithcode.com/paper/neural-graph-collaborative-filtering
  * Transnets: Learning to transform for recommendation [RecSys 2017][[__PDF__]](https://arxiv.org/pdf/1704.02298)
    - code : https://github.com/rosecatherinek/TransNets

  
  
    
## Explainable Recommendation Systems
* Explainable Recommendation via Multi-Task Learning in Opinionated Text Data [SIGIR 2018][[__PDF__]](https://arxiv.org/pdf/1806.03568.pdf)
* TEM: Tree-enhanced Embedding Model for Explainable Recommendation [WWW 2018][[__PDF__]](http://staff.ustc.edu.cn/~hexn/papers/www18-tem.pdf)
* Neural Attentional Rating Regression with Review-level Explanations [WWW 2018] [[__PDF__]](http://www.thuir.cn/group/~YQLiu/publications/WWW2018_CC.pdf)
   - code : https://github.com/chenchongthu/NARRE

## Sequence-Aware Recommendation Systems

### Session-based Recommendation Systems
* Session-based Recommendations with Recurrent Neural Networks [ICLR 2016] [[__PDF__]](https://arxiv.org/pdf/1511.06939.pdf)
  - code : https://github.com/hidasib/GRU4Rec
* Neural Attentive Session-based Recommendation [CIKM 2017] [[__PDF__]](https://arxiv.org/pdf/1711.04725.pdf)
  - code : https://github.com/lijingsdu/sessionRec_NARM
* When Recurrent Neural Networks meet the Neighborhood for Session-Based Recommendation [RecSys 2017][[__PDF__]](http://ls13-www.cs.tu-dortmund.de/homepage/publications/jannach/Conference_RecSys_2017.pdf)
* STAMP: Short-Term Attention/Memory Priority Model for Session-based Recommendation [KDD 2018] [[__PDF__]](https://dl.acm.org/ft_gateway.cfm?id=3219950&type=pdf)
  - code : https://github.com/uestcnlp/STAMP
* RepeatNet: A Repeat Aware Neural Recommendation Machine for Session-based Recommendation [AAAI 2019][[__PDF__]](https://staff.fnwi.uva.nl/m.derijke/wp-content/papercite-data/pdf/ren-repeatnet-2019.pdf)
  - code : https://github.com/PengjieRen/RepeatNet
* Session-based Recommendation with Graph Neural Networks [AAAI 2019][[__PDF__]](https://arxiv.org/pdf/1811.00855.pdf)
  - code : https://github.com/CRIPAC-DIG/SR-GNN
* Streaming Session-based Recommendation [KDD 2019] [[__PDF__]](http://delivery.acm.org/10.1145/3340000/3330839/p1569-guo.pdf?ip=159.226.43.46&id=3330839&acc=ACTIVE%20SERVICE&key=33E289E220520BFB%2ED25FD1BB8C28ADF7%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1568101881_a5ecfb5dd698417af3d9c29d32e00c86)

### Graph based approach
* Graph Convolutional Neural Networks for Web-Scale Recommender Systems [KDD 2018][[__PDF__]](https://arxiv.org/pdf/1806.01973)
* Session-based Social Recommendation via Dynamic Graph Attention Networks [WSDM 2019][[__PDF__]](http://www.cs.toronto.edu/~lcharlin/papers/fp4571-songA.pdf)
  - code : https://github.com/DeepGraphLearning/RecommenderSystems/tree/master/socialRec  

### Last-N based approach 
* Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding [WSDM 2018]
  - code : https://github.com/graytowne/caser_pytorch [Pytorch]
* Hierarchical Gating Networks for Sequential Recommendation [KDD 2019][[__PDF__]](https://arxiv.org/pdf/1906.09217.pdf)
  - code : https://github.com/graytowne/caser_pytorch
* Self-Attentive Sequential Recommendation [ICDM 2018] [[__PDF__]](https://arxiv.org/pdf/1808.09781)
  - code : https://github.com/kang205/SASRec

### Long and short-term sequential recommendation systems
* Next Item Recommendation with Self-Attention [ACM 2018][[__PDF__]](https://arxiv.org/pdf/1808.06414)
  - code : https://github.com/cheungdaven/DeepRec/blob/master/models/seq_rec/AttRec.py
* Collaborative Memory Network for Recommendation Systems [SIGIR 2018][[__PDF__]](https://arxiv.org/pdf/1804.10862)
  - code : https://github.com/tebesu/CollaborativeMemoryNetwork
* Sequential Recommender System based on Hierarchical Attention Network [IJCAI 2018] [[__PDF__]](https://www.ijcai.org/proceedings/2018/0546.pdf)
  - code : https://github.com/uctoronto/SHAN
  
### Context-Aware Sequential Recommendations
* Context-Aware Sequential Recommendations withStacked Recurrent Neural Networks [[__PDF__]](http://delivery.acm.org/10.1145/3320000/3313567/p3172-rakkappan.pdf?ip=159.226.43.46&id=3313567&acc=ACTIVE%20SERVICE&key=33E289E220520BFB%2ED25FD1BB8C28ADF7%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1568094616_ce11de1b93788a1072a48c6212d24287)
  - code : https://bitbucket.org/cdal/stackedcontextawarernn
  
### others
* Hierarchical Neural Variational Model for Personalized Sequential Recommendation [WWW 2019]
* Online Purchase Prediction via Multi-Scale Modeling of Behavior Dynamics [KDD 2019]
* Log2Intent: Towards Interpretable User Modeling via Recurrent Semantics Memory Unit [KDD 2019]
* Taxonomy-aware Multi-hop Reasoning Networks for Sequential Recommendation [WSDM 2019]
  - code : https://github.com/RUCDM/TMRN

## Knowledge Graph-based Recommendations
* DKN: Deep Knowledge-Aware Network for News Recommendation [WWW 2018] [[__PDF__]](https://arxiv.org/pdf/1801.08284.pdf)
  - code : https://github.com/hwwang55/DKN
* RippleNet: Propagating User Preferences on the Knowledge Graph for Recommender Systems [CIKM 2018] [[__PDF__]](https://arxiv.org/abs/1803.03467)
  - code : https://github.com/hwwang55/RippleNet
* Knowledge Graph Convolutional Networks for Recommender Systems [WWW 2019] [[__PDF__]](https://arxiv.org/pdf/1904.12575.pdf)
  - code : https://github.com/hwwang55/KGCN
* Improving Sequential Recommendation with Knowledge-Enhanced Memory Networks [SIGIR 2018] [[__PDF__]](http://delivery.acm.org/10.1145/3220000/3210017/p505-huang.pdf?ip=159.226.43.46&id=3210017&acc=ACTIVE%20SERVICE&key=33E289E220520BFB%2ED25FD1BB8C28ADF7%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1568100770_6bdfb19ed107162bbc2ad22e7ebf1463)
  - dataset and code : https://github.com/RUCDM/KB4Rec
  
## Reinforcement learning approachs
* DRN: A Deep Reinforcement Learning Framework for News Recommendation [WWW 2018] [[__PDF__]](http://www.personal.psu.edu/~gjz5038/paper/www2018_reinforceRec/www2018_reinforceRec.pdf)
## Industry
### CTR prediction
* Representation Learning-Assisted Click-Through Rate Prediction [IJCAI 2019] [[__PDF__]](https://arxiv.org/abs/1906.04365)[Alibaba]

* Deep Session Interest Network for Click-Through Rate Prediction [IJCAI 2019] [[__PDF__]](https://arxiv.org/abs/1905.06482) [Alibaba]

* Deep Spatio-Temporal Neural Networks for Click-Through Rate Prediction] [KDD2019] [[__PDF__]](https://arxiv.org/abs/1906.03776)  [Alibaba]

* Graph Intention Network for Click-through Rate Prediction in Sponsored Search [SIGIR2019] [[__PDF__]](https://dl.acm.org/citation.cfm?id=3331283)[Alibaba]

* Order-aware Embedding Neural Network for CTR Prediction][SIGIR 2019] [[__PDF__]](https://dl.acm.org/citation.cfm?id=3331332) [Huawei]

* Feature Generation by Convolutional Neural Network for Click-Through Rate Prediction [WWW 2019] [[__PDF__]](https://arxiv.org/abs/1904.04447) [Huawei]

* xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems] [KDD2018] [[__PDF__]](https://arxiv.org/pdf/1803.05170.pdf) [Microsoft]
* DeepFM: A Factorization-Machine based Neural Network for CTR Prediction [[IJCAI 2017] [[__PDF__]](https://arxiv.org/abs/1703.04247), [IJCAI 2017] [Huawei]

### Embedding
* Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba [KDD 2018][Alibaba][[__PDF__]](https://arxiv.org/pdf/1803.02349.pdf)
* Real-time Personalization using Embeddings for Search Ranking at Airbnb [KDD 2018] [[__PDF__]](https://astro.temple.edu/~tua95067/kdd2018.pdf)
* Learning and Transferring IDs Representation in E-commerce [Alibaba] [KDD 2018] [[__PDF__]](https://arxiv.org/pdf/1712.08289.pdf)
* Item2Vec-Neural Item Embedding for Collaborative Filtering [Microsoft 2017]][__PDF__]](https://arxiv.org/pdf/1603.04259.pdf)
* Graph Convolutional Neural Networks for Web-Scale Recommender Systems [Pinterest][KDD 2018] [[__PDF__]](https://arxiv.org/pdf/1806.01973)

### Others
* ATRank: An Attention-Based User Behavior Modeling Framework for Recommendation [AAAI 2018] [Alibaba] [[__PDF__]](https://arxiv.org/pdf/1711.06632.pdf)
* Deep Neural Networks for YouTube Recommendations [Youtube] [RecSys 2016] [[__PDF__]](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45530.pdf)
* Multi-Interest Network with Dynamic Routing for Recommendation at Tmall [Alibaba] [[__PDF__]](https://arxiv.org/pdf/1904.08030)
* Latent Cross: Making Use of Context in Recurrent Recommender Systems [WSDM 2018][[__PDF__]](http://delivery.acm.org/10.1145/3160000/3159727/p46-beutel.pdf?ip=159.226.43.46&id=3159727&acc=OA&key=33E289E220520BFB%2ED25FD1BB8C28ADF7%2E4D4702B0C3E38B35%2E5945DC2EABF3343C&__acm__=1568103183_98476c18cb349d52e835c76d85b83253)
* Learning from History and Present: Next-item Recommendation via Discriminatively Exploting Users Behaviors [KDD 2018][[__PDF__]](https://arxiv.org/pdf/1808.01075.pdf)
* Deep Semantic Matching for Amazon Product Search [WSDM 2019][Amazon][[__PDF__]](https://wsdm2019-dapa.github.io/slides/05-YiweiSong.pdf)
* Perceive Your Users in Depth: Learning Universal User Representations from Multiple E-commerce Tasks [KDD2018] [Alibaba] [[__PDF__]](https://arxiv.org/pdf/1805.10727.pdf)


