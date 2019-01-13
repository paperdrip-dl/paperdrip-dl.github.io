---
layout: post
title:  "Distillation - Survey Top"
date:   2018-12-22
author: maya
categories: Distillation
tags:	training distillation transfer_learning
cover:  "https://cdn-images-1.medium.com/max/800/1*tTEDSbllMhT0o2NKOIcjog.png"
---

Distillation に興味があるので，関係する論文を調べていきます．

論文を紹介する前に，前提知識(背景)を自分なりにすこしだけまとめておきます．
技術の全容をよりわかりやすく丁寧に俯瞰したい方は[こちら][distill-jp]の素晴らしい記事を読むと良いです．

(ヘッダ画像ソース: [Knowledge Distillation by Ujjwal Upadhyay - Mediam][distill-eng])

## Distillation について 
機械学習，特に深層学習における*Distillation(蒸留)* とは，ざっくり言うと**巨大で複雑なモデルで獲得した知識を使って，より小さなモデルを高精度に学習する**技術です．

なぜ小さいモデルで高精度に学習がしたいかといえば，推論環境ではリアルタイム性や省リソース性が求められているからです．
GPUなどを潤沢に使用できることが学習フェーズとは対照的に，学習モデルを実際に利用する推論フェーズではコストやサイズの観点から，より簡易なマシン環境でタスクを捌きたいという要求があります (e.g., 監視カメラに搭載された組み込み推論チップ, TensorRTとか)．
当然ですが精度の高い巨大なモデルはこのような簡易な環境に搭載することは難しく，載せられたとしても今度はスループットが問題になってきます．パラメータ数が多ければ自明に演算回数も増えるからです．

では，最初から小さいモデルで学習を行えばよいのでしょうか？巨大なモデルと同じような精度が達成できるならばそれも良いでしょう．ただし，一般的にはネットワークパラメータが少ないほどモデルの推論精度が下がることが知られています．

![](https://cdn-images-1.medium.com/max/1200/1*kfpO_fJ4bc92sffY4bxnSA.jpeg)
*ネットワークパラメータ数と精度の関係([Neural Network Architectures][nna]より)*
{: style="text-align: center"}

Distillation はこのような**モデルサイズと精度の両立**という推論環境特有の要求にマッチしている手法と言えます．
基本的には，まず巨大で複雑なネットワークを学習し，学習済みモデル(Teacher)を作ります．
その後，Teacher モデルの出力(予測確率や特徴量)を小さいモデル(Student)を学習するときの損失に組み込むことで，Teacher が持つ知識をStudent に教えてあげることができます．

![](https://cdn-images-1.medium.com/max/800/1*U79yXdHqjbRSidDBwyD5MA.png)
*Distillation のしくみ ([Knowledge Distillation - Mediam][distill-eng]より)*
{: style="text-align: center"}

Teacher のドメイン知識をStudent に転移させているわけなので，Distillation は一種の**転移学習**とみなすことができます．(転移学習についての厳密な定義をこの記事では述べません．機会があれば解説に挑戦したい．)
このときの知識の転移をどのようにして行うかが Distillation に関係する各論文のアイデア(オリジナリティ)になっているはずです．

## サーベイの対象
このブログでは以下の論文についてのサーベイを行い，それなりに時間はかかるかもしれませんが一つずつできるだけ丁寧にまとめていきたいと思います．

* [Distilling Knowledge in a Neural Network][survey1] [[1][paper1]]
* [FitNets: Hint for Thin Deep Nets][survey2] [[2][paper2]]
* [Deep Mutual Learning][survey3] [[3][paper3]]
* [Knowledge Concentration: Learning 100K Object Classifier in a Single CNN][survey4] [[4][paper4]]

以下，準備中
* Data Distillation [[5][paper5]]
* Born-Again Neural Networks [[6][paper6]]
* MEAL: Multi-Model Ensemble via Adversarial Learning [[7][paper7]]
* A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning [[8][paper8]]
* Progressive Blockwise Knowledge Distillation for Neural Network Acceleration [[9][paper9]]
* Knowledge Distillation by On-the-Fly Native Ensemble [[10][paper10]]
* Dataset Distillation [[11][paper11]]
* Adversarial Distillation of Bayesian Neural Network Posteriors [[12][paper12]]
* Teaching Semi-Supervised Classifier via Generalized Distillation [[13][paper13]]
* Learning from Noisy Labels with Distillation [[14][paper14]]
* Cross Modal Distillation for Supervision Transfer [[15][paper15]]
* Dropout Distillation [[16][paper16]]
* Self-supervised Knowledge Distillation Using Singular Value Decomposition [[17][paper17]]

(順次更新予定)

オーソドックスな手法は[Code Craft House][distill-jp]様がまとめてくださっているので，私は新しめの論文を中心にサーベイしたいと思います．
また，特定タスク向けのチューニングにはあまり興味が無いのでなるべくタスク依存でない手法を選んでいます．
面白そうな論文があったら追加していきたいと思います．

[distill-jp]: http://codecrafthouse.jp/p/2018/01/knowledge-distillation/
[distill-eng]: https://medium.com/neural-machines/knowledge-distillation-dc241d7c2322
[nna]: https://towardsdatascience.com/neural-network-architectures-156e5bad51ba
[survey1]: https://paperdrip-dl.github.io/distillation/2018/12/23/Distillating-Knowledge-in-Neural-Networks.html
[survey2]: https://paperdrip-dl.github.io/distillation/2018/12/25/FitNets.html
[survey3]: https://paperdrip-dl.github.io/distillation/2019/01/02/Deep_Mutual_Learning.html
[survey4]: https://paperdrip-dl.github.io/distillation/2019/01/13/Knowledge-Concentration.html
[paper1]: https://arxiv.org/abs/1503.02531
[paper2]: https://arxiv.org/abs/1412.6550
[paper3]: https://arxiv.org/abs/1706.00384
[paper4]: https://arxiv.org/abs/1711.07607
[paper5]: https://arxiv.org/abs/1712.04440
[paper6]: https://arxiv.org/abs/1805.04770
[paper7]: https://arxiv.org/abs/1812.02425
[paper8]: http://openaccess.thecvf.com/content_cvpr_2017/papers/Yim_A_Gift_From_CVPR_2017_paper.pdf
[paper9]: https://www.ijcai.org/proceedings/2018/0384.pdf
[paper10]: https://arxiv.org/abs/1806.04606
[paper11]: https://arxiv.org/abs/1811.10959
[paper12]: https://arxiv.org/abs/1806.10317
[paper13]: https://www.ijcai.org/proceedings/2018/0298.pdf
[paper14]: https://arxiv.org/abs/1703.02391
[paper15]: https://arxiv.org/abs/1507.00448
[paper16]: http://proceedings.mlr.press/v48/bulo16.pdf
[paper17]: https://arxiv.org/abs/1807.06819




