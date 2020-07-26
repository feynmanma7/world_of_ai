# Traditional Text Classification Methods



# Naive Bayes

朴素贝叶斯是一个生成式模型，对训练数据$X_{M*N}$与类别$Y$的联合概率建模$p(X, Y)$。

训练数据$X$一共有$M$条样本，每一条样本维度是$N$；$y$的取值范围是$k\in \{1, 2, ..., K\}$。

总训练样本是$\{(X^{(1)}, Y^{(1)}), (X^{(2)}, Y^{(2)}), ..., (X^{(M)}, Y^{(M)})\}$。用上标表示样本，下标表示样本的特征（因为后面要频繁使用样本特征，下标用Latex语法写起来更轻松一点）。

对测试样本$(x, y)$，

> $y^* = \arg\max_{C_k}p(x, y=C_k)$

令，

（1）$p(y=C_k)=\theta_k, k \in \{1, 2, ..., K\}$，表示$y$是第$k$类的概率，则$p(y)=\prod_{k=1}^K \theta_k^{\mathbb{I}(y=C_k)}$。

（2）$p(X_n=a_{nl}|y=C_k)=\theta_{knl}=\prod_{k=1}^K \prod_n^N \prod_{l=1}^{V_n} \theta_{knl}^{\mathbb{I}(X_n=a_{nl})}$，表示在$y=C_k$时，$X$的第$n$个特征取第$l$个值的概率。假设$x$的第$n$维特征有$V_n$个取值，每一个取值是$a_{nl}$。

假设给定$y=C_k$， $X_n$之间互相**条件独立**，$p(X_n|y=C_k)$。这个简单的假设是非常简单（**Naive**）的，这也是模型叫**朴素**贝叶斯的原因。

模型参数的似然函数是，

$L(\theta|X,Y)=f(X,Y|\theta)=\prod_m p(X^{(m)},Y^{(m)}=C_k)=\prod_m p(X^{(m)}|Y^{(m)}=C_k)p(Y^{(m)}=C_k)$

> $=\prod_{m}p(Y^{(m)}=C_k)\prod_{n}p(X_n^{(m)}|Y^{(m)}=C_k)$
>
> $=\prod_m (\prod_k \theta_k^{\mathbb{I}(Y^{(m)}=C_k)})$$(\prod_n \prod_k \prod_n \prod_l \theta_{knl}^{\mathbb{I}(X_n^{(m)} = a_{nl})})$
>
> $=(\prod_k \theta_k^{N_k})$$(\prod_{k} \prod_{n} \theta_{knl}^{N_{knl}})$

> $N_k=\sum_m \mathbb{I}(Y^{(m)}=C_k)$，表示所有$M$个样本中，属于第$k$个类别的样本个数。
>
> $N_{knl}=\sum_m \sum_n \sum_l \mathbb{I}(X_n^{(m)}=a_{nl})$，表示所有$M$个样本中，属于第$k$个类别，且第$n$维特征取值为$a_{nl}$的个数。

$log$似然是，

> $l(\theta|X,Y)=\log L(\theta|X, Y)=\sum_k N_k \log \theta_k$$+\sum_k \sum_n N_{knl} \log \theta_{knl}$  

加上两个约束，

> $\sum_k \theta_k = 1$
>
> $\sum_l^{a_{nl}} \theta_{knl}=1$

引入拉格朗日乘子，计算得到，

> $\theta_k = \frac{N_k}{M}$
>
> $\theta_{knl}=\frac{N_{knl}}{N_k}$



对于测试样本$(X, y)$，

> $y^* = \arg\max_{C_k} p(y = C_k | X) = \arg\max_{C_k} \frac{p(X|y=C_k)p(y=C_k)}{p(X)}=\arg\max_{C_k} p(X|y=C_k)p(y=C_k)$



scikit-learn有多种实现方式。

### 1. BernoulliNB

假设当$y$是第$k$个类时，一条训练样本$X$的第$j$个特征$X_j$的概率是，

> $p(X_n=a_{nl}|y={C_k})= X_n * p(X_{nl}=1|y={C_k}) + (1-X_n) * (1 - p(X_n=1|y={C_k})),k\in\{1,2,...,K\}$

### 2. MultinomialNB

> $p(X_m=|y=k)$



# SVM



# GBDT

# KNN

