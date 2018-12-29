
#########################################
Weekly Share
#########################################
:Auther: Xuemei Wang
:Date: 2018-12-24

.. contents:: :depth: 1

All Abaut Boosting
=========================================

Vocabularies
-----------------------------------------
Boosting, Bootstrapping, Bootstrap Aggregating, Gradient Boosting,
Gradient Boosting Decision Tree, Gradient Boosting Machine, XGBoost...

Behind the Vocabularies
-----------------------------------------
If we have some/many (weak) learners, how to organize these (weak) learners to make a final string learner?

Dose the final learn take full (more and more?) advantage of weak learners? How?

1. Randomly, disorderedly, averagely?
2. Boosting: orderedly, weightedly, sysmatically

Boosting
=========================================

1. Boosting is a machine learning *ensemble meta-algorithm* for primarily reducing bias, and also variance in supervised learning,
and a family of machine learning algorithms that convert weak lelarnere to strong ones.

2. Algorithms that achieve hypothesis boosting quickly became simply known a "boosting".

3. While boosting is not algorithmically constrained, most boosting algorithms consist of iteratively learning weak classifiers with respect to a distribution and adding them to a final strong classifier.

4. (:math:`\star`) When they are added, they are typically weighted in some way that is usually related to the weak learners' accuracy.

5. (:math:`\star`) After a weak learner is added, the data weights are readjusted, known as "re-weighting".

6. (:math:`\star`) Misclassified input data gain a higher weight and examples that are classified correctly lost weight.

7. (:math:`\star`) Future weak learners focus more on the examples that previous weak learners misclassified.

8. Only algorithms that are provable boosting algorithms in the probably approximately correct learning formulation can accurately be called boosting glgorithms.

9. Other algorithms that are similar in spirit to boosting algorithms are sometimes called "leveraging algorithms"

10. (:math:`\star`) The main variation between many boosting algorithms is their method of weighting training data points and hypothess.

Bootstraping
=========================================
In statistics, boostrapping is any test or metric that relies on random sampling with replacement.

Bootstrap Aggregating
=========================================
1. Boostrap aggregrating, also called bagging, is a machine learning ensemble meta-algorithm designed to improve the stability
and accuracy of machine learning algorithms used in statistical classification and regression.

2. It also reduces variance and helps to avoid overfitting.

AdaBoost (adaptive boosting)
=========================================
AdaBoost, short for Adaptive Boosting, is a machine learning meta-algorithm formulated by Yoav Freund and Robert Schapire.

  * It can be used in conjunction with many other learning algorithms ('weak learner') is combined into a weighted sum that represents the final output of the boosted classifer.
  * AdaBoost is sensitive to noisy data and outliers.
  * In some problems it can be less susceptible to the overfitting problem than other learning algorithms.
  * The individual learners can be weak, but as long as the performance of each one is slightly better than random guessing, the final model can be proven to converge to a strong learner.
  * When used with decision tree learning, information gatherd at each stage of the AdaBoost algorithm about the relaive 'hardness' of each training sample is fed into the tree growing algorithm such that later trees tend to focus on harder-to-classify example.

Before math
-----------------------------------------
Example: Is a person male or female?
Classifiers: 

  1.height
  2.hair length
  3.voice frequency
  4.ratio of lengths of waist vs. brest

Intuition:

  1. all related to (or caused by) gender, but not caused by each other (vs. height & weight)
  2. for classifiers the best the fist: 4 ==> 3 ==> 1 ==> 2 ( greedy, potentially less iterations but generally not sufficient)


How does it work sysmetrically?
-----------------------------------------

Suppose we have:

  * data set :math:`\{(x_1, y_1), ..., (x_N, y_N)\}` where each item :math:`x_i` has an associated class :math:`y_i \in \{-1, 1\}`, and
  * a set of weak classifiers :math:`\{k_1, k_2, ..., k_L\}` each of which outputs a classification :math:`k_j(x_i) \in \{-1, 1\}` for each item.
  * After the :math:`(m - 1)`-th iteration our boosted classifier is a linear combination of the weak classifiers of the form:

.. math::

    C_{(m-1)}(x_i) = \alpha_1k_1(x_i) + ... + \alpha_{m-1}k_{m-1}(x_i)
    C_m(x_i) = C_{(m -1)}(x_i) + \alpha_mk_m(x_i)

Our goal is to decide :math:`\{\alpha_i\}`

Error function:

.. math::

  \begin{eqnarray}
  E &=& \sum_{i=1}^N e^{-y_iC_m(x_i)} \\
  &=& \sum_{i=1}^N w_i^{(m)}e^{-y_i\alpha_mk_m(x_i)}
  \end{eqnarray}

  \begin{eqnarray}
  \alpha_m &=& \frac{1}{2}\ln(\frac{\sum_{y_i = k_m(x_i)}w_i^{(m)}}
  {\sum_{y\neq k_m(x_i)}w_i^{(m)}})\\
  \alpha_m &=& \frac{1}{2}\ln(\frac{1 - \epsilon_m}{\epsilon_m})
  \end{eqnarray}

where :math:`\epsilon_m = \sum_{yi \neq k_m(x_i)} w_i^{(m)} / \sum_{i=1}^N w_i^{(m)}`

Gradient boosting
=========================================

1. Gradient boosting is a machine learning technique for * regression * and classification problem,
   which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees.
2. It, builds the model in a stage-wise fashion like other boosting methods do,
   and t generalizes other boosting methods by allowing optimization of * an arbitrary differentiable loss function * .
3. Like other boosting methods, gradient boosting combines weak "learners" into a single strong learner in an iterative fashion.

Intuition
-----------------------------------------
Iteratively(gradiently) find hypotheses (decide how to organize them)

Math
-----------------------------------------


.. math::
  
  \begin{eqnarray}
  \hat F(x) &=& \sum_{i=1}^M\gamma_ih_i(x) + const. \\
  F_0(x) &=& argmin_{\gamma}\sum_{i=1}^nL(y_i, \gamma),  (just const.) \\
  F_m(x) &=& F_{m-1}(x) + argmin_{h_m \in \mathcal{H}}\Big[\sum_{i=1}^nL(y_i, F_{m-1}(x_i) + h_m(x_i))\Big]
  \end{eqnarray}

where :math:`h_m \in \mathcal{H}` is a base learner function.

Unfortunately, choosing the best function h at each step for an arbitrary loss funtion L is
a computationally infeasible optimization problem in general.
(complex? fundamentally is it automatically the best solution? generally contexed?) 
Therefore, we restrict our approach to a simplified version of the problem.

The idea is to apply a steepest descent step to this minimization problem.
If we considered the continuous case, i.e. where :math:`\mathcal{H}` is the set of arbitrary differentiable functions on R,
we would update the model in accordance with the following equations

.. math::

  \begin{eqnarray}
  F_m(x) &=& F_{m-1}(x) - \gamma_m\sum_{i = 1}^n\bigtriangledown L(y_i, F_{m-1}(x_i)),\\
  \gamma_m &=& argmin_{\gamma}\sum_{i=1}^n L(y_i, F_{m-1}(x_i) - \gamma\bigtriangledown F_{m-1} L(y_i, F_{m-1}(x_i))
  \end{eqnarray}


where the derivatives are taken with respect to the functions :math:`F_i` for :math:`i \in \{1, ..., m\}`.
In the discrete  case however, i.e. when the set :math:`\mathcal{H}` is finite,
we choose the candidate function :math:`h` closest to the gradient to :math:`L` for which the coefficient :math:`\gamma`
may then the calculated with the aid of line search on the avove equations.
Note thatt this approach is a heuristic and therefore doesn't yield an exact solution to the given problem,
but rather an approximation.





