# Extra Credit Challenge

## Challenge I

$$\begin{equation} \begin{aligned} \frac{\partial J}{\partial v_c} &= -u_o + \frac{\sum_{w \in Vocab}exp(u_w^Tv_c)u_w}{\sum_{w\in Vocab}exp(u_w^Tv_c)} \\ &= -u_o + \sum_{w\in Vocab}\frac{exp(u_w^Tv_c)}{\sum_{w\in Vocab}exp(u_w^Tv_c)}u_w \\ &= -u_o + \sum_{w\in Vocab}\hat{y}_wu_w \\ &= U(\hat{y} - y) \end{aligned} \end{equation} \\ Q.E.D.$$

## Challenge II

$$\begin{equation}\begin{aligned} \frac{\partial J}{\partial u_w} &= \frac{exp(u_w^Tv_c)v_c}{\sum_{w'\in Vocab} exp(u_w'^Tv_c)} \\ &= \hat{y}_wv_c \end{aligned}\end{equation}$$

$$\begin{equation}\begin{aligned} \frac{\partial J}{\partial u_o} &= -v_c + \frac{exp(u_o^Tv_c)v_c}{\sum_{w'\in Vocab} exp(u_w'^Tv_c)} \\ &= (\hat{y}_o - 1)v_c \end{aligned}\end{equation}\\ Q.E.D. $$

