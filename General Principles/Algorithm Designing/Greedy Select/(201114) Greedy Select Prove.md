# 贪心法、贪心法的数学归纳法证明步骤

以「活动安排问题」为例，描述：

- 贪心法的设计要素；及
- 证明一个贪心法能得到最优解的归纳法证明步骤 (证真)。

## 范例问题描述

输入：$$S={1,2,...,n}$$ 为 $$n$$ 项活动的集合，$$s_i, f_i$$ 分别是活动 $$i$$ 的开始及结束时间。如果 $$s_i \ge f_j$$ 或 $$s_j \ge f_i$$，即要么活动 $$i$$ 在活动 $$j$$ 前面进行，要么活动 $$i$$ 在活动 $$j$$ 后面进行，就说活动 $$i$$ 与活动 $$j$$ 相容。

求：**最大的两两相容的活动集合 $$A$$**，即求能安排最多活动、且活动间两两相容的活动选择方式。

## 贪心法 (求最优解) 的设计要素

- 求解属于**多步判断过程**，**最后一步**判断的结果序列对应原问题最优解。
- 每一步**依据某种「短视」的选择策略**，从待选集合中挑选「能导致最优解」的下一元素。
- 可能有多种贪心解法，不一定能求出最优解。要证明一个贪心解法能求出最优解，必须求证。
- 贪心法求最优解证真，一般用数学归纳法 (第一 / 第二数学归纳法)。
- 贪心法求最优解证伪，可以举反例。

## 问题的一种贪心解法

将 $$n$$ 个活动按照其结束时间 $$f_i$$ 从前到后排序，排序后的活动序列亦按 $$S={1,2,...,n}$$ 编号。

第一次先选 1 号活动，然后接下来的每一步，从 $$S$$ 中按顺序选出下一个相容的活动，直到 $$S$$ 中所有活动都被检查过一遍。

这一贪心解法能得到「活动安排问题」的最优解。证明如下：

## 证明能得到最优解基本步骤

1. 【**问题转化**】证明该贪心解法能得到「活动安排问题」的最优解，即考察如下问题：该算法执行到第 $$k$$ 步时，选择了 $$k$$ 个活动：$$1, ..., i_k$$，则**存在最优解 $$A$$ 包含这 $$k$$ 个活动** (即，该算法执行的每一步的结果都是最优解的一部分)。

2. 【**归纳基础**】证明第 1 步时选择的活动**可以**在最优解中。

   - 算法第 1 步选择的是活动 1。下面证明活动 1 在最优解 $$A$$ 中。假如活动 1 不在最优解 $$A$$ 中，即最优解可表述为
     $$
     A=i_j, ..., i_f
     $$
     也就是最优解的第一个活动为 $$i_j$$。由于**活动 1 的结束时间是所有活动中最前的**，因此 $$f_1 \le f_{i_j}$$。这样，就将 $$A$$ 中的 $$i_j$$ 换成 $$1$$，得到 $$A'$$：
     $$
     A'=(A-\{i_j\}) \cup \{1\} = 1,...,i_f
     $$

   - **由于 $$f_1 \le f_{i_j}$$，因此 $$A'$$ 中的活动也是相容的**，而且活动个数与 $$A$$ 一致。**因此，$$A'$$ 也是一个最优解**。

   - 所以，第 1 步时选择的活动 1 肯定可以在最优解中。

3. 【**归纳步骤**】证明：若第 $$k$$ 步选择的活动 $$i_k$$ 在最优解中，则第 $$k+1$$ 步选择的活动 $$i_{k+1}$$ 亦在最优解中。

   - 归纳假设「第 $$k$$ 步选择的活动 $$i_k$$ 在最优解中」可以表述为：第 $$k$$ 步已经选择的活动 $$S_k=1, i_2 ..., i_k$$ 都在 $$A$$ 中。

   - 第 $$k+1$$ 步时，选择只能在**待选活动集合 $$S'$$** 中选取。所谓待选活动集合，即原集合 $$S$$ 刨除了**已判断为冲突**的活动、**已选择的活动后**剩下的集合。这样，待选活动集合 $$S'$$ 中的元素，有的会和最终 $$A$$ 集合冲突，有的会被选入 $$A$$ 。

   - 下面证明，**最优解 $$A$$ 是 $$S_k$$ 和 $$S'$$ 的一个最优解 $$B$$ 的并 (即 $$A=S_k \cup B$$)**。

     如果 $$B$$ 不是 $$S'$$ (子问题) 的最优解，且 $$S'$$ 的子问题的最优解是 $$B'$$，则 $$|B'| > |B|$$ 。将 $$A = S_k\cup B$$ 右边的 $$B$$ 换成 $$B'$$，使 $$A’=S_k \cup B'$$，则 $$|A'| > |A|$$，因此 $$A$$ 不是最优解，此为矛盾。

     因此，$$A=S_k \cup B$$。

   - 下面证明，**算法第 $$k+1$$ 步选择的元素 $$i_{k+1}$$ 在 $$S'$$ 的一个最优解 $$B^*$$ 中**。

     由于 $$S'$$ 已经刨除已知冲突的活动，因此，$$S'$$ 的第一个元素就是这步要选的 $$i_{k+1}$$。

     根据归纳假设，可以知道： **$$i_{k+1}$$ 必在 $$S'$$ 的一个最优解当中**，得证。

   - **不同最优解，因其数量一致所以为等效**。

     设第 $$k+1$$ 步选择的元素 $$i_{k+1}$$ 在 $$S'$$ 的一个最优解 $$B^*$$ 中，因此
     $$
     A^*=S_k\cup B^*
     $$
     是一个和 $$A=S_k \cup B$$ 元素数量相等的一个母问题的最优解 (将 $$S'$$ 的一个最优解 $$B$$ 换成了另一个最优解 $$B^*$$)。

     因此，得证第 $$k+1$$ 步选择的活动 $$i_{k+1}$$ 在最优解 $$A^*$$ 中。即这种在第 $$k+1$$ 步贪婪选择活动 $$i_{k+1}$$ 的悬法，能导致产生最优解。

4. 综上所述，在某个 $$f$$ $$(f \le n)$$ 处算法可以选出最优解 $$A=i_j, ..., i_f$$。
