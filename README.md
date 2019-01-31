# RouaultDayanFleming

This repository contains analysis code for the following paper:
Rouault M., Dayan P. & Fleming S. M. Forming global estimates of self-performance from local confidence. Nature Communications (2019)

Script and data files are included in the repository to enable replication of data analyses and rapid generation of the figures in the paper.

The folder DATA contains anonymised behavioral data files for each of the three experiments of the paper, providing the summarised individual data for all plots and statistics: 

•	Exp1.mat

•	Exp2.mat

•	Exp3.mat

It also contains a file (perf_data_for_jasp.csv) allowing replication of Bayesian t-tests under JASP (https://jasp-stats.org/).
The folder SCRIPTS contains three main scripts to enable replication of statistical analyses and rapid generation of the figures for each of the three experiments in the paper:

•	BehaviorGroupExp1.m

•	BehaviorGroupExp2.m

•	BehaviorGroupExp3.m

The folder SCRIPTS also a script running the hierarchical learning model simulations (Group_simulations.m), and a number of helper scripts, for instance for ANOVA and regressions.

To measure metacognition, we make use of previous work. Metacognitive efficiency can be measured by analysing the correspondence between accuracy and confidence, for instance using a signal detection theoretic metric, meta-d' (http://www.columbia.edu/~bsm2105/type2sdt/) (Maniscalco & Lau, 2012). Metacognitive efficiency can also be estimated hierarchically (https://github.com/metacoglab/HMeta-d) (Fleming, 2017).

The folder SCRIPTS also contains a file generating the meta-d’ recovery for hierarchical vs. MLE fit (metacog_recovery.m).

License.

This code is being released with a permissive open-source license. You should feel free to use or adapt the utility code as long as you follow the terms of the license, which are enumerated below. If you make use of or build on the analyses, we would appreciate that you cite the paper.
Copyright (c) 2018, Marion Rouault
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
