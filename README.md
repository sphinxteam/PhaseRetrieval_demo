# Phase retrieval in high dimensions: Statistical and computational phase transitions

We provide notebooks and codes for the paper : "Phase retrieval in high dimensions: Statistical and computational phase transitions". In particular, see:
- The notebook "example" containing an example code for the state evolution and G-VAMP algorithms in noiseless phase retrieval with a complex gaussian matrix.
- The different notebooks containing the codes for the plots of the figures in the paper.
- The run files that were used to generate the data shown in the figures, and the associated data files.

 ## Abstract
We consider the \emph{phase retrieval} problem of reconstructing a n-dimensional real or complex signal X* from m (possibly noisy) observations <img src="https://bit.ly/30pUtcW" align="center" border="0" alt="$Y_\mu = | \sum_{i=1}^n \Phi_{\mu i} X^{\star}_i/\sqrt{n}|$" width="174" height="26" />
       , for a large class of correlated real and complex random sensing matrices Phi, in a high-dimensional setting where m,n go to infinity while alpha = m/n = O(1).
        First, we derive sharp asymptotics for the lowest possible estimation error achievable statistically 
        and we unveil the existence of sharp phase transitions for the weak- and full-recovery thresholds as a function of the singular values of the sensing matrix.. 
        This is achieved by providing a rigorous proof of a result first obtained by the replica method from statistical mechanics.   
        In particular, the information-theoretic transition to perfect recovery for full-rank matrices appears at alpha = 1 (real case) and alpha = 2 (complex case).                
        Secondly, we analyze the performance of the best-known polynomial time algorithm for this problem --- approximate message-passing--- establishing the existence of a statistical-to-algorithmic gap depending, again, on the spectral properties of the sensing matrix       
        Our work  provides an extensive classification of the statistical and algorithmic thresholds in high-dimensional phase retrieval for a broad class of random matrices.
