#### [Extension of paper Table II] Comparing feature extractors for image-level prediction on official CBIS split. 
| Model         | $\text{Precision}^{Our}$ | $\text{Recall}^{Our}$    | $\text{F1}^{Our}$        | $\text{AUC}^{Our}$       | $\text{AUC}^{Paper}$     |
| ----------    | -------------------      | -------                  | ---------                | ---------                | -----------              |
| DIB-MG        | $0.52 \pm 0.01$          | $0.56 \pm 0.03$          | $0.54 \pm 0.02$          | $0.64 \pm 0.00$          | n.a.                     |
| DenseNet169   |                          |                          |                          |                          |                          |
| +avgpool      | $0.65 \pm 0.04$          | $0.59 \pm 0.08$          | $0.61 \pm 0.03$          | $0.76 \pm 0.01$          | $0.76 \pm 0.00$          |
| +maxpool      | $0.59 \pm 0.03$          | $0.68 \pm 0.04$          | $0.63 \pm 0.00$          | $0.74 \pm 0.00$          | $0.74 \pm 0.00$          |
| +RGP (k=0.7)  | $0.66 \pm 0.02$          | $0.58 \pm 0.03$          | $0.62 \pm 0.01$          | $0.76 \pm 0.01$          | $0.84 \pm 0.00$          |
| +GGP (k=0.7)  | $\textbf{0.70} \pm 0.02$ | $0.55 \pm 0.05 $         | $0.62 \pm 0.03$          | $0.76 \pm 0.02$          | $0.82 \pm 0.00$          |
| Shen at al.   |                          |                          |                          |                          |                          |
| ResNet34      | $0.62 \pm 0.01$          | $0.71 \pm 0.04$          | $\textbf{0.66} \pm 0.02$ | $0.78 \pm 0.01$          | $0.79 \pm 0.01$          |
| GMIC-ResNet18 | $0.57 \pm 0.06$          | $\textbf{0.80} \pm 0.10$ | $\textbf{0.66} \pm 0.02$ | $\textbf{0.79} \pm 0.02$ | $0.83 \pm 0.00$          |


#### [Extension of paper Table IV] Performance of MIL pooling approaches on CBIS, VinDr, MGM-FV. 
| Model     | Precision                | Recall                   | F1                       | AUC                      |
| --------- | -------------------      | -------                  | ---------                | ---------                | 
<td colspan=5> *CBIS* </td>                                                                                                 
| DIB-MG    | $0.58 \pm 0.01$          | $0.47 \pm 0.05$          | $0.52 \pm 0.03$          | $0.64 \pm 0.01$          |
| ISMeanImg | $0.72 \pm 0.11$          | $0.63 \pm 0.18$          | $0.65 \pm 0.07$          | $0.79 \pm 0.03$          |
| ISMaxImg  | $0.64 \pm 0.06$          | $0.68 \pm 0.07$          | $0.66 \pm 0.05$          | $0.75 \pm 0.05$          |
| ISAttImg  | $\textbf{0.78} \pm 0.08$ | $0.51 \pm 0.17$          | $0.60 \pm 0.08$          | $0.78 \pm 0.04$          |
| ISGattImg | $0.66 \pm 0.06$          | $\textbf{0.71} \pm 0.12$ | $\textbf{0.68} \pm 0.04$ | $0.78 \pm 0.03$          |
| ESMeanImg | $0.68 \pm 0.04$          | $0.63 \pm 0.11$          | $0.65 \pm 0.04$          | $0.76 \pm 0.02$          |
| ESMaxImg  | $0.76 \pm 0.02$          | $0.54 \pm 0.06$          | $0.63 \pm 0.04$          | $0.76 \pm 0.01$          |
| ESAttImg  | $0.72 \pm 0.15$          | $0.68 \pm 0.22$          | $0.67 \pm 0.06$          | $\textbf{0.81} \pm 0.02$ |
| ESGAttImg | $0.77 \pm 0.04$          | $0.58 \pm 0.09$          | $0.66 \pm 0.06$          | $0.78 \pm 0.03$          |
| ESAttSide | $0.70 \pm 0.07$          | $0.68 \pm 0.16$          | $\textbf{0.68} \pm 0.06$ | $0.79 \pm 0.02$          |
|           |                          | *VinDr*                  |                          |                          |   
| DIB-MG    | $\textbf{0.43} \pm 0.13$ | $0.26 \pm 0.02$          | $0.32 \pm 0.03$          | $0.68 \pm 0.02$          | 
| ISMeanImg | $0.39 \pm 0.10$          | $0.57 \pm 0.07$          | $0.45 \pm 0.06$          | $0.81 \pm 0.01$          |
| ISMaxImg  | $0.37 \pm 0.01$          | $0.63 \pm 0.10$          | $0.46 \pm 0.03$          | $0.82 \pm 0.04$          |
| ISAttImg  | $0.34 \pm 0.06$          | $0.61 \pm 0.06$          | $0.43 \pm 0.04$          | $0.81 \pm 0.03$          |
| ISGattImg | $0.29 \pm 0.02$          | $0.64 \pm 0.05$          | $0.40 \pm 0.01$          | $0.82 \pm 0.02$          |
| ESMeanImg | $0.22 \pm 0.05$          | $\textbf{0.73} \pm 0.10$ | $0.34 \pm 0.04$          | $0.80 \pm 0.00$          |
| ESMaxImg  | $0.42 \pm 0.11$          | $0.55 \pm 0.08$          | $0.46 \pm 0.06$          | $0.80 \pm 0.01$          |
| ESAttImg  | $0.34 \pm 0.06$          | $0.63 \pm 0.03$          | $0.44 \pm 0.04$          | $0.81 \pm 0.01$          |
| ESGAttImg | $0.41 \pm 0.08$          | $0.59 \pm 0.07$          | $\textbf{0.48} \pm 0.04$ | $0.82 \pm 0.01$          |
| ESAttSide | $0.37 \pm 0.04$          | $0.67 \pm 0.04$          | $\textbf{0.48} \pm 0.03$ | $\textbf{0.83} \pm 0.02$ |
|           |                          | *MGM-FV*                 |                          |                          |   
| DIB-MG    | $0.41 \pm 0.01$          | $0.33 \pm 0.01$          | $0.37 \pm 0.01$          | $0.71 \pm 0.00$          |
| ISMeanImg | $0.50 \pm 0.11$          | $0.63 \pm 0.08$          | $0.55 \pm 0.04$          | $0.83 \pm 0.01$          |
| ISMaxImg  | $0.46 \pm 0.05$          | $0.63 \pm 0.07$          | $0.53 \pm 0.01$          | $0.81 \pm 0.01$          |
| ISAttImg  | $0.45 \pm 0.01$          | $0.72 \pm 0.03$          | $0.56 \pm 0.02$          | $\textbf{0.85} \pm 0.01$ |
| ISGAttImg | $0.42 \pm 0.02$          | $\textbf{0.73} \pm 0.01$ | $0.53 \pm 0.01$          | $0.84 \pm 0.00$          |
| ESMeanImg | $0.47 \pm 0.01$          | $0.65 \pm 0.04$          | $0.55 \pm 0.02$          | $0.83 \pm 0.03$          |
| ESMaxImg  | $0.42 \pm 0.06$          | $0.69 \pm 0.07$          | $0.52 \pm 0.03$          | $0.82 \pm 0.00$          |
| ESAttImg  | $0.50 \pm 0.05$          | $0.67 \pm 0.01$          | $0.57 \pm 0.03$          | $0.84 \pm 0.01$          |
| ESGAttImg | $0.47 \pm 0.06$          | $0.67 \pm 0.02$          | $0.55 \pm 0.04$          | $0.83 \pm 0.02$          |
| ESAttSide | $\textbf{0.56} \pm 0.03$ | $0.65 \pm 0.04$          | $\textbf{0.60} \pm 0.00$ | $\textbf{0.85} \pm 0.01$ |

#### [Extension of paper Table V] SIL VS. MIL comparison on CBIS, VinDr, MGM-VV.
|  Model   | Pred. Level | Precision                | Recall                   | F1                       | AUC                      |
| ---------| ------------| -------                  | ---------                | ---------                | -------------------------|
|          |             |                          |   *CBIS*                 |                          |                          |
| SILil    | Image       | $\textbf{0.70} \pm 0.08$ | $0.62 \pm 0.07$          | $0.66 \pm 0.03$          | $\textbf{0.79} \pm 0.05$ |
| SILil    | Case        | $\textbf{0.70} \pm 0.09$ | $0.68 \pm 0.06$          | $\textbf{0.68} \pm 0.03$ | n.a.                     |
| SILcl    | Image       | $0.67 \pm 0.05$          | $0.63 \pm 0.08$          | $0.65 \pm 0.02$          | $0.76 \pm 0.00$          |
| SILcl    | Case        | $0.63 \pm 0.05$          | $\textbf{0.72} \pm 0.06$ | $0.67 \pm 0.02$          | n.a.                     |
| ESAttSide | Case       | $\textbf{0.70} \pm 0.07$ | $0.68 \pm 0.16$          | $\textbf{0.68} \pm 0.06$ | $\textbf{0.79} \pm 0.02$ |
|          |             |                          |   *VinDr*                |                          |                          | 
| SILil    | Image       | $0.17 \pm 0.01$          | $0.67 \pm 0.01$          | $0.27 \pm 0.01$          | $\textbf{0.83} \pm 0.01$ |
| SILil    | Case        | $0.18 \pm 0.00$          | $0.83 \pm 0.03$          | $0.30 \pm 0.00$          | n.a.                     |
| SILcl    | Image       | $0.22 \pm 0.05$          | $0.63 \pm 0.09$          | $0.32 \pm 0.05$          | $0.76 \pm 0.00$          |
| SILcl    | Case        | $0.16 \pm 0.03$          | $\textbf{0.85} \pm 0.07$ | $0.26 \pm 0.05$          | n.a.                     |
| ESAttImg | Case        | $\textbf{0.34} \pm 0.06$ | $0.63 \pm 0.03$          | $\textbf{0.44} \pm 0.04$ | $0.81 \pm 0.01$          |
|          |             |                          |   *MGM-VV*               |                          |                          |
| SILcl    | Image       | $0.34 \pm 0.02$          | $0.61 \pm 0.08$          | $0.44 \pm 0.01$          | $0.75 \pm 0.01$          |
| SILcl    | Case        | $0.29 \pm 0.02$          | $\textbf{0.86} \pm 0.06$ | $0.44 \pm 0.01$          | n.a.                     |
| ESAttSide| Case        | $\textbf{0.56} \pm 0.03$ | $0.65 \pm 0.04$          | $\textbf{0.60} \pm 0.00$ | $\textbf{0.85} \pm 0.01$ |
