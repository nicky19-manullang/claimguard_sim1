# Laporan mesin deteksi hibrida (tahap 5)

Semua angka dari split **test**, anggaran review 5% klaim teratas, CI 95% bootstrap per faskes. Komponen di-fit pada latih, bobot hibrida dipelajari pada val (label biner anomali), test hanya untuk pelaporan. `rec_*` = recall per modus pada anggaran 5%. Varian bukan dipilih berdasarkan test: seluruh daftar ablation dilaporkan.

Mode `retro` = batch retrospektif; `asof` = hanya membandingkan dengan klaim yang sudah diajukan lebih dulu.


## Skema `faskes`, mode `retro`

| varian | PR-AUC | CI95 | Precision@5% | Recall@5% | FPR | Review 80% | rec_repeat | rec_claim | rec_inflated | rec_upcoding |
|---|---|---|---|---|---|---|---|---|---|---|
| Aturan repeat | 0.266 | [0.201, 0.385] | 0.124 | 0.253 | 0.045 | 0.745 | 1.000 | 0.000 | 0.000 | 0.000 |
| Similarity cloning | 0.048 | [0.030, 0.098] | 0.074 | 0.152 | 0.048 | 0.955 | 0.089 | 0.500 | 0.000 | 0.037 |
| Residual inflated | 0.327 | [0.173, 0.482] | 0.181 | 0.371 | 0.042 | 0.565 | 0.067 | 0.068 | 0.968 | 0.000 |
| Kejutan severity | 0.122 | [0.080, 0.169] | 0.135 | 0.275 | 0.044 | 0.695 | 0.044 | 0.341 | 0.081 | 1.000 |
| Isolation Forest | 0.277 | [0.241, 0.318] | 0.220 | 0.449 | 0.040 | 0.155 | 0.289 | 0.318 | 0.548 | 0.704 |
| ECOD | 0.302 | [0.277, 0.355] | 0.242 | 0.494 | 0.039 | 0.123 | 0.422 | 0.500 | 0.468 | 0.667 |
| COPOD | 0.299 | [0.273, 0.331] | 0.217 | 0.444 | 0.040 | 0.136 | 0.156 | 0.295 | 0.661 | 0.667 |
| Ensemble tak terawasi | 0.297 | [0.274, 0.338] | 0.236 | 0.483 | 0.039 | 0.118 | 0.311 | 0.432 | 0.565 | 0.667 |
| Aturan, maks | 0.564 | [0.455, 0.650] | 0.332 | 0.680 | 0.034 | 0.105 | 1.000 | 0.318 | 0.806 | 0.444 |
| Aturan, bobot val | 0.495 | [0.431, 0.562] | 0.316 | 0.646 | 0.035 | 0.125 | 1.000 | 0.182 | 0.823 | 0.407 |
| Hibrida, maks | 0.602 | [0.506, 0.678] | 0.360 | 0.736 | 0.033 | 0.081 | 1.000 | 0.364 | 0.806 | 0.741 |
| Hibrida, bobot val | 0.612 | [0.544, 0.675] | 0.349 | 0.713 | 0.033 | 0.087 | 1.000 | 0.341 | 0.790 | 0.667 |
| Hibrida, dipilih (CV val) | 0.602 | [0.506, 0.678] | 0.360 | 0.736 | 0.033 | 0.081 | 1.000 | 0.364 | 0.806 | 0.741 |

Bobot hibrida (bobot val): {"Aturan repeat": 0.701, "Similarity cloning": 0.165, "Residual inflated": 0.417, "Kejutan severity": 0.252, "Ensemble tak terawasi": 0.707}, intersep -7.146

Bobot aturan saja (bobot val): {"Aturan repeat": 0.815, "Similarity cloning": 0.196, "Residual inflated": 0.475, "Kejutan severity": 0.539}, intersep -7.025

Pemilihan penggabung (CV per faskes pada val, average precision): bobot val=0.468, maks=0.616 -> dipilih **maks**.

Recall tingkat pasangan pada anggaran 5% (repeat/cloning terdeteksi jika salinan atau klaim sumbernya ditandai): claim_cloning=0.386, repeat_billing=1.000

Tingkat ditandai pada hard negatives (hibrida terpilih; makin rendah makin baik): legit_atypical_coding=0.200, legit_high_cost=0.062, legit_recurring=0.012, legit_revisit=0.127, legit_similar=0.071

Recall hibrida per modus x kesulitan (anggaran 5%):

| modus | easy | medium | hard |
|---|---|---|---|
| claim_cloning | 0.800 | 0.176 | 0.083 |
| inflated_bills | 1.000 | 0.870 | 0.357 |
| repeat_billing | 1.000 | 1.000 | 1.000 |
| upcoding | 1.000 | 0.875 | 0.333 |


## Skema `faskes`, mode `asof`

| varian | PR-AUC | CI95 | Precision@5% | Recall@5% | FPR | Review 80% | rec_repeat | rec_claim | rec_inflated | rec_upcoding |
|---|---|---|---|---|---|---|---|---|---|---|
| Aturan repeat | 0.139 | [0.083, 0.282] | 0.102 | 0.208 | 0.046 | 0.772 | 0.822 | 0.000 | 0.000 | 0.000 |
| Similarity cloning | 0.053 | [0.037, 0.114] | 0.071 | 0.146 | 0.048 | 0.896 | 0.044 | 0.500 | 0.016 | 0.037 |
| Residual inflated | 0.325 | [0.173, 0.481] | 0.181 | 0.371 | 0.042 | 0.565 | 0.067 | 0.091 | 0.952 | 0.000 |
| Kejutan severity | 0.122 | [0.080, 0.169] | 0.135 | 0.275 | 0.044 | 0.695 | 0.044 | 0.341 | 0.081 | 1.000 |
| Isolation Forest | 0.218 | [0.184, 0.262] | 0.187 | 0.382 | 0.042 | 0.215 | 0.089 | 0.295 | 0.516 | 0.704 |
| ECOD | 0.218 | [0.187, 0.279] | 0.192 | 0.393 | 0.041 | 0.189 | 0.133 | 0.523 | 0.339 | 0.741 |
| COPOD | 0.236 | [0.192, 0.294] | 0.201 | 0.410 | 0.041 | 0.175 | 0.133 | 0.250 | 0.581 | 0.741 |
| Ensemble tak terawasi | 0.228 | [0.191, 0.290] | 0.203 | 0.416 | 0.041 | 0.187 | 0.111 | 0.364 | 0.532 | 0.741 |
| Aturan, maks | 0.483 | [0.358, 0.591] | 0.283 | 0.579 | 0.037 | 0.111 | 0.467 | 0.364 | 0.839 | 0.519 |
| Aturan, bobot val | 0.382 | [0.231, 0.511] | 0.277 | 0.567 | 0.037 | 0.167 | 0.644 | 0.114 | 0.855 | 0.519 |
| Hibrida, maks | 0.504 | [0.385, 0.609] | 0.299 | 0.612 | 0.036 | 0.109 | 0.467 | 0.386 | 0.806 | 0.778 |
| Hibrida, bobot val | 0.414 | [0.286, 0.530] | 0.286 | 0.584 | 0.037 | 0.114 | 0.511 | 0.318 | 0.774 | 0.704 |
| Hibrida, dipilih (CV val) | 0.504 | [0.385, 0.609] | 0.299 | 0.612 | 0.036 | 0.109 | 0.467 | 0.386 | 0.806 | 0.778 |

Bobot hibrida (bobot val): {"Aturan repeat": 0.459, "Similarity cloning": 0.073, "Residual inflated": 0.414, "Kejutan severity": 0.318, "Ensemble tak terawasi": 0.582}, intersep -6.366

Bobot aturan saja (bobot val): {"Aturan repeat": 0.61, "Similarity cloning": 0.1, "Residual inflated": 0.447, "Kejutan severity": 0.534}, intersep -6.33

Pemilihan penggabung (CV per faskes pada val, average precision): bobot val=0.276, maks=0.433 -> dipilih **maks**.

Recall tingkat pasangan pada anggaran 5% (repeat/cloning terdeteksi jika salinan atau klaim sumbernya ditandai): claim_cloning=0.523, repeat_billing=0.622

Tingkat ditandai pada hard negatives (hibrida terpilih; makin rendah makin baik): legit_atypical_coding=0.143, legit_high_cost=0.094, legit_recurring=0.150, legit_revisit=0.006, legit_similar=0.071

Recall hibrida per modus x kesulitan (anggaran 5%):

| modus | easy | medium | hard |
|---|---|---|---|
| claim_cloning | 0.867 | 0.235 | 0.000 |
| inflated_bills | 1.000 | 0.870 | 0.357 |
| repeat_billing | 1.000 | 0.150 | 0.125 |
| upcoding | 1.000 | 0.875 | 0.444 |


## Skema `waktu`, mode `retro`

| varian | PR-AUC | CI95 | Precision@5% | Recall@5% | FPR | Review 80% | rec_repeat | rec_claim | rec_inflated | rec_upcoding |
|---|---|---|---|---|---|---|---|---|---|---|
| Aturan repeat | 0.309 | [0.209, 0.433] | 0.165 | 0.300 | 0.043 | 0.791 | 1.000 | 0.000 | 0.000 | 0.000 |
| Similarity cloning | 0.051 | [0.036, 0.073] | 0.083 | 0.152 | 0.047 | 0.926 | 0.078 | 0.640 | 0.000 | 0.019 |
| Residual inflated | 0.267 | [0.162, 0.346] | 0.184 | 0.335 | 0.042 | 0.684 | 0.078 | 0.120 | 0.947 | 0.037 |
| Kejutan severity | 0.147 | [0.068, 0.254] | 0.145 | 0.265 | 0.044 | 0.712 | 0.026 | 0.240 | 0.026 | 0.963 |
| Isolation Forest | 0.226 | [0.145, 0.332] | 0.160 | 0.292 | 0.043 | 0.214 | 0.104 | 0.280 | 0.211 | 0.685 |
| ECOD | 0.264 | [0.189, 0.355] | 0.205 | 0.374 | 0.041 | 0.150 | 0.234 | 0.640 | 0.132 | 0.667 |
| COPOD | 0.265 | [0.184, 0.370] | 0.207 | 0.377 | 0.041 | 0.158 | 0.390 | 0.240 | 0.303 | 0.593 |
| Ensemble tak terawasi | 0.249 | [0.168, 0.347] | 0.188 | 0.342 | 0.042 | 0.148 | 0.195 | 0.360 | 0.237 | 0.685 |
| Aturan, maks | 0.569 | [0.472, 0.659] | 0.385 | 0.700 | 0.032 | 0.089 | 0.974 | 0.360 | 0.776 | 0.519 |
| Aturan, bobot val | 0.505 | [0.387, 0.608] | 0.363 | 0.661 | 0.033 | 0.108 | 0.844 | 0.260 | 0.855 | 0.500 |
| Hibrida, maks | 0.563 | [0.477, 0.638] | 0.382 | 0.696 | 0.032 | 0.099 | 0.974 | 0.380 | 0.697 | 0.593 |
| Hibrida, bobot val | 0.559 | [0.463, 0.642] | 0.395 | 0.720 | 0.031 | 0.077 | 0.857 | 0.480 | 0.776 | 0.667 |
| Hibrida, dipilih (CV val) | 0.563 | [0.477, 0.638] | 0.382 | 0.696 | 0.032 | 0.099 | 0.974 | 0.380 | 0.697 | 0.593 |

Bobot hibrida (bobot val): {"Aturan repeat": 0.511, "Similarity cloning": 0.189, "Residual inflated": 0.559, "Kejutan severity": 0.304, "Ensemble tak terawasi": 0.655}, intersep -7.041

Bobot aturan saja (bobot val): {"Aturan repeat": 0.653, "Similarity cloning": 0.181, "Residual inflated": 0.615, "Kejutan severity": 0.606}, intersep -6.95

Pemilihan penggabung (CV per faskes pada val, average precision): bobot val=0.481, maks=0.526 -> dipilih **maks**.

Recall tingkat pasangan pada anggaran 5% (repeat/cloning terdeteksi jika salinan atau klaim sumbernya ditandai): claim_cloning=0.400, repeat_billing=0.974

Tingkat ditandai pada hard negatives (hibrida terpilih; makin rendah makin baik): legit_atypical_coding=0.221, legit_high_cost=0.108, legit_recurring=0.051, legit_revisit=0.060, legit_similar=0.030

Recall hibrida per modus x kesulitan (anggaran 5%):

| modus | easy | medium | hard |
|---|---|---|---|
| claim_cloning | 0.684 | 0.231 | 0.000 |
| inflated_bills | 1.000 | 0.722 | 0.278 |
| repeat_billing | 1.000 | 1.000 | 0.882 |
| upcoding | 1.000 | 0.474 | 0.143 |


## Skema `waktu`, mode `asof`

| varian | PR-AUC | CI95 | Precision@5% | Recall@5% | FPR | Review 80% | rec_repeat | rec_claim | rec_inflated | rec_upcoding |
|---|---|---|---|---|---|---|---|---|---|---|
| Aturan repeat | 0.171 | [0.108, 0.244] | 0.132 | 0.241 | 0.045 | 0.801 | 0.805 | 0.000 | 0.000 | 0.000 |
| Similarity cloning | 0.047 | [0.034, 0.069] | 0.068 | 0.125 | 0.048 | 0.926 | 0.065 | 0.500 | 0.000 | 0.037 |
| Residual inflated | 0.266 | [0.162, 0.344] | 0.184 | 0.335 | 0.042 | 0.688 | 0.078 | 0.120 | 0.947 | 0.037 |
| Kejutan severity | 0.147 | [0.068, 0.254] | 0.145 | 0.265 | 0.044 | 0.712 | 0.026 | 0.240 | 0.026 | 0.963 |
| Isolation Forest | 0.194 | [0.124, 0.292] | 0.147 | 0.268 | 0.044 | 0.313 | 0.065 | 0.340 | 0.145 | 0.667 |
| ECOD | 0.185 | [0.121, 0.274] | 0.154 | 0.280 | 0.044 | 0.278 | 0.117 | 0.420 | 0.105 | 0.630 |
| COPOD | 0.208 | [0.139, 0.302] | 0.158 | 0.288 | 0.043 | 0.272 | 0.169 | 0.320 | 0.118 | 0.667 |
| Ensemble tak terawasi | 0.201 | [0.134, 0.295] | 0.147 | 0.268 | 0.044 | 0.276 | 0.091 | 0.380 | 0.105 | 0.648 |
| Aturan, maks | 0.448 | [0.362, 0.522] | 0.291 | 0.529 | 0.036 | 0.139 | 0.481 | 0.400 | 0.697 | 0.481 |
| Aturan, bobot val | 0.395 | [0.267, 0.509] | 0.301 | 0.549 | 0.036 | 0.212 | 0.481 | 0.200 | 0.882 | 0.500 |
| Hibrida, maks | 0.456 | [0.370, 0.533] | 0.295 | 0.537 | 0.036 | 0.142 | 0.455 | 0.400 | 0.697 | 0.556 |
| Hibrida, bobot val | 0.403 | [0.279, 0.531] | 0.316 | 0.576 | 0.035 | 0.186 | 0.390 | 0.320 | 0.829 | 0.722 |
| Hibrida, dipilih (CV val) | 0.456 | [0.370, 0.533] | 0.295 | 0.537 | 0.036 | 0.142 | 0.455 | 0.400 | 0.697 | 0.556 |

Bobot hibrida (bobot val): {"Aturan repeat": 0.317, "Similarity cloning": 0.056, "Residual inflated": 0.571, "Kejutan severity": 0.317, "Ensemble tak terawasi": 0.566}, intersep -6.388

Bobot aturan saja (bobot val): {"Aturan repeat": 0.471, "Similarity cloning": 0.134, "Residual inflated": 0.602, "Kejutan severity": 0.593}, intersep -6.49

Pemilihan penggabung (CV per faskes pada val, average precision): bobot val=0.361, maks=0.413 -> dipilih **maks**.

Recall tingkat pasangan pada anggaran 5% (repeat/cloning terdeteksi jika salinan atau klaim sumbernya ditandai): claim_cloning=0.560, repeat_billing=0.571

Tingkat ditandai pada hard negatives (hibrida terpilih; makin rendah makin baik): legit_atypical_coding=0.159, legit_high_cost=0.054, legit_recurring=0.124, legit_revisit=0.000, legit_similar=0.045

Recall hibrida per modus x kesulitan (anggaran 5%):

| modus | easy | medium | hard |
|---|---|---|---|
| claim_cloning | 0.579 | 0.269 | 0.400 |
| inflated_bills | 1.000 | 0.722 | 0.278 |
| repeat_billing | 1.000 | 0.167 | 0.000 |
| upcoding | 1.000 | 0.421 | 0.071 |