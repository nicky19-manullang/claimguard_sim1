# Laporan kalibrasi konformal (tahap 6)

Semua angka dari split **test**. Kalibrasi memakai klaim normal (termasuk hard negatives) di split **val**; detektor tahap 5 tidak diubah. p-value konformal dengan pemecahan seri acak; skor 0-100 = 100 x (1 - p). Tingkat: Critical p<=0,01; High p<=0,03; Medium p<=0,05; Low selainnya. Jaminan FPR <= a hanya berlaku jika klaim normal baru dapat dipertukarkan dengan klaim kalibrasi; di skema faskes (faskes baru) dan skema waktu asumsi itu tidak penuh, jadi dicek empiris dan dilaporkan apa adanya.


## Skema `faskes`, mode `retro` (kalibrasi: klaim normal val per `jenis_layanan`, n = {'RITL': 1401, 'RJTL': 8921}, pool = 10322)
### 1. Validitas: FPR empiris pada klaim normal (test) vs level yang dijanjikan

| kalibrasi | a (dijanjikan) | FPR empiris | CI95 | rasio FPR/a | Precision | Recall | ditandai |
|---|---|---|---|---|---|---|---|
| Mondrian (jenis_layanan) | 0.010 | 0.010 | [0.007, 0.014] | 1.029 | 0.576 | 0.556 | 0.024 |
| Mondrian (jenis_layanan) | 0.030 | 0.025 | [0.020, 0.031] | 0.846 | 0.408 | 0.697 | 0.042 |
| Mondrian (jenis_layanan) | 0.050 | 0.046 | [0.039, 0.053] | 0.922 | 0.283 | 0.725 | 0.063 |
| Mondrian (jenis_layanan) | 0.100 | 0.092 | [0.075, 0.105] | 0.915 | 0.176 | 0.781 | 0.108 |
| Marginal | 0.010 | 0.011 | [0.008, 0.014] | 1.100 | 0.592 | 0.635 | 0.026 |
| Marginal | 0.030 | 0.027 | [0.023, 0.034] | 0.917 | 0.400 | 0.730 | 0.045 |
| Marginal | 0.050 | 0.049 | [0.041, 0.057] | 0.970 | 0.286 | 0.775 | 0.066 |
| Marginal | 0.100 | 0.091 | [0.078, 0.103] | 0.914 | 0.186 | 0.831 | 0.109 |

### 2. Per kelompok (Mondrian)

| kelompok | n normal | n anomali | FPR@0.01 | Recall@0.01 | FPR@0.05 | Recall@0.05 |
|---|---|---|---|---|---|---|
| RITL | 889.000 | 84.000 | 0.019 | 0.250 | 0.064 | 0.512 |
| RJTL | 6203.000 | 94.000 | 0.009 | 0.830 | 0.044 | 0.915 |

### 3. Per tingkat prioritas (tiap tingkat berdiri sendiri, tidak kumulatif)

| tingkat | klaim | % klaim | anomali | precision | % anomali tertangkap | FPR |
|---|---|---|---|---|---|---|
| Critical | 172.000 | 0.024 | 99.000 | 0.576 | 0.556 | 0.010 |
| High | 132.000 | 0.018 | 25.000 | 0.189 | 0.140 | 0.015 |
| Medium | 152.000 | 0.021 | 5.000 | 0.033 | 0.028 | 0.021 |
| Low | 6814.000 | 0.937 | 49.000 | 0.007 | 0.275 | 0.954 |

### 4. Recall per modus x kesulitan pada p <= 0.05

| modus | easy | medium | hard |
|---|---|---|---|
| claim_cloning | 0.667 | 0.176 | 0.000 |
| inflated_bills | 1.000 | 0.870 | 0.571 |
| repeat_billing | 1.000 | 1.000 | 1.000 |
| upcoding | 1.000 | 0.875 | 0.111 |

### 4. Recall per modus x kesulitan pada p <= 0.01

| modus | easy | medium | hard |
|---|---|---|---|
| claim_cloning | 0.400 | 0.118 | 0.000 |
| inflated_bills | 0.880 | 0.739 | 0.214 |
| repeat_billing | 1.000 | 0.900 | 0.625 |
| upcoding | 0.900 | 0.000 | 0.000 |

### 5. Tingkat ditandai pada hard negatives

| tipe | p<=0.01 | p<=0.03 | p<=0.05 |
|---|---|---|---|
| legit_atypical_coding | 0.014 | 0.057 | 0.114 |
| legit_high_cost | 0.000 | 0.031 | 0.031 |
| legit_recurring | 0.002 | 0.012 | 0.084 |
| legit_revisit | 0.017 | 0.155 | 0.215 |
| legit_similar | 0.048 | 0.060 | 0.083 |

### 6. Sebaran FPR antar faskes (klaim normal)

| a | faskes | FPR median | FPR p90 | FPR maks | faskes dengan FPR > 2a |
|---|---|---|---|---|---|
| 0.05 | 11.000 | 0.046 | 0.054 | 0.066 | 0.000 |
| 0.01 | 11.000 | 0.008 | 0.019 | 0.020 | 0.000 |


## Skema `faskes`, mode `asof` (kalibrasi: klaim normal val per `jenis_layanan`, n = {'RITL': 1401, 'RJTL': 8921}, pool = 10322)
### 1. Validitas: FPR empiris pada klaim normal (test) vs level yang dijanjikan

| kalibrasi | a (dijanjikan) | FPR empiris | CI95 | rasio FPR/a | Precision | Recall | ditandai |
|---|---|---|---|---|---|---|---|
| Mondrian (jenis_layanan) | 0.010 | 0.009 | [0.006, 0.011] | 0.888 | 0.512 | 0.371 | 0.018 |
| Mondrian (jenis_layanan) | 0.030 | 0.024 | [0.015, 0.031] | 0.790 | 0.361 | 0.534 | 0.036 |
| Mondrian (jenis_layanan) | 0.050 | 0.044 | [0.032, 0.055] | 0.874 | 0.264 | 0.624 | 0.058 |
| Mondrian (jenis_layanan) | 0.100 | 0.087 | [0.068, 0.103] | 0.867 | 0.177 | 0.742 | 0.103 |
| Marginal | 0.010 | 0.008 | [0.006, 0.010] | 0.790 | 0.594 | 0.461 | 0.019 |
| Marginal | 0.030 | 0.026 | [0.017, 0.033] | 0.874 | 0.356 | 0.579 | 0.040 |
| Marginal | 0.050 | 0.044 | [0.032, 0.053] | 0.874 | 0.272 | 0.652 | 0.059 |
| Marginal | 0.100 | 0.085 | [0.066, 0.102] | 0.852 | 0.189 | 0.792 | 0.102 |

### 2. Per kelompok (Mondrian)

| kelompok | n normal | n anomali | FPR@0.01 | Recall@0.01 | FPR@0.05 | Recall@0.05 |
|---|---|---|---|---|---|---|
| RITL | 889.000 | 84.000 | 0.012 | 0.179 | 0.054 | 0.512 |
| RJTL | 6203.000 | 94.000 | 0.008 | 0.543 | 0.042 | 0.723 |

### 3. Per tingkat prioritas (tiap tingkat berdiri sendiri, tidak kumulatif)

| tingkat | klaim | % klaim | anomali | precision | % anomali tertangkap | FPR |
|---|---|---|---|---|---|---|
| Critical | 129.000 | 0.018 | 66.000 | 0.512 | 0.371 | 0.009 |
| High | 134.000 | 0.018 | 29.000 | 0.216 | 0.163 | 0.015 |
| Medium | 158.000 | 0.022 | 16.000 | 0.101 | 0.090 | 0.020 |
| Low | 6849.000 | 0.942 | 67.000 | 0.010 | 0.376 | 0.956 |

### 4. Recall per modus x kesulitan pada p <= 0.05

| modus | easy | medium | hard |
|---|---|---|---|
| claim_cloning | 0.733 | 0.235 | 0.000 |
| inflated_bills | 1.000 | 0.870 | 0.429 |
| repeat_billing | 1.000 | 0.500 | 0.125 |
| upcoding | 1.000 | 0.750 | 0.111 |

### 4. Recall per modus x kesulitan pada p <= 0.01

| modus | easy | medium | hard |
|---|---|---|---|
| claim_cloning | 0.267 | 0.118 | 0.000 |
| inflated_bills | 0.880 | 0.609 | 0.143 |
| repeat_billing | 1.000 | 0.000 | 0.000 |
| upcoding | 0.500 | 0.000 | 0.000 |

### 5. Tingkat ditandai pada hard negatives

| tipe | p<=0.01 | p<=0.03 | p<=0.05 |
|---|---|---|---|
| legit_atypical_coding | 0.000 | 0.043 | 0.086 |
| legit_high_cost | 0.031 | 0.031 | 0.031 |
| legit_recurring | 0.037 | 0.129 | 0.223 |
| legit_revisit | 0.000 | 0.000 | 0.011 |
| legit_similar | 0.048 | 0.048 | 0.060 |

### 6. Sebaran FPR antar faskes (klaim normal)

| a | faskes | FPR median | FPR p90 | FPR maks | faskes dengan FPR > 2a |
|---|---|---|---|---|---|
| 0.05 | 11.000 | 0.033 | 0.059 | 0.070 | 0.000 |
| 0.01 | 11.000 | 0.008 | 0.010 | 0.015 | 0.000 |


## Skema `waktu`, mode `retro` (kalibrasi: klaim normal val per `jenis_layanan`, n = {'RITL': 1320, 'RJTL': 7821}, pool = 9141)
### 1. Validitas: FPR empiris pada klaim normal (test) vs level yang dijanjikan

| kalibrasi | a (dijanjikan) | FPR empiris | CI95 | rasio FPR/a | Precision | Recall | ditandai |
|---|---|---|---|---|---|---|---|
| Mondrian (jenis_layanan) | 0.010 | 0.008 | [0.005, 0.013] | 0.824 | 0.625 | 0.486 | 0.021 |
| Mondrian (jenis_layanan) | 0.030 | 0.027 | [0.021, 0.035] | 0.890 | 0.404 | 0.642 | 0.044 |
| Mondrian (jenis_layanan) | 0.050 | 0.047 | [0.040, 0.056] | 0.934 | 0.293 | 0.685 | 0.064 |
| Mondrian (jenis_layanan) | 0.100 | 0.093 | [0.080, 0.105] | 0.925 | 0.189 | 0.763 | 0.111 |
| Marginal | 0.010 | 0.011 | [0.008, 0.015] | 1.132 | 0.567 | 0.525 | 0.025 |
| Marginal | 0.030 | 0.028 | [0.022, 0.036] | 0.930 | 0.409 | 0.685 | 0.046 |
| Marginal | 0.050 | 0.045 | [0.038, 0.054] | 0.890 | 0.319 | 0.739 | 0.064 |
| Marginal | 0.100 | 0.090 | [0.078, 0.101] | 0.896 | 0.209 | 0.837 | 0.110 |

### 2. Per kelompok (Mondrian)

| kelompok | n normal | n anomali | FPR@0.01 | Recall@0.01 | FPR@0.05 | Recall@0.05 |
|---|---|---|---|---|---|---|
| RITL | 1362.000 | 116.000 | 0.007 | 0.172 | 0.067 | 0.448 |
| RJTL | 7738.000 | 141.000 | 0.008 | 0.745 | 0.043 | 0.879 |

### 3. Per tingkat prioritas (tiap tingkat berdiri sendiri, tidak kumulatif)

| tingkat | klaim | % klaim | anomali | precision | % anomali tertangkap | FPR |
|---|---|---|---|---|---|---|
| Critical | 200.000 | 0.021 | 125.000 | 0.625 | 0.486 | 0.008 |
| High | 208.000 | 0.022 | 40.000 | 0.192 | 0.156 | 0.018 |
| Medium | 193.000 | 0.021 | 11.000 | 0.057 | 0.043 | 0.020 |
| Low | 8756.000 | 0.936 | 81.000 | 0.009 | 0.315 | 0.953 |

### 4. Recall per modus x kesulitan pada p <= 0.05

| modus | easy | medium | hard |
|---|---|---|---|
| claim_cloning | 0.579 | 0.115 | 0.000 |
| inflated_bills | 1.000 | 0.833 | 0.389 |
| repeat_billing | 1.000 | 1.000 | 0.882 |
| upcoding | 1.000 | 0.368 | 0.000 |

### 4. Recall per modus x kesulitan pada p <= 0.01

| modus | easy | medium | hard |
|---|---|---|---|
| claim_cloning | 0.263 | 0.000 | 0.000 |
| inflated_bills | 0.955 | 0.500 | 0.167 |
| repeat_billing | 1.000 | 0.833 | 0.588 |
| upcoding | 0.619 | 0.000 | 0.000 |

### 5. Tingkat ditandai pada hard negatives

| tipe | p<=0.01 | p<=0.03 | p<=0.05 |
|---|---|---|---|
| legit_atypical_coding | 0.009 | 0.097 | 0.168 |
| legit_high_cost | 0.000 | 0.027 | 0.081 |
| legit_recurring | 0.021 | 0.058 | 0.144 |
| legit_revisit | 0.010 | 0.073 | 0.080 |
| legit_similar | 0.015 | 0.023 | 0.030 |

### 6. Sebaran FPR antar faskes (klaim normal)

| a | faskes | FPR median | FPR p90 | FPR maks | faskes dengan FPR > 2a |
|---|---|---|---|---|---|
| 0.05 | 60.000 | 0.037 | 0.069 | 0.212 | 2.000 |
| 0.01 | 60.000 | 0.003 | 0.020 | 0.121 | 6.000 |


## Skema `waktu`, mode `asof` (kalibrasi: klaim normal val per `jenis_layanan`, n = {'RITL': 1320, 'RJTL': 7821}, pool = 9141)
### 1. Validitas: FPR empiris pada klaim normal (test) vs level yang dijanjikan

| kalibrasi | a (dijanjikan) | FPR empiris | CI95 | rasio FPR/a | Precision | Recall | ditandai |
|---|---|---|---|---|---|---|---|
| Mondrian (jenis_layanan) | 0.010 | 0.011 | [0.008, 0.015] | 1.121 | 0.472 | 0.354 | 0.021 |
| Mondrian (jenis_layanan) | 0.030 | 0.029 | [0.022, 0.036] | 0.967 | 0.320 | 0.482 | 0.041 |
| Mondrian (jenis_layanan) | 0.050 | 0.051 | [0.042, 0.061] | 1.026 | 0.241 | 0.576 | 0.066 |
| Mondrian (jenis_layanan) | 0.100 | 0.097 | [0.083, 0.111] | 0.967 | 0.165 | 0.677 | 0.113 |
| Marginal | 0.010 | 0.013 | [0.009, 0.017] | 1.286 | 0.471 | 0.405 | 0.024 |
| Marginal | 0.030 | 0.033 | [0.026, 0.040] | 1.084 | 0.312 | 0.521 | 0.046 |
| Marginal | 0.050 | 0.052 | [0.043, 0.064] | 1.048 | 0.244 | 0.599 | 0.067 |
| Marginal | 0.100 | 0.099 | [0.087, 0.113] | 0.991 | 0.171 | 0.724 | 0.116 |

### 2. Per kelompok (Mondrian)

| kelompok | n normal | n anomali | FPR@0.01 | Recall@0.01 | FPR@0.05 | Recall@0.05 |
|---|---|---|---|---|---|---|
| RITL | 1362.000 | 116.000 | 0.008 | 0.172 | 0.063 | 0.422 |
| RJTL | 7738.000 | 141.000 | 0.012 | 0.504 | 0.049 | 0.702 |

### 3. Per tingkat prioritas (tiap tingkat berdiri sendiri, tidak kumulatif)

| tingkat | klaim | % klaim | anomali | precision | % anomali tertangkap | FPR |
|---|---|---|---|---|---|---|
| Critical | 193.000 | 0.021 | 91.000 | 0.472 | 0.354 | 0.011 |
| High | 195.000 | 0.021 | 33.000 | 0.169 | 0.128 | 0.018 |
| Medium | 227.000 | 0.024 | 24.000 | 0.106 | 0.093 | 0.022 |
| Low | 8742.000 | 0.934 | 109.000 | 0.012 | 0.424 | 0.949 |

### 4. Recall per modus x kesulitan pada p <= 0.05

| modus | easy | medium | hard |
|---|---|---|---|
| claim_cloning | 0.474 | 0.192 | 0.200 |
| inflated_bills | 1.000 | 0.861 | 0.389 |
| repeat_billing | 1.000 | 0.567 | 0.000 |
| upcoding | 1.000 | 0.263 | 0.000 |

### 4. Recall per modus x kesulitan pada p <= 0.01

| modus | easy | medium | hard |
|---|---|---|---|
| claim_cloning | 0.158 | 0.000 | 0.000 |
| inflated_bills | 0.955 | 0.528 | 0.167 |
| repeat_billing | 1.000 | 0.000 | 0.000 |
| upcoding | 0.667 | 0.053 | 0.000 |

### 5. Tingkat ditandai pada hard negatives

| tipe | p<=0.01 | p<=0.03 | p<=0.05 |
|---|---|---|---|
| legit_atypical_coding | 0.009 | 0.071 | 0.106 |
| legit_high_cost | 0.000 | 0.000 | 0.000 |
| legit_recurring | 0.062 | 0.141 | 0.207 |
| legit_revisit | 0.000 | 0.000 | 0.003 |
| legit_similar | 0.023 | 0.023 | 0.038 |

### 6. Sebaran FPR antar faskes (klaim normal)

| a | faskes | FPR median | FPR p90 | FPR maks | faskes dengan FPR > 2a |
|---|---|---|---|---|---|
| 0.05 | 60.000 | 0.037 | 0.075 | 0.129 | 3.000 |
| 0.01 | 60.000 | 0.005 | 0.023 | 0.047 | 8.000 |