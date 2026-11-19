# Validation report

Clips per model: 60, tokens per clip: 1024

## 1. Training convergence
- **lstm**: best val loss 0.761 at epoch 10/10; final train 0.712 vs val 0.761 (gap +0.049); last 3 val ['0.765', '0.761', '0.761']; still improving: False
- **transformer**: best val loss 0.564 at epoch 10/10; final train 0.563 vs val 0.564 (gap +0.001); last 3 val ['0.570', '0.566', '0.564']; still improving: True

## 2. Control accuracy (95% confidence intervals; with ~60 clips differences < 10 points are noise)
- **lstm**: instrument set exact 80% (95% CI 68-88%); density bin exact 53% (95% CI 41-65%), within one bin 90% (95% CI 80-95%), mean error 0.58 bins (chance is about 25% exact)
    - drums: played 57/58 when requested; played 0/2 when NOT requested
    - piano: played 34/38 when requested; played 0/22 when NOT requested
    - guitar: played 41/43 when requested; played 0/17 when NOT requested
    - bass: played 53/54 when requested; played 0/6 when NOT requested
    - strings: played 32/36 when requested; played 0/24 when NOT requested
- **transformer**: instrument set exact 78% (95% CI 66-87%); density bin exact 57% (95% CI 44-68%), within one bin 95% (95% CI 86-98%), mean error 0.48 bins (chance is about 25% exact)
    - drums: played 58/58 when requested; played 0/2 when NOT requested
    - piano: played 33/38 when requested; played 0/22 when NOT requested
    - guitar: played 38/43 when requested; played 0/17 when NOT requested
    - bass: played 54/54 when requested; played 0/6 when NOT requested
    - strings: played 33/36 when requested; played 0/24 when NOT requested

## 3. Musical-structure metrics: generated vs REAL test clips (mean ± std)
| source | notes_per_bar | key_corr | in_key | bar_repeat | empty_bars |
|---|---|---|---|---|---|
| real_baseline | 39.63 ± 16.11 | 0.83 ± 0.13 | 0.94 ± 0.09 | 0.04 ± 0.13 | 0.00 ± 0.00 |
| lstm | 42.73 ± 18.33 | 0.82 ± 0.09 | 0.97 ± 0.05 | 0.01 ± 0.05 | 0.00 ± 0.00 |
| transformer | 46.70 ± 22.65 | 0.80 ± 0.08 | 0.96 ± 0.07 | 0.14 ± 0.20 | 0.00 ± 0.00 |

(key_corr / in_key: how clearly the notes sit in one musical key. bar_repeat: share of repeated bars. Values close to the real row = music that is structured like real music.)

## 4. Copying check: share of bars (first 48 tokens) found verbatim in the training set
- real test clips (natural baseline): 2.8%
- lstm: 0.0% of 398 bars
- transformer: 0.0% of 366 bars

If a model's number is far ABOVE the real baseline it is memorising songs; at or below the baseline means it is producing new material.

## 5. Listening test
Files in `D:\Music Generation Project\results\listening_test`: rate each clip 1-5 in `rating_sheet.csv` (ask 3-5 friends, do NOT open the answer key first), then average the ratings per source.