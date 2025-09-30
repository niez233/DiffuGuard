This directory contains the code used for the experiments described in Section 3 of the paper.

## Quick usage

- Generate the logits heatmap used for Figure 2:

    ```python
    python heatmap.py
    ```

    Additional parameters can be changed inside the `main()` function of `heatmap.py`.

- Reproduce Figure 3 (random remask experiments):

    ```python
    bash exp_remask_randomness.sh
    ```
  - You can modify `random_rate` inside the script to change the randomness.
  - After running experiments, evaluate results with:
    `bash eval.sh`.

- Reproduce Figures 4 & 5 (token injection experiments):

```python
  bash exp_token_injection.sh
```

  - You can modify configuration options in the script such as `injection_str` and `injection_step`.
  - Evaluate results with: `bash eval.sh`.