# Kaggle notebooks

All compute for this project runs on Kaggle. The local machine only edits code and commits.
Each folder is a Kaggle kernel: one `.ipynb` plus its `kernel-metadata.json`.

| Notebook | Accelerator | What it does |
|---|---|---|
| `nb00-setup-tests` | CPU | Clones the branch, runs `pytest`, runs the synthetic smoke test, writes `environment.json` and `pip_freeze.txt` |
| NB01–NB08 | see `IMPLEMENTATION_PLAN.md` §2 | Added phase by phase |

Push and run a kernel with the [Kaggle CLI](https://github.com/Kaggle/kaggle-api), using credentials in `~/.kaggle/kaggle.json`:

```bash
kaggle kernels push -p kaggle/nb00-setup-tests
```

Then check its status and download its outputs:

```bash
kaggle kernels status devgurucodes/eegrep-nb00-setup-tests
```

```bash
kaggle kernels output devgurucodes/eegrep-nb00-setup-tests -p kaggle/nb00-setup-tests/output
```

Kernels are private. Every notebook clones the `feat/implementation` branch at run time, so push code before you push a kernel.
