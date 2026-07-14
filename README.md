## “Old-school” Guidance to Learn A Better Traffic Signal Control Agent

This is official repo of paper "“Old-school” Guidance to Learn A Better Traffic Signal Control Agent".

### How to use


#### Dataset

The dataset is not included in this repository. Download
[`sumo_fenglin_base_sub1.zip`](https://github.com/AnonymousIDforSubmission/GuidedLight/blob/main/sumo_files/scenarios/sumo_fenglin_base_sub1.zip)
from the original repository, place it under `sumo_files/scenarios`, and
extract it before training or evaluation:

```shell
cd sumo_files/scenarios
unzip sumo_fenglin_base_sub1.zip
cd -
```

#### Reproducibility environment

The experiments documented in this repository were run in the conda
environment named `GuideLight`. The versions below were read directly from
that environment and replace the versions from the original reproducibility
instructions.

| Component | Version |
| --- | --- |
| Python | 3.9.25 |
| Eclipse SUMO / libsumo / sumolib / TraCI | 1.14.0 |
| PyTorch | 1.11.0+cu113 |
| torchvision | 0.12.0+cu113 |
| torchaudio | 0.11.0+cu113 |
| CUDA toolkit used by PyTorch | 11.3 |
| cuDNN | 8.2.0 |
| NumPy | 1.19.5 |
| SciPy | 1.7.3 |
| Matplotlib | 3.5.3 |
| Gym | 0.25.2 |
| TensorBoard / TensorBoardX | 2.14.1 / 2.6.5 |
| Weights & Biases | 0.15.3 |
| tqdm | 4.59.0 |

Activate the environment and configure SUMO and the project import path:

```shell
conda activate GuideLight
export SUMO_HOME="${CONDA_PREFIX}/lib/python3.9/site-packages/sumo"
export PYTHONPATH="${PYTHONPATH}:${SUMO_HOME}/tools:$(pwd)"
```

Verify the principal runtime versions with:

```shell
python --version
sumo --version
python -c "import numpy, scipy, matplotlib, gym, torch; print(numpy.__version__, scipy.__version__, matplotlib.__version__, gym.__version__, torch.__version__)"
```

#### Training and evaluation

- Model training
```shell
python -u tsc/main.py
```
- Eval
```shell
python -u tsc/eval_v2.py   # Evaluate all checkpoints and save results.
# or
python -u tsc/eval_model.py  # Evaluate the specified checkpoint, you can also adjust eval_config for visualization
```

### Sudden Traffic-Volume Evaluation
Use `sumo_fenglin_base_sub4` as the isolated shock experiment folder. To
create 50 paired 24-hour `1.5x` shock routes from baseline routes `base_1`
through `base_50`:
```shell
cd sumo_files/scenarios/sumo_fenglin_base_sub4
python generate_shock_flows.py \
  --indices 1-50 \
  --multiplier 1.5 \
  --shock-begin 25200 \
  --shock-end 28800 \
  --generate-routes
cd -
```

Export 5-minute departure-flow statistics for the paired 50-route cohort. The
CSV stores raw per-route bins plus cohort summary statistics; the PNG plots
mean flow with standard-deviation shading:
```shell
cd GuidedLight
python sumo_files/scenarios/sumo_fenglin_base_sub4/plot_shock_analysis.py flow \
  --indices 1-50
```

For policy action-generated cycle time over a full day, evaluate the same 50
shock and 50 baseline routes in four segments. A continuous 24-hour run can
cause SUMO to terminate after extreme accumulated congestion on these routes;
segmenting keeps the requested 24-hour time axis while retaining a 15-minute
observation pre-roll at each non-midnight boundary. Each segment starts on an
empty network and excludes departures before its start. The policy still acts
every 15 minutes, as in training, and analysis places each cycle time in its
three covered 5-minute bins:
```shell
conda activate GuideLight
source setup_env39.sh
for segment in h00_h06 h06_h12 h12_h18 h18_h24; do
  case "$segment" in
    h00_h06) start=0; duration=21600 ;;
    h06_h12) start=20700; duration=22500 ;;
    h12_h18) start=42300; duration=22500 ;;
    h18_h24) start=63900; duration=22500 ;;
  esac
  python -u tsc/eval_model.py --condition shock_routes_1_50_${segment} \
    --model model_760.pt --num-eps 50 --start-time "$start" --duration "$duration" \
    --route-dir sumo_fenglin_base_sub4 --port-start 23969 --no-tripinfo --skip-pre-start-departures --sumo-seed 777 \
    > sumo_files/scenarios/sumo_fenglin_base_sub4/eval_logs/shock_routes_1_50_${segment}.log 2>&1
  python -u tsc/eval_model.py --condition baseline_routes_1_50_${segment} \
    --model model_760.pt --num-eps 50 --start-time "$start" --duration "$duration" \
    --route-dir sumo_fenglin_base_sub1 --port-start 23970 --no-tripinfo --skip-pre-start-departures --sumo-seed 777 \
    > sumo_files/scenarios/sumo_fenglin_base_sub4/eval_logs/baseline_routes_1_50_${segment}.log 2>&1
done
python sumo_files/scenarios/sumo_fenglin_base_sub4/plot_shock_analysis.py cycle \
  --indices 1-50 \
  --shock-conditions shock_routes_1_50_h00_h06,shock_routes_1_50_h06_h12,shock_routes_1_50_h12_h18,shock_routes_1_50_h18_h24 \
  --baseline-conditions baseline_routes_1_50_h00_h06,baseline_routes_1_50_h06_h12,baseline_routes_1_50_h12_h18,baseline_routes_1_50_h18_h24 \
  --window-begin 0 --window-end 86400
```
`eval_model.py` stores cycle time after applying every policy action for each
executed 15-minute interval. The midnight segment has no earlier traffic
observation, so `00:00-00:15` is empty in the CSV; the other 285 five-minute
bins from `00:15` through `24:00` contain action-generated cycle times.
`--no-tripinfo` avoids large trip XML files; the required policy pkl records
are still stored. All outputs remain under `sumo_fenglin_base_sub4`.

To run the same complete evaluation, cycle-time export, and synchronization
plots with another checkpoint under `tsc/eval_models`, use the wrapper script.
Condition names and analysis folders include the checkpoint name, so outputs
from different models do not overwrite one another:
```shell
cd GuidedLight
conda activate GuideLight
source setup_env39.sh

# model_4520.pt: evaluation, cycle-time statistics, and figures.
bash sumo_files/scenarios/sumo_fenglin_base_sub4/run_model_shock_evaluation.sh model_4520.pt

# model_4660.pt: a separate set of evaluation logs and figures.
bash sumo_files/scenarios/sumo_fenglin_base_sub4/run_model_shock_evaluation.sh model_4660.pt
```
The optional second argument limits the number of paired routes for a quicker
check, for example `... model_4520.pt 1`. Model-specific results are written
under `sumo_fenglin_base_sub4/analysis_plots/*_model_4520/` and
`sumo_fenglin_base_sub4/analysis_plots/*_model_4660/`. Each directory contains
cycle-time CSV/plots, paired overall synchronization figures, and
`by_intersection/` figures whose baseline/shock scales are taken from that
intersection's shock result.

For a `2x` shock, replace `--multiplier 1.5` with `--multiplier 2`.

Note that we currently do not provide scats in the guidance model here (the relevant code has been commented out). You can refer to official scats and modify get_sdk_label in PPO.py to complete.