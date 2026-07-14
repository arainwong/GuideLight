#!/usr/bin/env python3
# encoding: utf-8

import argparse
import os
import sys
from eval_config import config
from sumo_files.env.sim_env import TSCSimulator
# from model import NN_Model
import numpy as np
import pickle
from utils import *
from model import NN_Model
from utils import load_checkpoint2
import torch


def get_action(prediction):
    return prediction['a'].cpu().detach().numpy()


def get_reward(reward, all_tls):
    ans = []
    for i in all_tls:
        ans.append(sum(reward[i].values()))
    return ans


def get_cycle_flow_record(env, state):
    """Store the interval executed after a policy action in plot-friendly form."""
    interval = 60 * 15
    return {
        "begin": env._current_time - interval,
        "end": env._current_time,
        "junctions": {
            tl: {
                "cycle_time": float(sum(state[tl]["duration"])),
                "flow": float(sum(state[tl]["flow"])),
            } for tl in env.all_tls
        },
    }


def required_route_paths(num_eps):
    sumocfg_file = env_config['sumocfg_file']
    route_name = sumocfg_file.split("/")[-1][:-8]
    route_dir = env_config.get("route_dir", "sumo_fenglin_base_sub1")
    if os.path.isabs(route_dir):
        route_root = route_dir
    else:
        route_root = "/".join(sumocfg_file.split("/")[:-2] + [route_dir])
    return [os.path.join(route_root, "{}_{}.rou.xml".format(route_name, i))
            for i in range(1, num_eps + 1)]


def model_eval():
    all_reward_list = {}
    for k in config['environment']['reward_type'] + ['all']:
        all_reward_list[k] = []

    num_eps = config['episode']['test_num_eps']
    missing_routes = [path for path in required_route_paths(num_eps) if not os.path.exists(path)]
    if missing_routes:
        raise FileNotFoundError("Missing evaluation routes: {}".format(", ".join(missing_routes)))
    start_episode = env_config.get('start_episode', 0)
    if start_episode >= num_eps:
        print("No episodes to run: start_episode={} num_episodes={}".format(start_episode, num_eps))
        return
    print("Evaluating episodes {}-{} with routes in {}".format(
        start_episode, num_eps - 1, env_config['route_dir']))

    spe_model = config['model_save']['spe_path']
    #判断当前模型是属于所有checkpoints里第几个check
    p = int(spe_model.split('_')[1][:-3]) // config['model_save']['frequency']
    for i in range(start_episode, num_eps):#重复模拟的次数
        lstm_state = (
            torch.zeros(1, 10, 128).to(device),
            torch.zeros(1, 10, 128).to(device),
        )
        model = NN_Model(12, 4, state_keys=env_config['state_key'], device=device)
        # model = NN_Model(9, 4, state_keys=env_config['state_key'], device=device)
        model = load_checkpoint2(model, config['model_save']['path'] + spe_model) #读取指定的模型文件
        model.eval()
        # model = load_checkpoint(model, config['model_save']['path'])
        env_config['step_num'] = i + 1 #使用对应checkpoint所使用的route文件
        env_config['p'] = 2 * p + 1
        env_config['output_path'] = env_config['output_path_head'] + f'trial_{i}/' #不同的重复实验保存到不同的文件夹
        if not os.path.exists(env_config['output_path']):
            os.makedirs(env_config['output_path'])
        env = TSCSimulator(env_config, port)
        # The constructor probes a route and advances step_num; reset it so the
        # episode evaluates the route requested for this trial.
        env.step_num = env_config['step_num']
        state = env.reset()
        unava_phase_index = []
        for j in env.all_tls:
            unava_phase_index.append(env._crosses[j].unava_index)
        tl_action = []
        tl_phase_duration = []
        tl_state = []
        tl_reward = []
        tl_cycle_flow = []
        while True:
            tmp_state = {}
            for tl_index in range(len(env.all_tls)):
                tmp_state[env.all_tls[tl_index]] = state[env.all_tls[tl_index]]
            tl_state.append(tmp_state)
            state = batch(state, config['environment']['state_key'], env.all_tls)
            prediction, lstm_state, _ = model(state, lstm_state, unava_phase_index, eval=True)
            # action = get_action(prediction)

            action = prediction['a']
            # get phase duration

            for tl_index in range(len(env.all_tls)):
                tl_phase_duration.append({env.all_tls[tl_index]:
                    (env._crosses[env.all_tls[tl_index]].getCurrentPhase())})
                # tl_action.append({env.all_tls[tl_index]: action[tl_index]})
                tl_action.append({env.all_tls[tl_index]: [action[i][tl_index] for i in range(4)]})

            tl_action_select = {}
            # for tl_index in range(len(env.all_tls)):
            #     tl_action_select[env.all_tls[tl_index]] = action[tl_index]
            for tl_index in range(len(env.all_tls)):
                tl_action_select[env.all_tls[tl_index]] = []
                for index in range(4):
                    tl_action_select[env.all_tls[tl_index]].append(action[index][tl_index])

            state, reward, done, _all_reward = env.step(tl_action_select)
            tl_cycle_flow.append(get_cycle_flow_record(env, state))
            tl_reward.append(reward)
            reward = get_reward(reward, env.all_tls)
            if done:
                all_reward = _all_reward
                with open(os.path.join(env_config['output_path'],
                                       "plt_{}_{}.pkl".format(
                                               env_config['sumocfg_file'].split("/")[-1], p)),
                          "wb") as f:
                    pickle.dump({"tl_phase_duration": tl_phase_duration,
                                 "tl_state":tl_state,
                                 "tl_reward": tl_reward,
                                 "tl_action": tl_action,
                                 "tl_cycle_flow": tl_cycle_flow},
                                f)
                break
        for tl in all_reward.keys():
            all_reward[tl]['all'] = sum(all_reward[tl].values())
        for k in config['environment']['reward_type']+['all']:
            tmp = 0
            for tl in all_reward.keys():
                tmp += all_reward[tl][k]
            all_reward_list[k].append(tmp/len(all_reward))

    for k, v in all_reward_list.items():
        print("{} Model Avg {}: {}".format(env_config['sumocfg_file'], k, sum(v)/len(v)))


def default_eval():
    # time = [25, 25, 25, 25] in intersection
    env_config['output_path'] = env_config['output_path_head'] + f'trial_0/'
    if not os.path.exists(env_config['output_path']):
        os.makedirs(env_config['output_path'])
    env = TSCSimulator(env_config, port, not_default=False)
    state = env.reset_default()
    tl_action = []
    tl_phase_duration = []
    tl_state = []
    tl_reward = []
    while True:
        tmp_state = {}
        for tl_index in range(len(env.all_tls)):
            tmp_state[env.all_tls[tl_index]] = state[env.all_tls[tl_index]]
        tl_state.append(tmp_state)
        for tl_index in range(len(env.all_tls)):
            tl_phase_duration.append({env.all_tls[tl_index]:
                                          (env._crosses[env.all_tls[tl_index]].getCurrentPhase())})
            tl_action.append({env.all_tls[tl_index]: [0,0,0,0]})
        state, reward, done, _all_reward = env.default_step()
        tl_reward.append(reward)
        if done:
            all_reward = _all_reward
            with open(os.path.join(env_config['output_path'], "plt_tsc_tmp.pkl"), "wb") as f:
                pickle.dump({"tl_phase_duration": tl_phase_duration,
                             "tl_state":tl_state,
                             "tl_reward": tl_reward,
                             "tl_action": tl_action},
                            f)
            break
    for tl in all_reward.keys():
        all_reward[tl]['all'] = sum(all_reward[tl].values())
    for k in config['environment']['reward_type']+["all"]:
        tmp = 0
        for tl in all_reward.keys():
            tmp += all_reward[tl][k]
        print("{} Default Avg {}: {}".format(env_config['sumocfg_file'], k, tmp/len(all_reward)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a model on configured SUMO routes.")
    parser.add_argument("--num-eps", type=int, default=config['episode']['test_num_eps'],
                        help="Number of evaluation episodes.")
    parser.add_argument("--start-episode", type=int, default=0,
                        help="First zero-based trial index to run; useful for resuming partial conditions.")
    parser.add_argument("--start-time", type=int, default=None,
                        help="Override simulation start time in seconds.")
    parser.add_argument("--duration", type=int, default=None,
                        help="Override simulated duration in seconds; use a short value for smoke tests.")
    parser.add_argument("--condition", default=None,
                        help="Output condition name, for example baseline or shock_1p5_h7.")
    parser.add_argument("--route-dir", default=None,
                        help="Route scenario directory, for example sumo_fenglin_base_sub1.")
    parser.add_argument("--model", default=None, help="Checkpoint filename under the model path.")
    parser.add_argument("--port-start", type=int, default=None,
                        help="Override TraCI port, useful when evaluating conditions in parallel.")
    parser.add_argument("--no-tripinfo", action="store_true",
                        help="Skip SUMO tripinfo XML output while still saving policy pickle data.")
    parser.add_argument("--skip-pre-start-departures", action="store_true",
                        help="Begin SUMO at --start-time with an empty network, for robust time-sliced runs.")
    parser.add_argument("--sumo-seed", type=int, default=None,
                        help="Use a fixed SUMO random seed so paired evaluations are comparable.")
    args = parser.parse_args()
    if args.num_eps < 1:
        parser.error("--num-eps must be positive")
    if args.start_episode < 0:
        parser.error("--start-episode must not be negative")
    if args.start_time is not None and args.start_time < 0:
        parser.error("--start-time must not be negative")
    if args.duration is not None and args.duration < 1:
        parser.error("--duration must be positive")

    device = 'cpu'
    # config["environment"]['gui'] = True
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    env_config = config['environment']
    config['episode']['test_num_eps'] = args.num_eps
    env_config['start_episode'] = args.start_episode
    if args.start_time is not None:
        env_config['episode_start_time'] = args.start_time
    if args.duration is not None:
        env_config['episode_length_time'] = args.duration
    if args.model:
        config['model_save']['spe_path'] = args.model
    if args.route_dir:
        env_config['route_dir'] = args.route_dir
    if args.port_start is not None:
        env_config['port_start'] = args.port_start
    if args.no_tripinfo:
        env_config['is_record'] = False
    if args.skip_pre_start_departures:
        env_config['skip_pre_start_departures'] = True
    if args.sumo_seed is not None:
        env_config['seed'] = args.sumo_seed
        env_config['use_fixed_seed'] = True
    if args.condition:
        env_config['name'] = args.condition
        env_config['output_path_head'] = os.path.join(
            'sumo_files/scenarios/sumo_fenglin_base_sub4/eval_logs', args.condition, '')

    for i in range(len(env_config['sumocfg_files'])):
        env_config['sumocfg_file'] = env_config['sumocfg_files'][i]
        port = env_config['port_start']

        model_eval() # eval model
        # default_eval()  # eval FTC
