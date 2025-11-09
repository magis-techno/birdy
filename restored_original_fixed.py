import torch
import torch.nn as nn
import math
from torch.distributions import Normal
from collections import OrderedDict
from easydict import EasyDict as edict
import numpy as np

class DiffusionPolicyDense_RL(nn.Module):
    """
    扩散策略密集强化学习模型
    """
    train_phase = False
    trained_head_state_dict = None

    def __init__(self, cfg):
        super().__init__(cfg)
        ####### rlft参数 #######
        self.rlft = cfg.get("rlft", True)
        self.pre_trained_model_dir = cfg.get("pre_trained_model_dir","obs://yw-ads-training-gy1/data/external/personal/x00665867/fv18.2_altC/Iter-150000_StepShot.pth")
        self.gaussian = cfg.get('gaussian', False)
        ###### GRPO Params #######
        self.sde_solver = cfg.get('sde_solver', True) # sde solver开关
        self.sigma_max = cfg.get('sigma_max', 0.99)
        self.white_box_sample = cfg.get('white_box_sample', False)
        DiffusionPolicyDense_RL.train_phase = True if self.grpo_train_phase else False
        self.use_replay_memory = cfg.get("use_replayMemory", False)
        # 修复：这可能是某个函数调用的参数，需要补充上下文
        self.replay_memory = ReplayMemory(
            capacity=cfg.get("memory_capacity", 1000),
            mem_in_mode=self.mem_in_mode
        )
    def forward(self, *inputs, **kwargs):
        # self.load_rl_model(inputs)
        """
        适用于RLFT阶段的Forward
        output:
        ode output: predict_flow, gt_flow, pred_traj, init_noise, t
        sde output: (noise_trajs_gp, sigma_maxs, sde_infos, ego_gt_with_groups), (group_trajs, group_trajs_each_step, sde_predict_flow, group_flows_each_step), log_pi
        """
        # 模型参数初始化
        if DiffusionPolicyDense_RL.train_phase and self.init_count==0:
        self._init_params()
        if not self.grpo_train_phase and (self.rule == 'monitor' or self.rule == 'policy_old'):
        # 'monitor', 'policy_old'纯推理模式不过
        return None
        ## params update for policy old
        if DiffusionPolicyDense_RL.train_phase and (self.rule == "policy_old" or self.rule == "trained"):
        self._update_policy_old_params()
        # 解析输入参数
            (ego_condition,
            static_condition,
            cognition_condition,
            object_feat,
            object_mask,
            navi_condition,
            navi_condition_mask,
            navinn_condition,
            navinn_condition_mask,
            train_input
        ) = inputs[:12]
        b = ego_condition.shape[0]
        # 原地操作的clone进buffer用于_new
        obj_white_exclude_ego = object_feat.clone()
        static_condition_ori = static_condition.clone()
        cognition_condition_ori = cognition_condition.clone()
        navinn_condition_ori = navinn_condition.clone()
        navinn_condition_mask_ori = navinn_condition_mask.clone()
        multi_joint_key_obj_index = train_input['multi_joint_key_obj_index']
        multi_joint_key_obj_mask = train_input['multi_joint_key_obj_mask']
        modality_mask = train_input['modality_mask']
        lateral_goal = train_input['lateral_goal']
        inter_frame = train_input['inter_frame']
        inter_frame_mask = torch.rand(b, dtype=ego_condition.dtype, device=ego_condition.device)
        neg_gt_traj_with_path = kwargs['labels'][0]["neg_gt_traj_with_path"]
        gt_traj_10m_path = kwargs['labels'][0]["path_10m"]
        ego_gt_traj_mask = kwargs["labels"][0]["gt_traj_mask"][:, :1, : self.traj_len]
        obj_gt_traj_mask = kwargs["labels"][0]["gt_traj_mask"][
            :, 1:, : self.traj_len
        ]
        obj_full_gt_traj = kwargs["labels"][0]["full_gt_traj"][:, 1:, :self.traj_len, :2]
        else:
        ego_condition = inputs[0]['conditions_dict'].ego_condition # (b*memory_capacity, ... )
        obj_white_exclude_ego = inputs[0]['conditions_dict'].obj_white_exclude_ego
        object_mask = inputs[0]['conditions_dict'].object_mask
        multi_joint_key_obj_mask = inputs[0]['conditions_dict'].multi_joint_key_obj_mask
        modality_mask = inputs[0]['conditions_dict'].modality_mask
        lateral_goal = inputs[0]['conditions_dict'].lateral_goal
        inter_frame = inputs[0]['conditions_dict'].inter_frame
        gt_traj_10m_path = inputs[0]['labels_dict'].gt_traj_10m_path
        device = ego_condition.device
        self.device = device
        ego_feat_with_mm, obj_feat_with_mm = self.get_condition_feat(ego_condition, condition, is_onnx=False, inter_frame=inter_frame_mask)
        # pred key object
        object_feat = ego_condition + object_feat
        object_feat = self.static_ca_layers(object_feat, static_condition, None, "axial")[0]
        object_score = inputs[0]['conditions_dict'].object_score
        score_mask = (topk_score > self.interaction_obj_score_threshold).to(torch.float).squeeze(-1)
        expanded_indices = topk_indices[..., None, None].expand(-1, -1, self.traj_len, 2)
        "static_condition": static_condition,
            "cognition_condition": cognition_condition,
            "navinn_condition": navinn_condition,
            "navinn_condition_mask": navinn_condition_mask,
            "navi_condition": navi_condition,
            "navi_condition_mask": navi_condition_mask,
            }
        if self.open_behavior:
        c.update({"ego_feat_with_mm": ego_feat_with_mm})
        shortcut_inputs.obj_gt_traj = obj_gt_traj
        shortcut_inputs.object_feat = object_feat
        shortcut_inputs.gt_traj_mask = gt_traj_mask
        shortcut_inputs.ori_b = ori_b
        utokens = edict()
        utokens.ego_feat_with_mm = ego_feat_with_mm
        ego_gt_traj = full_gt_traj[:, :self.traj_len, :2]
        ego_gt_traj = torch.cat((full_gt_traj[:, :self.traj_len, :2], gt_traj_10m_path), dim=1)
        t, gt_flow, ode_traj, pred_traj = None, None, None, None
        ############### 生成noise input ###############
        init_state =self.get_init_state_z0(bs=b, mode=self.init_z0_mode, obj_num=obj_num) ## 初始化z0
        if self.rule == "policy_old":
        )
        else:
        # monitor/policy_old get groups trajs(init noise) from policy flow model
        ############### reverse sde ###############
        gt_guidance_tag = kwargs['labels'][0]['is_cutin_scene'] if self.use_cutin_tag else torch.ones_like(kwargs['labels'][0]['is_cutin_scene'])
        decoder_conditions = \
        [
            tokens, ego_condition, object_feat,
            final_condition, gt_traj_mask,
            ego_feat_with_mm, obj_feat_with_mm
        ]
        # grpo
        sde_sample_results = self.sample_sde(
        # 对齐decoder输入
        conditions = decoder_conditions,
        # rlft特性
        neg_gt_traj_with_path = neg_gt_traj_with_path,
            gt_guidance_tag = gt_guidance_tag,
            groups=self.groups,
            tps_train=self.tps_train,
            denoise_trajs=noise_trajs_gp,
            sigma_maxs=sigma_maxs,
            x_t_old=x_t_old if self.rule != 'policy_old' else None # pi_new时基于pi_old推; ref时基于pi_new
        )
        # gt 监督obj 轨迹
        z0 = None
        ode_traj = None
        ode_obj_traj = None
        # if self.rule == 'trained':
        #  _noise_traj_t = torch.zeros_like(noise_traj_t)
        ## 可视化接口 For ODE Sampling
        infer_condition = bool((~self.training) & (self.rule == "trained")) # 推理基于pi_new推
        ode_traj = self._traj_interpolates(trajs=ode_traj)
        )
        if self.training and self.rule == "trained":
        masks_dict = edict()
        if self.rule == "policy_old":
        ## Package Conditions for memory
        conditions_for_memory=edict()
        conditions_for_memory.navinn_condition = navinn_condition_ori
        conditions_for_memory.navinn_condition_mask = navinn_condition_mask_ori
        conditions_for_memory.multi_obj_interaction_tag = multi_obj_interaction_tag
        conditions_for_memory.inter_frame = inter_frame
        conditions_for_memory.object_score = object_score.clone() # pi_old
        trj_rewards=trj_rewards,
            gt_rewards=gt_rewards,
            memory_valid_mask=memory_valid_mask
        )
        memory_capacity = self.replay_memory.cur_capacity
        conditions_dict.ego_condition = sde_sample_results_from_memory.ego_condition
        conditions_dict.inst_condition = sde_sample_results_from_memory.inst_condition
        conditions_dict.god_condition = sde_sample_results_from_memory.god_condition
        conditions_dict.static_condition = sde_sample_results_from_memory.static_condition
        conditions_dict.cognition_condition = sde_sample_results_from_memory.cognition_condition
        conditions_dict.object_feat = sde_sample_results_from_memory.object_feat
        conditions_dict.object_score = sde_sample_results_from_memory.object_score
        conditions_dict.obj_gt_traj_mask = sde_sample_results_from_memory.obj_gt_traj_mask
        labels_dict.full_gt_traj = sde_sample_results_from_memory.full_gt_traj
        labels_dict.neg_gt_traj_with_path = sde_sample_results_from_memory.neg_gt_traj_with_path
        labels_dict.gt_traj_10m_path = sde_sample_results_from_memory.gt_traj_10m_path
        labels_dict.gt_traj = sde_sample_results_from_memory.gt_traj
        labels_dict.is_follow_car = sde_sample_results_from_memory.is_follow_car
        labels_dict.dis_to_road_end = sde_sample_results_from_memory.dis_to_road_end
        labels_dict.obj_gt_traj_mask = sde_sample_results_from_memory.obj_gt_traj_mask_ori
        conditions_dict.navi_condition = sde_sample_results_from_memory.navi_condition
        conditions_dict.modality_mask = sde_sample_results_from_memory.modality_mask
        labels_dict.object_position = sde_sample_results_from_memory.object_position
        conditions_dict.inst_condition = inst_condition_ori
        labels_dict.full_gt_traj = kwargs['labels'][0]["full_gt_traj_trans"]
        conditions_dict.navi_condition = navi_condition_ori
        conditions_dict.modality_mask = modality_mask
        conditions_dict.lateral_goal = lateral_goal
        conditions_dict.inter_frame = inter_frame
        labels_dict.obj_gt_traj_mask = kwargs["labels"][0]["gt_traj_mask"][:, 1:, : self.traj_len]
        labels_dict.object_position = kwargs['labels'][0]["object_position"]
        return dict(
            conditions_dict = conditions_dict,
            read_replaymemory_flag = read_replaymemory_flag,
            ode_predict_traj=ode_traj,
            ode_predict_obj_traj=ode_obj_traj,
            gt_obj_traj = obj_gt_traj,
        # joint_predict_object_mask = joint_predict_object_mask,
            object_score = object_score if self.rule == "policy_old" else object_score_new,
            gt_flow=gt_flow,
            pred_traj=pred_traj,
            init_noise=init_state,
            sigmas=sde_sample_results.sde_infos.sigmas,
            sigma_maxs=sde_sample_results.sigma_maxs,
            group_trajs_each_step=sde_sample_results.denoise_trajs_each_step,
            gt_rewards=sde_sample_results.gt_rewards,
            shortcut_outputs = shortcut_outputs if self.rule == "trained" and self.shortcut_bs > 0 else None
        object_score = object_score,
            gt_flow=gt_flow,
            pred_traj=pred_traj,
            init_noise=init_state,
            sigmas=sde_sample_results.sde_infos.sigmas,
            std=sde_sample_results.sde_infos.std_dev_t,
            ego_gt_with_groups=ego_gt_with_groups,
            x_t_records=sde_sample_results.x_t_records,
            sample_traj=sde_sample_results.denoise_trajs_each_step[..., -1, :2*(self.traj_len+self.len_10m_path)],
            log_pi=sde_sample_results.log_probs,
    def _update_train_iter(self):
        self.train_iter += 1
    def _reset_train_iter(self):
        self.train_iter = 0
    def get_train_iter(self):
        return self.train_iter
    def _traj_interpolates(self, trajs):
        topk = 1
        b = trajs.shape[0]
        traj_new = torch.zeros((b, 50, 2), dtype=torch.float32, device=trajs.device) # [1, 50 , 2]
    def sample_sde(self, conditions, neg_gt_traj_with_path, gt_guidance_tag=None, groups=5, tps_train=None, denoise_trajs=None, return_logpi=True, sigma_maxs=None, x_t_old = None):
        final_sde_results = edict()
        conditions=conditions,
            groups=groups
        )
        tokens,ego_condition, object_feat, final_condition, gt_traj_mask, ego_feat_with_mm, obj_feat_with_mm = update_conditions
        if tps_train is None:
        tps_train = self.denoising_num # t
        x_t_records = [x_t]
        log_probs = []
        obj_log_probs = []
        ego_traj_len = 2*(self.traj_len + self.len_10m_path)
        denoise_trajs_each_step = []
        pred_mu_flow_each_step = []
        pred_obj_flow_each_step = []
        # select sde_solver
        if self.sde_solver_type == "Euler": # flow_grpo sde_solver
        drift_enable=self.drift_enable,
            shift=1.0, # default
        raise NotImplementedError('Undefined SDE Solver !!!')
        ## sde_solvers统一时间调度
        sigmas, std_dev_t = sde_scheduler.get_sde_sigma_infos(flip_flag=True)
        sigmas = sigmas[..., :tps_train]
        z = sample[:, :ego_traj_len].reshape(b, ego_traj_len//2, 2)
        obj_z = sample[:, ego_traj_len:].reshape(b, self.obj_num, self.traj_len, 2)
        t = sigmas[..., i].unsqueeze(-1).reshape(b, -1)
        ego_condition=ego_condition,
            object_flow=obj_z,
            object_condition=object_feat,
            final_condition=final_condition,
            all_traj_flow_mask=gt_traj_mask,
            ego_feat_with_mm=ego_feat_with_mm,
            obj_feat_with_mm=obj_feat_with_mm,
            )
        group_b = pred_mu_flow.shape[0]
        return_ode=False
        )
        if self.gt_guidance and i+1 < tps_train:
        x_t = x_t.reshape(b//groups, groups, ego_traj_len + obj_traj_len)
        t_to = sigmas[0, i+1]
        noise = torch.rand_like(drift_mean[::groups][:, :ego_traj_len])
        x_t[gt_guidance_tag, 0, :ego_traj_len] = t_to * neg_gt_traj_with_path[gt_guidance_tag] + (1.0 - t_to) * noise[gt_guidance_tag]
        x_t = x_t.reshape(-1, ego_traj_len + obj_traj_len)
        if self.rule == "policy_old":
        samples=cur_sample[:, :ego_traj_len],
            means=drift_mean[:, :ego_traj_len],
            noise_std=noise_std,
            dt=sde_scheduler.get_dt() if self.sde_solver_type == "Euler" else None
        noise_std=noise_std,
            dt=sde_scheduler.get_dt() if self.sde_solver_type == "Euler" else None
        )
        log_probs.append(log_pi)
        log_probs = torch.stack(log_probs, dim=1) if (return_logpi and not self.deterministic) else []
        entropys = torch.stack(entropys, dim=1) if (return_logpi and not self.deterministic) else []
        obj_log_probs = torch.stack(obj_log_probs, dim=1) if (return_logpi and not self.deterministic) else []
        final_sde_results.sde_infos = sde_infos
    def get_sde_denoise_init_traj(self, z0=None, obj_num=0, device=None):
        """
        for sde sampler
        z0: init_noise
        """
        # tps = None # debug
        A, G, K = z0.shape
        x_t = z0
        x_t = x_t.reshape(A*G, -1)
    def get_init_state_z0(self, bs, mode, obj_num):
        if mode == "random":
        return torch.randn((bs, self.groups, 2 * (self.traj_len+self.len_10m_path + obj_num * self.traj_len)), device=self.device)
        elif mode == "zero":
        return torch.zeros((bs, self.groups, 2 * (self.traj_len+self.len_10m_path + obj_num * self.traj_len)), device=self.device)
        else:
        raise NotImplementedError("Unsupported init z0 mode ... ...")
    def _conditions_cache_improve_preprocess(self, conditions, groups=1):
        if not isinstance(conditions, (list, tuple)):
        raise TypeError("Error: 'conditions' must be a List!")
        # kv preprocess for cache & inference time-saving
        tokens, ego_condition, object_feat, final_condition, gt_traj_mask, ego_feat_with_mm, obj_feat_with_mm = conditions
        if groups > 1:
        # rebatch conditions -> rep group times
        conditions = [tokens, ego_condition, object_feat, final_condition, gt_traj_mask, ego_feat_with_mm, obj_feat_with_mm]
        tokens, ego_condition, object_feat, final_condition, gt_traj_mask, ego_feat_with_mm, obj_feat_with_mm = \
        self.batch_process_multiple_tensors(conditions, groups)
        update_conditions = [tokens, ego_condition, object_feat, final_condition, gt_traj_mask, ego_feat_with_mm, obj_feat_with_mm]
        return update_conditions
    def _init_params(self):
        pretrain_head_dict = OrderedDict()
        print("====== load the state dict from the pre-trained head =======islearn=",
            self.rule,'pretrained_param',
            pretrained_param['diffusion_policy_head.diffusion_decoder.time_embed.mlp.0.bias'].cpu().numpy()[:2],
            "head_p", self.state_dict()['diffusion_decoder.time_embed.mlp.0.bias'].cpu().numpy()[:2])
    def _update_policy_old_params(self):
        if (self.count+1) % self.target_update_period == 0 and self.rule == "trained": # policy_new先upload
        self._update_params(role="trained")
        if self.count > 0 and self.count % self.target_update_period == 0 and self.rule == "policy_old":
        # if self.policy_old_cnt % (self.target_update_period) == 0:
    def _update_params(self, role):
        # 参数更新
        if role =='trained': #evaluate network
        # trained上传参数
        DiffusionPolicyDense_RL.trained_head_state_dict = self.state_dict().copy()
        #soft update
        if self.mode == "hard":
    def get_sde_gaussian_logpdf(self,
            samples: torch.Tensor,
            means: torch.Tensor,
            noise_std: torch.Tensor,
            dt: float = 0.1,
        """
        gaussian log pdf
        """
        std = noise_std * torch.sqrt(dt) if self.sde_solver_type == "Euler" else (noise_std)
        term3 = - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi))+ eps)
        log_prob_elem = term1 + term2 + term3 # shape [B, ...]
        dims = list(range(1, log_prob_elem.ndim))
        dist = Normal(means, std)
        log_prob = dist.log_prob(samples+eps).mean(dim=-1)
        entropy = -log_prob
        return log_prob, entropy
    def batch_process_multiple_tensors(self, tensor_list, repeat_times):
        """
        扩展batch任意list
        """
        if type(tensor_list) is list:
    def _repeat_batch(x, repeat_times):
        """
        conditions rebatch
        """
        B, *rest = x.shape
        x = x.unsqueeze(1)    # shape: (B, 1, *rest)
        reps = [1, repeat_times] + [1] * len(rest)
        return x.view(B * repeat_times, *rest)
    def _viz_tool_for_sde_sampler(self, labels, ego_condition, group_trajs_each_step, ego_gt_with_groups, sigma_maxs=None):
        """
        Denoise 可视化工具
        - 横轴 --- Groups数模态数
        - 纵轴 --- Denoise步数
        """
        # debug可视化Groups条轨迹的Denoise过程
import matplotlib.pyplot as plt
        _, Tps, _ = group_trajs_each_step.size()
        gt = gt.detach().cpu().numpy()
        fig, axes = plt.subplots(nrows=T, ncols=groups, figsize=(4*groups, 4*T))
        # tmax
        t_max = sigma_maxs.view(b, groups, -1)[select_BS]
        t_max = t_max.detach().cpu().numpy()
        for g in range(groups):
        ax.set_title(f'G{g} T{t}-Tmax{sigma_t}')
        else:
        ax.set_title(f'G{g} T{t}')
import os
    def s_sample(self, pred_traj_from_scratch, ego_gt_traj, sample_s_num=5):
        total_len = self.traj_len + self.len_10m_path
        device = pred_traj_from_scratch.device
        pred_s = torch.cumsum(xy_dis, dim=-1)
        gt = ego_gt_traj
        gt = torch.concat((torch.zeros((b,1,2),device=device), gt), dim=1)
        if pred_s[i,-1] > gt_s[i,-1]:
        gt_s[i] = 0.9 * gt_s[i]
        pred_s_diff[i] = pred_s[i] * 1.1 - gt_s[i]
        for i in range(1, sample_s_num):
        sample_s[:, i] = gt_s + pred_s_diff / (sample_s_num - 1) * i
        index_back = index_front + 1
        i = np.array(range(b)).reshape(-1,1)
        s_percent = torch.where(s_percent == np.inf, torch.tensor(0, dtype=torch.float32).to(dist_to_ref.device), s_percent)
        new_point = front_point + (back_point - front_point) * s_percent.unsqueeze(-1) # [n, a*t, d]
        new_traj = new_point.reshape(b, sample_s_num, 25, 2)
        return new_traj
    def _update_sde_sample_results(self, sde_sample_results, sde_sample_results_from_memory):
        for key in sde_sample_results_from_memory.keys():
        sde_sample_results[key] = sde_sample_results_from_memory[key]
        return sde_sample_results
    def get_train_tuple(self, z0=None, z1=None, device=None, t=None):
        a, k = z0.shape
        if t is None:
        t = torch.rand((a, 1), device=device)
    def train_shortcut_pipe(self, shortcut_inputs: edict, utokens: edict):
        shortcut_pipe_feats = edict()
        init_noise = torch.randn_like(shortcut_inputs.ego_gt_traj)
        obj_init_noise = torch.randn_like(shortcut_inputs.obj_gt_traj)
        final_condition=utokens.final_condition,
            object_feat=shortcut_inputs.object_feat,
            ego_feat_with_mm=utokens.ego_feat_with_mm,
            obj_feat_with_mm=utokens.obj_feat_with_mm
        )
        ego_condition_st = utokens.ego_condition[shortcut_index].clone()
        tokens_st = utokens.tokens[shortcut_index].clone()
        ego_feat_with_mm_st = utokens.ego_feat_with_mm[shortcut_index].clone() if utokens.ego_feat_with_mm is not None else utokens.ego_feat_with_mm
        object_flow=obj_noise_traj_t[bs_ori:],
            object_condition=object_feat_st,
            final_condition=final_condition_st,
            all_traj_flow_mask=gt_traj_mask_st,
            ego_feat_with_mm=ego_feat_with_mm_st,
            obj_feat_with_mm=obj_feat_with_mm_st
        )
        _, gt_flow_shortcut = torch.split(gt_flow, (bs_ori, self.shortcut_bs), dim=0)
        _, obj_gt_flow_shortcut = torch.split(obj_gt_flow, (bs_ori, self.shortcut_bs), dim=0)
        shortcut_pipe_feats.ego_condition_st = ego_condition_st
        shortcut_pipe_feats.predict_flow_shortcut = predict_flow_st[:, :self.traj_len]
    def sample_ode_ori(self, conditions, noise_traj_t, obj_noise_traj_t):
        tokens, ego_condition, obj_noise_traj_t, object_feat, final_condition, gt_traj_mask, ego_feat_with_mm, obj_feat_with_mm = conditions
        b = ego_condition.shape[0]
        dt = torch.ones((b, 1), device=ego_condition.device)/self.tps_train
        n = self.tps_train
        t = torch.ones((b, 1), device=ego_condition.device) * i/n
        predict_flow, predict_obj_flow = self.diffusion_decoder(
            time=t,
            dt=dt,
            tokens=tokens,
            flow=z,
            ego_condition=ego_condition,
            object_flow=z_obj,
            object_condition=object_feat,
            final_condition=final_condition,
            all_traj_flow_mask=gt_traj_mask,
            ego_feat_with_mm=ego_feat_with_mm,
            obj_feat_with_mm=obj_feat_with_mm
        )
        z = z + predict_flow * dt.unsqueeze(-1)
        return z, z_obj
