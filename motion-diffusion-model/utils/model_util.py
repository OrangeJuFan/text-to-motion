import torch
from model.mdm import MDM
from model.lora_adapter import add_lora_to_mdm
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from utils.parser_util import get_cond_mode
from data_loaders.humanml_utils import HML_EE_JOINT_NAMES

def load_model_wo_clip(model, state_dict):
    # assert (state_dict['sequence_pos_encoder.pe'][:model.sequence_pos_encoder.pe.shape[0]] == model.sequence_pos_encoder.pe).all()  # TEST
    # assert (state_dict['embed_timestep.sequence_pos_encoder.pe'][:model.embed_timestep.sequence_pos_encoder.pe.shape[0]] == model.embed_timestep.sequence_pos_encoder.pe).all()  # TEST
    
    # 检查模型是否是 PeftModel（使用 LoRA）
    from peft import PeftModel
    is_peft_model = isinstance(model, PeftModel)
    
    # 删除不需要的键
    keys_to_delete = []
    if 'sequence_pos_encoder.pe' in state_dict:
        keys_to_delete.append('sequence_pos_encoder.pe')
    if 'embed_timestep.sequence_pos_encoder.pe' in state_dict:
        keys_to_delete.append('embed_timestep.sequence_pos_encoder.pe')
    
    for key in keys_to_delete:
        del state_dict[key]
    
    # 如果模型是 PeftModel，需要将检查点中的键名添加 base_model.model. 前缀
    if is_peft_model:
        print("检测到 PeftModel，将检查点键名添加 'base_model.model.' 前缀...")
        remapped_state_dict = {}
        for key, value in state_dict.items():
            # 检查点中的键名是直接的模型参数名，需要添加 base_model.model. 前缀
            new_key = f"base_model.model.{key}"
            remapped_state_dict[new_key] = value
        state_dict = remapped_state_dict
    
    # 获取模型期望的键名
    model_keys = set(model.state_dict().keys())
    
    # 过滤掉非模型参数的键（如优化器状态、步数等）
    # 只保留模型参数的键
    filtered_state_dict = {}
    unexpected_keys = []
    
    for key, value in state_dict.items():
        if key in model_keys:
            filtered_state_dict[key] = value
        else:
            unexpected_keys.append(key)
    
    # 加载模型参数
    missing_keys, remaining_unexpected = model.load_state_dict(filtered_state_dict, strict=False)
    
    # 打印调试信息
    if unexpected_keys:
        print(f"警告: 检查点中包含 {len(unexpected_keys)} 个非模型参数键（已忽略）:")
        for key in unexpected_keys[:10]:  # 只打印前10个
            print(f"  - {key}")
        if len(unexpected_keys) > 10:
            print(f"  ... 还有 {len(unexpected_keys) - 10} 个键")
    
    if remaining_unexpected:
        print(f"警告: 模型加载后仍有 {len(remaining_unexpected)} 个意外键:")
        for key in remaining_unexpected[:10]:
            print(f"  - {key}")
        if len(remaining_unexpected) > 10:
            print(f"  ... 还有 {len(remaining_unexpected) - 10} 个键")
    
    # 检查 missing_keys 是否合理
    # 正常的缺失键包括：
    # 1. clip_model.* - CLIP 模型权重，从 CLIP 库加载，不保存在检查点中
    # 2. sequence_pos_encoder.* - 位置编码，固定值，不需要加载
    # 3. lora_* - LoRA 层权重，初始化为零，不需要从检查点加载
    if missing_keys:
        # 过滤出真正异常的缺失键
        normal_missing = [k for k in missing_keys if (
            k.startswith('clip_model.') or 
            'sequence_pos_encoder' in k or 
            'lora_' in k or
            k.startswith('base_model.model.clip_model.')  # PeftModel 中的 clip_model
        )]
        invalid_missing = [k for k in missing_keys if k not in normal_missing]
        
        if normal_missing:
            print(f"信息: 有 {len(normal_missing)} 个正常的缺失键（CLIP/位置编码/LoRA，将从其他来源加载或使用默认值）")
        
        if invalid_missing:
            print(f"警告: 有 {len(invalid_missing)} 个异常的缺失模型参数键:")
            for key in invalid_missing[:10]:
                print(f"  - {key}")
            if len(invalid_missing) > 10:
                print(f"  ... 还有 {len(invalid_missing) - 10} 个键")
            print("  这些键的缺失可能影响模型性能，请检查检查点是否完整")
        else:
            print(f"✓ 所有缺失的键都是正常的（CLIP/位置编码/LoRA），模型加载成功")


def create_model_and_diffusion(args, data):
    model = MDM(**get_model_args(args, data))

    # 可选：在 MDM 上挂载 LoRA 适配器，用于显存友好的微调
    if getattr(args, "use_lora", False):
        model = add_lora_to_mdm(
            model,
            r=getattr(args, "lora_r", 128),
            lora_alpha=getattr(args, "lora_alpha", 16),
            lora_dropout=getattr(args, "lora_dropout", 0.0),
            target_spec=getattr(args, "lora_target_spec", "all"),
        )

    diffusion = create_gaussian_diffusion(args)
    return model, diffusion


def get_model_args(args, data):

    # default args
    clip_version = 'ViT-B/32'
    action_emb = 'tensor'
    cond_mode = get_cond_mode(args)
    if hasattr(data.dataset, 'num_actions'):
        num_actions = data.dataset.num_actions
    else:
        num_actions = 1

    # SMPL defaults
    data_rep = 'rot6d'
    njoints = 25
    nfeats = 6
    all_goal_joint_names = []

    if args.dataset == 'humanml':
        data_rep = 'hml_vec'
        njoints = 263
        nfeats = 1
        all_goal_joint_names = ['pelvis'] + HML_EE_JOINT_NAMES
    elif args.dataset == 'kit':
        data_rep = 'hml_vec'
        njoints = 251
        nfeats = 1

    # Compatibility with old models
    if not hasattr(args, 'pred_len'):
        args.pred_len = 0
        args.context_len = 0
    
    emb_policy = args.__dict__.get('emb_policy', 'add')
    multi_target_cond = args.__dict__.get('multi_target_cond', False)
    multi_encoder_type = args.__dict__.get('multi_encoder_type', 'multi')
    target_enc_layers = args.__dict__.get('target_enc_layers', 1)

    return {'modeltype': '', 'njoints': njoints, 'nfeats': nfeats, 'num_actions': num_actions,
            'translation': True, 'pose_rep': 'rot6d', 'glob': True, 'glob_rot': True,
            'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu", 'data_rep': data_rep, 'cond_mode': cond_mode,
            'cond_mask_prob': args.cond_mask_prob, 'action_emb': action_emb, 'arch': args.arch,
            'emb_trans_dec': args.emb_trans_dec, 'clip_version': clip_version, 'dataset': args.dataset,
            'text_encoder_type': args.text_encoder_type,
            'pos_embed_max_len': args.pos_embed_max_len, 'mask_frames': args.mask_frames,
            'pred_len': args.pred_len, 'context_len': args.context_len, 'emb_policy': emb_policy,
            'all_goal_joint_names': all_goal_joint_names, 'multi_target_cond': multi_target_cond, 'multi_encoder_type': multi_encoder_type, 'target_enc_layers': target_enc_layers,
            }



def create_gaussian_diffusion(args):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = args.diffusion_steps
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]
    
    if hasattr(args, 'lambda_target_loc'):
        lambda_target_loc = args.lambda_target_loc
    else:
        lambda_target_loc = 0.

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=args.lambda_vel,
        lambda_rcxyz=args.lambda_rcxyz,
        lambda_fc=args.lambda_fc,
        lambda_target_loc=lambda_target_loc,
    )

def load_saved_model(model, model_path, use_avg: bool=False):  # use_avg_model
    state_dict = torch.load(model_path, map_location='cpu')
    
    # 打印检查点的键，用于调试
    print(f"检查点包含的顶级键: {list(state_dict.keys())}")
    
    # Use average model when possible
    if use_avg and 'model_avg' in state_dict.keys():
    # if use_avg_model:
        print('loading avg model')
        state_dict = state_dict['model_avg']
    else:
        if 'model' in state_dict:
            print('loading model without avg')
            state_dict = state_dict['model']
        else:
            print('checkpoint has no avg model, loading as usual.')
            # 如果检查点中没有 'model' 键，可能整个检查点就是模型权重
            # 检查是否包含模型参数（以常见的模型层名开头）
            if not any(key.startswith(('seqTransEncoder.', 'embed_timestep.', 'embed_text.', 'output_process.')) for key in state_dict.keys()):
                # 可能检查点结构不同，尝试直接使用
                print('警告: 检查点结构可能不同，尝试直接加载...')
    
    load_model_wo_clip(model, state_dict)
    return model