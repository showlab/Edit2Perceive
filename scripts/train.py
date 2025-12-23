import torch, os, json
from torch.utils.data import ConcatDataset
import numpy as np
from models.utils import load_state_dict,DiffusionTrainingModule, ModelLogger, launch_training_task, flux_parser, parse_flux_model_configs, find_latest_checkpoint
from pipelines.flux_image_new import FluxImagePipeline, ModelConfig
from lora.flux_lora import FluxLoRAConverter
from models.unified_dataset import UnifiedDataset
from utils.mixed_sampler import MixedBatchSampler
from utils.visualize import visualize_sample,prepare_image
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# torch.autograd.set_detect_anomaly(True)
class FluxTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp", lora_rank=32, lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        multi_res_noise=False,
        deterministic_flow=False,
        extra_loss=None,
        depth_normalization="log",
        matting_prompt=None,
    ):
        super().__init__()
        # Load models
        # model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)
        model_configs = parse_flux_model_configs(root_path=model_paths)
        self.pipe = FluxImagePipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cuda", model_configs=model_configs)
        # Update matting_prompt if provided
        if matting_prompt is not None and hasattr(self.pipe.dit, 'coord_encoder'):
            self.pipe.dit.coord_encoder.matting_prompt = matting_prompt
            print(f"Updated coord_encoder.matting_prompt to: {matting_prompt}")

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.multi_res_noise = multi_res_noise        
        self.deterministic_flow = deterministic_flow
        self.extra_loss = extra_loss
        self.depth_normalization = depth_normalization
    def forward_preprocess(self, data):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": ""}
        
        # CFG-unsensitive parameters
        if data["image"].ndim == 3:
            data["image"] = data["image"].unsqueeze(0)
        if self.deterministic_flow:
            timestep = self.pipe.scheduler.timesteps[0].repeat(data["image"].shape[0],).to(self.pipe.device,dtype=self.pipe.torch_dtype)
        else:
            timestep = self.pipe.scheduler.timesteps[torch.randint(0, self.pipe.scheduler.num_train_timesteps, size=(data["image"].shape[0],))].to(self.pipe.device,dtype=self.pipe.torch_dtype)
        
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_image": data["image"],
            "mask": data.get("mask", None),
            "height": data["image"].shape[2],
            "width": data["image"].shape[3],
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "seed": 42,
            "cfg_scale": 1,
            "embedded_guidance": 1,
            "t5_sequence_length": 512,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "multi_res_noise": self.multi_res_noise,
            "timestep": timestep,
            "deterministic_flow": self.deterministic_flow,
            "extra_loss": self.extra_loss,
            "depth_normalization": self.depth_normalization
        }
        
        # Extra inputs
        controlnet_input = {}
        for extra_input in self.extra_inputs:
            if extra_input.startswith("controlnet_"):
                controlnet_input[extra_input.replace("controlnet_", "")] = data[extra_input]
            else:
                inputs_shared[extra_input] = data[extra_input]
        
        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}
    
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        return loss



if __name__ == "__main__":
    parser = flux_parser()
    args = parser.parse_args()
    datasets_ls = []
    metadata_paths = args.dataset_metadata_path.split(",")
    dataset_base_paths = args.dataset_base_path.split(",")
    heights = [args.height]
    widths = [args.width]
    if len(metadata_paths) == 2:
        # heights = [768,352]
        # widths  = [1024,1216]
        heights = [args.height, args.height]
        widths  = [args.width, args.width]
    elif len(metadata_paths) == 3:
        heights = [args.height, args.height, args.height]
        widths  = [args.width, args.width, args.width]
    elif len(metadata_paths) == 4:
        heights = [args.height, args.height, args.height, args.height]
        widths  = [args.width, args.width, args.width, args.width]
    if "depth" in metadata_paths[0]:
        args.task = "depth"
        print("!!! Doing depth task !!!")
    elif "normal" in metadata_paths[0]:
        args.task = "normal"
        print("!!! Doing normal task !!!")
    elif "matting" in metadata_paths[0]:
        args.task = "matting"
        print("!!! Doing matting task !!!")
    else:
        raise ValueError("Cannot infer task from metadata path; please include 'depth' or 'normal' in the path.")
    if args.using_pdf:
        print("!!! Using PDF operator for image loading !!!")
        pdf = np.load("depth_mapping_lookup_table.npz")
    else:
        pdf = None
    for dataset_base_path,metadata_path,height,width in zip(dataset_base_paths,metadata_paths,heights,widths):
        dataset = UnifiedDataset(
            base_path=dataset_base_path,
            metadata_path=metadata_path,
            repeat=args.dataset_repeat,
            data_file_keys=args.data_file_keys.split(","),
            main_data_operator=UnifiedDataset.default_image_operator(
                base_path=dataset_base_path,
                max_pixels=args.max_pixels,
                height=height,
                width=width,
                height_division_factor=32,
                width_division_factor=32,
                using_log=args.using_log,
                using_sqrt=args.using_sqrt,
                using_sqrt_disp=args.using_sqrt_disp,
                using_pdf=args.using_pdf,
                pdf=pdf,
                with_mask=args.with_mask,
            ),
            special_operator_map=["mask","prompt"] if args.with_mask else ["prompt"],
            default_caption = args.default_caption,
            matting_prompt=args.matting_prompt if args.task=="matting" else None,
            use_coor_input=args.use_coor_input if args.task=="matting" else False,
            use_camera_intrinsics=args.use_camera_intrinsics,
            # use_attn_mask=args.use_attn_mask if args.task=="matting" else False,
        )
        print(f"Loading {metadata_path} with {len(dataset)} items of size Height {height} x Width {width}")
        # Example data item logging (handle optional keys safely)
        try:
            example = dataset[0]
            img_shape = example['image'].shape if 'image' in example else None
            if isinstance(example.get('kontext_images', None), list):
                ktx_shape = [ktx.shape for ktx in example['kontext_images']]
            else:
                ktx_shape = example.get('kontext_images', None).shape if example.get('kontext_images', None) is not None else None
            # ktx_shape = example.get('kontext_images', None).shape if example.get('kontext_images', None) is not None else None
            mask_obj = example.get('mask', None)
            mask_shape = mask_obj.shape if mask_obj is not None else None
            prompt_val = example.get('prompt', '')
            print(f"Example data item: image={img_shape}, kontext_images={ktx_shape}, mask={mask_shape}, prompt={prompt_val[:80]}")
        except Exception as e:
            print(f"Failed to print example data item: {e}")
        datasets_ls.append(dataset)
    print(f"Total datasets loaded: {len(datasets_ls)}")
    if len(datasets_ls) > 1:  
        dataset = ConcatDataset(datasets_ls)
        if args.task=="depth":
            prob=[0.9, 0.1]
        elif args.task=="normal":
            prob=[0.5,0.45,0.05]
        elif args.task=="matting":
            if len(datasets_ls)==2:
                prob=[0.5,0.5]
            elif len(datasets_ls)==3:
                prob=[0.02,0.65,0.33]
            elif len(datasets_ls)==4:
                prob=[0.22,0.22,0.3,0.26]
        mixed_sampler = MixedBatchSampler(datasets_ls, shuffle=True, batch_size=args.batch_size, drop_last=True, prob=prob)
        print(f"using {len(datasets_ls)} datasets, total length: {len(dataset)} with PROB:{prob}")
    else:
        dataset = datasets_ls[0]
        mixed_sampler = None
    print(args.eval_file_list)
    if args.eval_file_list:
        with open(args.eval_file_list, "r") as f:
            if args.task == "depth" or args.task == "normal":
                eval_file_list = [line.strip().split()[0] for line in f]
                base_dir = f"/mnt/nfs/workspace/syq/dataset/Eval/{args.task}/nyuv2"
            elif args.task == "matting":
                eval_file_list = [line.strip().split()[0] for line in f]
                base_dir = "/mnt/nfs/workspace/syq/dataset/matting/P3M-10k"
            else:
                raise ValueError(f"Unknown task {args.task}")
            print(f"Loaded {len(eval_file_list)} evaluation files.")
        
        eval_file_list = [os.path.join(base_dir, x) if not os.path.isabs(x) else x for x in eval_file_list]
        args.eval_file_list = eval_file_list
        print(f"top 5 evaluation files: {eval_file_list[:5]}")
        model = FluxTrainingModule(
            model_paths=args.model_paths,
            model_id_with_origin_paths=args.model_id_with_origin_paths,
            trainable_models=args.trainable_models,
            lora_base_model=args.lora_base_model,
            lora_target_modules=args.lora_target_modules,
            lora_rank=args.lora_rank,
            lora_checkpoint=find_latest_checkpoint(args.output_path) if (args.resume and args.lora_base_model is not None) else None,
            use_gradient_checkpointing=args.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
            extra_inputs=args.extra_inputs,
            multi_res_noise=args.multi_res_noise,
            deterministic_flow=args.deterministic_flow,
            extra_loss=args.extra_loss,
            depth_normalization=args.depth_normalization,
            matting_prompt=args.matting_prompt if args.task == "matting" else None,
        )
    if args.resume and os.path.isdir(args.output_path):
        latest_ckpt = find_latest_checkpoint(args.output_path)
        if latest_ckpt is not None:
            if args.lora_base_model is None:
                state_dict = load_state_dict(latest_ckpt)
                model.pipe.dit.load_state_dict(state_dict,strict=False)
                del state_dict
            args.resume_steps = int(latest_ckpt.split("step-")[-1].split(".")[0])
            print(f"Resumed training from step {args.resume_steps}")
            torch.cuda.empty_cache()
        else:
            print(f"No checkpoint found in {args.output_path}, starting fresh training.")
            args.resume_steps = 0
    else:
        args.resume_steps = 0
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        state_dict_converter=FluxLoRAConverter.align_to_opensource_format if args.align_to_opensource_format else lambda x:x,
        args=args
    )
    launch_training_task(dataset, model, model_logger, dataset_sampler=mixed_sampler, args=args)
