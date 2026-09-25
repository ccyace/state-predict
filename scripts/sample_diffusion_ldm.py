import argparse, os, sys, gc, glob, datetime, yaml
import logging
import time
import numpy as np
from tqdm import trange
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from PIL import Image

import torch
import torch.nn as nn

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_TAMING = os.path.join(_ROOT, "src", "taming-transformers")
if _TAMING not in sys.path:
    sys.path.insert(0, _TAMING)
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.util import instantiate_from_config

from qdiff import (
    QuantModel, QuantModule, BaseQuantBlock, 
    block_reconstruction, layer_reconstruction,
)
from qdiff.adaptive_rounding import AdaRoundQuantizer
from qdiff.quant_layer import UniformAffineQuantizer
from qdiff.utils import resume_cali_model, get_train_samples

logger = logging.getLogger(__name__)

rescale = lambda x: (x + 1.) / 2.

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def custom_to_np(x):
    # saves the batch in adm style as in https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample


def logs2pil(logs, keys=["sample"]):
    imgs = dict()
    for k in logs:
        try:
            if len(logs[k].shape) == 4:
                img = custom_to_pil(logs[k][0, ...])
            elif len(logs[k].shape) == 3:
                img = custom_to_pil(logs[k])
            else:
                print(f"Unknown format for key {k}. ")
                img = None
        except:
            img = None
        imgs[k] = img
    return imgs


@torch.no_grad()
def convsample(model, shape, return_intermediates=True,
               verbose=True,
               make_prog_row=False):


    if not make_prog_row:
        return model.p_sample_loop(None, shape,
                                   return_intermediates=return_intermediates, verbose=verbose)
    else:
        return model.progressive_denoising(
            None, shape, verbose=True
        )


@torch.no_grad()
def convsample_ddim(
    model,
    steps,
    shape,
    eta=1.0,
    noise_corrector=None,
    vsc_sampler_kwargs=None,
    conditioning=None,
    cfg_scale=1.0,
    uc=None,
):
    vsc_sampler_kwargs = vsc_sampler_kwargs or {}
    ddim = DDIMSampler(model, noise_corrector=noise_corrector, **vsc_sampler_kwargs)
    bs = shape[0]
    shape = shape[1:]
    samples, intermediates = ddim.sample(
        steps,
        batch_size=bs,
        shape=shape,
        eta=eta,
        verbose=False,
        conditioning=conditioning,
        unconditional_guidance_scale=cfg_scale,
        unconditional_conditioning=uc,
    )
    return samples, intermediates


@torch.no_grad()
def convsample_dpm(model, steps, shape, eta=1.0
                    ):
    dpm = DPMSolverSampler(model)
    bs = shape[0]
    shape = shape[1:]
    samples, intermediates = dpm.sample(steps, batch_size=bs, shape=shape, eta=eta, verbose=False,)
    return samples, intermediates


@torch.no_grad()
def make_convolutional_sample(
    model,
    batch_size,
    vanilla=False,
    custom_steps=None,
    eta=1.0,
    dpm=False,
    noise_corrector=None,
    vsc_sampler_kwargs=None,
    conditioning=None,
    cfg_scale=1.0,
    uc=None,
):
    log = dict()
    shape = [
        batch_size,
        model.model.diffusion_model.in_channels,
        model.model.diffusion_model.image_size,
        model.model.diffusion_model.image_size,
    ]
    t0 = time.time()
    if vanilla:
        sample, progrow = convsample(model, shape, make_prog_row=True)
    elif dpm:
        logger.info(f"Using DPM sampling with {custom_steps} sampling steps and eta={eta}")
        sample, intermediates = convsample_dpm(model, steps=custom_steps, shape=shape, eta=eta)
    else:
        sample, intermediates = convsample_ddim(
            model,
            steps=custom_steps,
            shape=shape,
            eta=eta,
            noise_corrector=noise_corrector,
            vsc_sampler_kwargs=vsc_sampler_kwargs,
            conditioning=conditioning,
            cfg_scale=cfg_scale,
            uc=uc,
        )
    t1 = time.time()
    x_sample = model.decode_first_stage(sample)
    log["sample"] = x_sample
    log["time"] = t1 - t0
    log["throughput"] = sample.shape[0] / (t1 - t0)
    logger.info(f"Throughput for this batch: {log['throughput']}")
    return log


def run(
    model,
    logdir,
    batch_size=50,
    vanilla=False,
    custom_steps=None,
    eta=None,
    n_samples=50000,
    nplog=None,
    dpm=False,
    noise_corrector=None,
    vsc_sampler_kwargs=None,
    cfg_scale=1.0,
):
    if vanilla:
        logger.info(f"Using Vanilla DDPM sampling with {model.num_timesteps} sampling steps.")
    else:
        logger.info(f"Using DDIM sampling with {custom_steps} sampling steps and eta={eta}")
        if vsc_sampler_kwargs and vsc_sampler_kwargs.get("vsc_stats") is not None:
            logger.info(
                f"VSC ON absorb={vsc_sampler_kwargs.get('vsc_absorb_strength', 1.0)} "
                f"max_budget_frac={vsc_sampler_kwargs.get('vsc_max_budget_fraction', 0.9)}"
            )

    tstart = time.time()
    n_saved = len(glob.glob(os.path.join(logdir, "*.png"))) - 1
    vsc_sampler_kwargs = vsc_sampler_kwargs or {}

    if model.cond_stage_model is None:
        all_images = []
        logger.info(f"Running unconditional sampling for {n_samples} samples")
        for _ in trange(n_samples // batch_size, desc="Sampling Batches (unconditional)"):
            logs = make_convolutional_sample(
                model,
                batch_size=batch_size,
                vanilla=vanilla,
                custom_steps=custom_steps,
                eta=eta,
                dpm=dpm,
                noise_corrector=noise_corrector,
                vsc_sampler_kwargs=vsc_sampler_kwargs,
            )
            n_saved = save_logs(logs, logdir, n_saved=n_saved, key="sample")
            all_images.extend([custom_to_np(logs["sample"])])
            if n_saved >= n_samples:
                logger.info(f"Finish after generating {n_saved} samples")
                break
        all_img = np.concatenate(all_images, axis=0)
        all_img = all_img[:n_samples]
        shape_str = "x".join([str(x) for x in all_img.shape])
        nppath = os.path.join(nplog, f"{shape_str}-samples.npz")
        np.savez(nppath, all_img)
    else:
        logger.info(
            f"Running class-conditional sampling for {n_samples} samples (CFG scale={cfg_scale})"
        )
        all_images = []
        n_classes = 1000
        n_per_class = max(1, n_samples // n_classes)
        with model.ema_scope():
            for cls in trange(n_classes, desc="Sampling classes"):
                remaining = n_samples - n_saved
                if remaining <= 0:
                    break
                n_cls = min(n_per_class, remaining)
                n_batches = (n_cls + batch_size - 1) // batch_size
                for _ in range(n_batches):
                    b = min(batch_size, n_cls)
                    n_cls -= b
                    xc = torch.tensor(b * [cls], device=model.device)
                    c = model.get_learned_conditioning({model.cond_stage_key: xc})
                    uc = model.get_learned_conditioning(
                        {model.cond_stage_key: torch.full((b,), 1000, device=model.device, dtype=xc.dtype)}
                    )
                    logs = make_convolutional_sample(
                        model,
                        batch_size=b,
                        vanilla=vanilla,
                        custom_steps=custom_steps,
                        eta=eta,
                        dpm=dpm,
                        noise_corrector=noise_corrector,
                        vsc_sampler_kwargs=vsc_sampler_kwargs,
                        conditioning=c,
                        cfg_scale=cfg_scale,
                        uc=uc,
                    )
                    n_saved = save_logs(logs, logdir, n_saved=n_saved, key="sample")
                    all_images.append(custom_to_np(logs["sample"]))
                    if n_saved >= n_samples:
                        break
                if n_saved >= n_samples:
                    break
        all_img = np.concatenate(all_images, axis=0)[:n_samples]
        shape_str = "x".join([str(x) for x in all_img.shape])
        np.savez(os.path.join(nplog, f"{shape_str}-samples.npz"), all_img)

    logger.info(f"sampling of {n_saved} images finished in {(time.time() - tstart) / 60.:.2f} minutes.")


def save_logs(logs, path, n_saved=0, key="sample", np_path=None):
    for k in logs:
        if k == key:
            batch = logs[key]
            if np_path is None:
                for x in batch:
                    img = custom_to_pil(x)
                    imgpath = os.path.join(path, f"{key}_{n_saved:06}.png")
                    img.save(imgpath)
                    n_saved += 1
            else:
                npbatch = custom_to_np(batch)
                shape_str = "x".join([str(x) for x in npbatch.shape])
                nppath = os.path.join(np_path, f"{n_saved}-{shape_str}-samples.npz")
                np.savez(nppath, npbatch)
                n_saved += npbatch.shape[0]
    return n_saved


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume_base",
        type=str,
        nargs="?",
        help="load fp32 base model from logdir or checkpoint in logdir (will deprecate after direct quantized model loading implemented)",
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        nargs="?",
        help="number of samples to draw",
        default=50000
    )
    parser.add_argument(
        "-e",
        "--eta",
        type=float,
        nargs="?",
        help="eta for ddim sampling (0.0 yields deterministic sampling)",
        default=1.0
    )
    parser.add_argument(
        "-v",
        "--vanilla_sample",
        default=False,
        action='store_true',
        help="vanilla sampling (default option is DDIM sampling)?",
    )
    parser.add_argument(
        "--seed",
        type=int,
        # default=42,
        required=True,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        nargs="?",
        help="extra logdir",
        default="none"
    )
    parser.add_argument(
        "-c",
        "--custom_steps",
        type=int,
        nargs="?",
        help="number of steps for ddim and fast dpm sampling",
        default=50
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="?",
        help="the bs",
        default=10
    )
    # linear quantization configs
    parser.add_argument(
        "--ptq", action="store_true", help="apply post-training quantization"
    )
    parser.add_argument(
        "--quant_act", action="store_true", 
        help="if to quantize activations when ptq==True"
    )
    parser.add_argument(
        "--weight_bit",
        type=int,
        default=8,
        help="int bit for weight quantization",
    )
    parser.add_argument(
        "--act_bit",
        type=int,
        default=8,
        help="int bit for activation quantization",
    )
    parser.add_argument(
        "--quant_mode", type=str, default="qdiff", 
        choices=["qdiff"], 
        help="quantization mode to use"
    )
    # qdiff specific configs
    parser.add_argument(
        "--cali_st", type=int, default=1, 
        help="number of timesteps used for calibration"
    )
    parser.add_argument(
        "--cali_batch_size", type=int, default=32, 
        help="batch size for qdiff reconstruction"
    )
    parser.add_argument(
        "--cali_n", type=int, default=1024, 
        help="number of samples for each timestep for qdiff reconstruction"
    )
    parser.add_argument(
        "--cali_iters", type=int, default=20000, 
        help="number of iterations for each qdiff reconstruction"
    )
    parser.add_argument('--cali_iters_a', default=5000, type=int, 
        help='number of iteration for LSQ')
    parser.add_argument('--cali_lr', default=4e-4, type=float, 
        help='learning rate for LSQ')
    parser.add_argument('--cali_p', default=2.4, type=float, 
        help='L_p norm minimization for LSQ')
    parser.add_argument(
        "--cali_running_batch", type=int, default=16,
        help="mini-batch size for activation running-stat pass (lower saves GPU memory)",
    )
    parser.add_argument(
        "--cali_ckpt", type=str,
        help="path for calibrated model ckpt"
    )
    parser.add_argument(
        "--cali_data_path", type=str, default="sd_coco_sample1024_allst.pt",
        help="calibration dataset name"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="resume the calibrated qdiff model"
    )
    parser.add_argument(
        "--resume_w", action="store_true",
        help="resume the calibrated qdiff model weights only"
    )
    parser.add_argument(
        "--cond", action="store_true",
        help="whether to use conditional guidance (class-conditional LDM + CFG when scale>1)",
    )
    parser.add_argument(
        "--scale", type=float, default=1.0,
        help="classifier-free guidance scale (ImageNet cin256 default 3.0)",
    )
    parser.add_argument(
        "--a_sym", action="store_true",
        help="act quantizers use symmetric quantization (empirically helpful in some cases)"
    )
    parser.add_argument(
        "--a_min_max", action="store_true",
        help="act quantizers initialize with min-max (empirically helpful in some cases)"
    )
    parser.add_argument(
        "--running_stat", action="store_true",
        help="use running statistics for act quantizers"
    )
    parser.add_argument(
        "--rs_sm_only", action="store_true",
        help="use running statistics only for softmax act quantizers"
    )
    parser.add_argument(
        "--sm_abit",type=int, default=8,
        help="attn softmax activation bit"
    )
    parser.add_argument(
        "--dpm", action="store_true",
        help="use dpm solver for sampling"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="print out info like quantized model arch"
    )
    parser.add_argument(
        "--enable_learned_noise_corr", action="store_true",
        help="apply learned DeltaEpsNet correction on UNet eps",
    )
    parser.add_argument("--learned_corr_ckpt", type=str, default="")
    parser.add_argument("--learned_corr_t_cut", type=int, default=999)
    parser.add_argument("--learned_corr_alpha", type=float, default=1.0)
    parser.add_argument("--vsc_stats", type=str, default="")
    parser.add_argument(
        "--vsc_var_field",
        type=str,
        default="trimmed_mean",
        choices=["trimmed_mean", "mean", "var_mle"],
    )
    parser.add_argument("--vsc_absorb_strength", type=float, default=1.0)
    parser.add_argument("--vsc_max_budget_fraction", type=float, default=0.9)
    parser.add_argument(
        "--ptq_output_ckpt",
        type=str,
        default="",
        help="when running PTQ (no --resume), also save qnn state_dict here",
    )
    parser.add_argument(
        "--efficientdm_ckpt",
        type=str,
        default="",
        help="EfficientDM full state_dict (e.g. quantw4a4_20steps_efficientdm.pth); "
             "mutually exclusive with --ptq/--resume",
    )
    parser.add_argument(
        "--efficientdm_steps",
        type=int,
        default=20,
        help="TALSQ length baked into EfficientDM ckpt (should match -c)",
    )
    parser.add_argument(
        "--efficientdm_root",
        type=str,
        default="",
        help="path to EfficientDM checkout (or set EFFICIENTDM_HOME)",
    )
    parser.add_argument("--efficientdm_weight_bit", type=int, default=4)
    parser.add_argument("--efficientdm_act_bit", type=int, default=4)
    return parser


def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd,strict=False)
    model.cuda()
    model.eval()
    return model


def load_model(config, ckpt, gpu, eval_mode):
    if ckpt:
        logger.info(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        global_step = pl_sd.get("global_step")
        sd = pl_sd.get("state_dict", pl_sd)
    else:
        sd = None
        global_step = None
    model = load_model_from_config(config.model, sd)

    return model, global_step


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    sys.path.append(os.getcwd())
    command = " ".join(sys.argv)

    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    ckpt = None

    # fix random seed
    seed_everything(opt.seed)

    if not os.path.exists(opt.resume_base):
        raise ValueError("Cannot find {}".format(opt.resume_base))
    if os.path.isfile(opt.resume_base):
        # paths = opt.resume.split("/")
        try:
            logdir = '/'.join(opt.resume_base.split('/')[:-1])
            # idx = len(paths)-paths[::-1].index("logs")+1
            print(f'Logdir is {logdir}')
        except ValueError:
            paths = opt.resume_base.split("/")
            idx = -2  # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
        ckpt = opt.resume_base
    else:
        assert os.path.isdir(opt.resume_base), f"{opt.resume_base} is not a directory"
        logdir = opt.resume_base.rstrip("/")
        ckpt = os.path.join(logdir, "model.ckpt")

    base_configs = sorted(glob.glob(os.path.join(logdir, "config.yaml")))
    opt.base = base_configs

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    gpu = True
    eval_mode = True

    if opt.logdir != "none":
        locallog = logdir.split(os.sep)[-1]
        if locallog == "": locallog = logdir.split(os.sep)[-2]
        print(f"Switching logdir from '{logdir}' to '{os.path.join(opt.logdir, locallog)}'")
        logdir = os.path.join(opt.logdir, locallog)

    logdir = os.path.join(logdir, "samples", now)
    os.makedirs(logdir)
    log_path = os.path.join(logdir, "run.log")
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    print(config)

    logger.info(75 * "=")
    logger.info(f"Host {os.uname()[1]}")
    logger.info("logging to:")
    imglogdir = os.path.join(logdir, "img")
    numpylogdir = os.path.join(logdir, "numpy")

    os.makedirs(imglogdir)
    os.makedirs(numpylogdir)
    logger.info(logdir)
    logger.info(75 * "=")

    model, global_step = load_model(config, ckpt, gpu, eval_mode)
    logger.info(f"global step: {global_step}")
    if hasattr(model, "model_ema"):
        logger.info("Switched to EMA weights")
        model.model_ema.store(model.model.parameters())
        model.model_ema.copy_to(model.model)

    # print(model.model)
    is_cond = opt.cond or (config.model.params.get("cond_stage_config") not in (None, "__is_unconditional__"))

    if opt.efficientdm_ckpt:
        if opt.ptq or opt.resume or opt.resume_w:
            raise ValueError("--efficientdm_ckpt cannot be combined with --ptq/--resume/--resume_w")
        if int(opt.custom_steps) != int(opt.efficientdm_steps):
            logger.warning(
                f"EfficientDM TALSQ length is {opt.efficientdm_steps} but -c/--custom_steps="
                f"{opt.custom_steps}; temporal act scales will mis-align. Prefer -c {opt.efficientdm_steps}."
            )
        from PTQD.imagenet256.efficientdm_adapter import attach_efficientdm
        attach_efficientdm(
            model,
            opt.efficientdm_ckpt,
            num_steps=int(opt.efficientdm_steps),
            weight_bit=int(opt.efficientdm_weight_bit),
            act_bit=int(opt.efficientdm_act_bit),
            efficientdm_root=opt.efficientdm_root,
            device=torch.device("cuda") if gpu else torch.device("cpu"),
        )

    if opt.ptq:
        if opt.quant_mode == 'qdiff':
            a_scale_method = 'mse' if not opt.a_min_max else 'max'
            wq_params = {'n_bits': opt.weight_bit, 'channel_wise': True, 'scale_method': 'mse'}
            aq_params = {
                'n_bits': opt.act_bit, 'symmetric': opt.a_sym, 'channel_wise': False,
                'scale_method': a_scale_method, 'leaf_param': opt.quant_act
            }
            if opt.resume:
                logger.info('Load with min-max quick initialization')
                wq_params['scale_method'] = 'max'
                aq_params['scale_method'] = 'max'
            if opt.resume_w:
                wq_params['scale_method'] = 'max'
            qnn = QuantModel(
                model=model.model.diffusion_model, weight_quant_params=wq_params, act_quant_params=aq_params,
                sm_abit=opt.sm_abit)
            # Official LDM ImageNet W4 ckpts (QuEST/BRECQ-style) keep first/last at 8-bit and
            # disable output-layer act quant; required to avoid yellow collapse / color blowout.
            if opt.resume or opt.resume_w:
                qnn.set_first_last_layer_to_8bit()
                qnn.disable_network_output_quantization()
            qnn.cuda()
            qnn.eval()

            image_size = config.model.params.image_size
            channels = config.model.params.channels
            if opt.resume:
                if is_cond:
                    cali_data = (
                        torch.randn(1, channels, image_size, image_size),
                        torch.randint(0, 1000, (1,)),
                        torch.randn(1, 1, 512),
                    )
                else:
                    cali_data = (torch.randn(1, channels, image_size, image_size), torch.randint(0, 1000, (1,)))
                resume_cali_model(qnn, opt.cali_ckpt, cali_data, opt.quant_act, "qdiff", cond=is_cond)
            else:
                logger.info(f"Sampling data from {opt.cali_st} timesteps for calibration")
                sample_data = torch.load(opt.cali_data_path, map_location="cpu")
                cali_data = get_train_samples(opt, sample_data)
                del sample_data
                gc.collect()
                if is_cond:
                    logger.info(
                        f"Calibration data shape: {cali_data[0].shape} {cali_data[1].shape} {cali_data[2].shape}"
                    )
                    cali_xs, cali_ts, cali_cs = cali_data
                else:
                    logger.info(f"Calibration data shape: {cali_data[0].shape} {cali_data[1].shape}")
                    cali_xs, cali_ts = cali_data
                if opt.resume_w:
                    resume_cali_model(qnn, opt.cali_ckpt, cali_data, False, cond=is_cond)
                else:
                    logger.info("Initializing weight quantization parameters")
                    qnn.set_quant_state(True, False)
                    if is_cond:
                        _ = qnn(cali_xs[:8].cuda(), cali_ts[:8].cuda(), cali_cs[:8].cuda())
                    else:
                        _ = qnn(cali_xs[:8].cuda(), cali_ts[:8].cuda())
                    logger.info("Initializing has done!")

                kwargs = dict(
                    cali_data=cali_data, batch_size=opt.cali_batch_size,
                    iters=opt.cali_iters, weight=0.01, asym=True, b_range=(20, 2),
                    warmup=0.2, act_quant=False, opt_mode='mse', cond=is_cond,
                )

                def recon_model(qmodel):
                    for name, module in qmodel.named_children():
                        if isinstance(module, QuantModule):
                            if module.ignore_reconstruction is True:
                                continue
                            layer_reconstruction(qnn, module, **kwargs)
                        elif isinstance(module, BaseQuantBlock):
                            if module.ignore_reconstruction is True:
                                continue
                            block_reconstruction(qnn, module, **kwargs)
                        else:
                            recon_model(module)

                if not opt.resume_w:
                    logger.info("Doing weight calibration")
                    recon_model(qnn)
                    qnn.set_quant_state(weight_quant=True, act_quant=False)
                    gc.collect()
                    torch.cuda.empty_cache()
                if opt.quant_act:
                    logger.info("Doing activation calibration")
                    qnn.set_quant_state(True, True)
                    run_bs = max(1, opt.cali_running_batch)
                    with torch.no_grad():
                        warm_bs = min(run_bs, cali_xs.size(0))
                        if is_cond:
                            _ = qnn(cali_xs[:warm_bs].cuda(), cali_ts[:warm_bs].cuda(), cali_cs[:warm_bs].cuda())
                        else:
                            _ = qnn(cali_xs[:warm_bs].cuda(), cali_ts[:warm_bs].cuda())
                        if opt.running_stat:
                            logger.info('Running stat for activation quantization')
                            qnn.set_running_stat(True)
                            n_batches = int(cali_xs.size(0) / run_bs)
                            for i in trange(n_batches):
                                sl = slice(i * run_bs, (i + 1) * run_bs)
                                if is_cond:
                                    _ = qnn(cali_xs[sl].cuda(), cali_ts[sl].cuda(), cali_cs[sl].cuda())
                                else:
                                    _ = qnn(cali_xs[sl].cuda(), cali_ts[sl].cuda())
                            qnn.set_running_stat(False)

                    kwargs = dict(
                        cali_data=cali_data, iters=opt.cali_iters_a, act_quant=True,
                        opt_mode='mse', lr=opt.cali_lr, p=opt.cali_p, cond=is_cond,
                    )
                    recon_model(qnn)
                    qnn.set_quant_state(weight_quant=True, act_quant=True)

                ckpt_out = opt.ptq_output_ckpt or os.path.join(logdir, "ckpt.pth")
                logger.info(f"Saving calibrated quantized UNet model -> {ckpt_out}")
                for m in qnn.model.modules():
                    if isinstance(m, AdaRoundQuantizer):
                        m.zero_point = nn.Parameter(m.zero_point)
                        m.delta = nn.Parameter(m.delta)
                    elif isinstance(m, UniformAffineQuantizer) and opt.quant_act:
                        if m.zero_point is not None and not torch.is_tensor(m.zero_point):
                            m.zero_point = nn.Parameter(torch.tensor(float(m.zero_point)))
                os.makedirs(os.path.dirname(os.path.abspath(ckpt_out)) or ".", exist_ok=True)
                torch.save(qnn.state_dict(), ckpt_out)

            model.model.diffusion_model = qnn

    noise_corrector = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if getattr(opt, "enable_learned_noise_corr", False):
        from noise_eps_corr.learned_noise_corrector import load_learned_corrector
        if not opt.learned_corr_ckpt:
            raise ValueError("--enable_learned_noise_corr requires --learned_corr_ckpt")
        noise_corrector = load_learned_corrector(opt.learned_corr_ckpt, device)
        noise_corrector.meta.t_cut = int(opt.learned_corr_t_cut)
        noise_corrector.meta.alpha = float(opt.learned_corr_alpha)
        logger.info(
            f"Learned corrector ON ckpt={opt.learned_corr_ckpt} "
            f"t_cut={opt.learned_corr_t_cut} alpha={opt.learned_corr_alpha}"
        )

    vsc_sampler_kwargs = {}
    if opt.vsc_stats:
        vsc_stats = torch.load(opt.vsc_stats, map_location="cpu")
        vsc_sampler_kwargs = dict(
            vsc_stats=vsc_stats,
            vsc_var_field=opt.vsc_var_field,
            vsc_absorb_strength=opt.vsc_absorb_strength,
            vsc_max_budget_fraction=opt.vsc_max_budget_fraction,
        )
        logger.info(
            f"VSC stats loaded from {opt.vsc_stats} absorb={opt.vsc_absorb_strength} "
            f"max_budget_frac={opt.vsc_max_budget_fraction}"
        )


    # write config out
    sampling_file = os.path.join(logdir, "sampling_config.yaml")
    sampling_conf = vars(opt)

    with open(sampling_file, 'a+') as f:
        yaml.dump(sampling_conf, f, default_flow_style=False)
    if opt.verbose:
        print(sampling_conf)
        logger.info("first_stage_model")
        logger.info(model.first_stage_model)
        logger.info("UNet model")
        logger.info(model.model)


    run(
        model,
        imglogdir,
        eta=opt.eta,
        vanilla=opt.vanilla_sample,
        n_samples=opt.n_samples,
        custom_steps=opt.custom_steps,
        batch_size=opt.batch_size,
        nplog=numpylogdir,
        dpm=opt.dpm,
        noise_corrector=noise_corrector,
        vsc_sampler_kwargs=vsc_sampler_kwargs,
        cfg_scale=float(opt.scale),
    )

    logger.info("done.")
