from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps, SpacedDiffusionDFoT

from diffusion import gaussian_diffusion_dfot as gd_dfot

def create_diffusion(
    timestep_respacing,
    noise_schedule="linear", 
    use_kl=False,
    sigma_small=False,
    predict_xstart=False,
    learn_sigma=True,
    rescale_learned_sigmas=False,
    diffusion_steps=1000
):
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type

    )

def create_diffusion_dfot(
    timestep_respacing,
    noise_schedule="linear", 
    use_kl=False,
    sigma_small=False,
    predict_xstart=False,
    learn_sigma=True,
    rescale_learned_sigmas=False,
    diffusion_steps=1000,
    shift=1.0,
    zero_terminal_snr=False
):
    betas = gd_dfot.get_named_beta_schedule(noise_schedule, diffusion_steps, shift=shift, zero_terminal_snr=zero_terminal_snr)
    if use_kl:
        loss_type = gd_dfot.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd_dfot.LossType.RESCALED_MSE
    else:
        loss_type = gd_dfot.LossType.MSE
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]
    return SpacedDiffusionDFoT(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd_dfot.ModelMeanType.EPSILON if not predict_xstart else gd_dfot.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd_dfot.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd_dfot.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd_dfot.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type

    )
