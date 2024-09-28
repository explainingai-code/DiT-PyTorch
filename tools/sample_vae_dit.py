import torch
import torchvision
import argparse
import yaml
import os
from torchvision.utils import make_grid
from PIL import Image
from tqdm import tqdm
from model.vae import VAE
from model.transformer import DIT
from scheduler.linear_scheduler import LinearNoiseScheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using mps')


def sample(model, scheduler, train_config, dit_model_config,
           autoencoder_model_config, diffusion_config, dataset_config, vae):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """
    im_size = dataset_config['im_size'] // 2 ** sum(autoencoder_model_config['down_sample'])
    xt = torch.randn((train_config['num_samples'],
                      autoencoder_model_config['z_channels'],
                      im_size,
                      im_size)).to(device)

    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
        # Get prediction of noise
        noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))

        # Use scheduler to get x0 and xt-1
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))

        # Save x0
        # ims = torch.clamp(xt, -1., 1.).detach().cpu()
        if i == 0:
            # Decode ONLY the final image to save time
            ims = vae.to(device).decode(xt)
        else:
            ims = xt
            ims = xt[:, :-1, :, :]

        ims = torch.clamp(ims, -1., 1.).detach().cpu()
        ims = (ims + 1) / 2

        grid = make_grid(ims, nrow=train_config['num_grid_rows'])
        img = torchvision.transforms.ToPILImage()(grid)

        if not os.path.exists(os.path.join(train_config['task_name'], 'samples')):
            os.mkdir(os.path.join(train_config['task_name'], 'samples'))
        img.save(os.path.join(train_config['task_name'], 'samples', 'x0_{}.png'.format(i)))
        img.close()


def infer(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################

    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    dit_model_config = config['dit_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']

    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])

    # Get latent image size
    im_size = dataset_config['im_size'] // 2 ** sum(autoencoder_model_config['down_sample'])
    model = DIT(im_size=im_size,
                im_channels=autoencoder_model_config['z_channels'],
                config=dit_model_config).to(device)

    model.eval()

    assert os.path.exists(os.path.join(train_config['task_name'],
                                       train_config['dit_ckpt_name'])), "Train DiT first"

    model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                  train_config['dit_ckpt_name']),
                                     map_location=device))
    print('Loaded dit checkpoint')

    # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])

    vae = VAE(im_channels=dataset_config['im_channels'],
              model_config=autoencoder_model_config)
    vae.eval()

    # Load vae if found
    assert os.path.exists(os.path.join(train_config['task_name'], train_config['vae_autoencoder_ckpt_name'])), \
        "VAE checkpoint not present. Train VAE first."
    vae.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                train_config['vae_autoencoder_ckpt_name']),
                                   map_location=device), strict=True)
    print('Loaded vae checkpoint')

    with torch.no_grad():
        sample(model, scheduler, train_config, dit_model_config,
               autoencoder_model_config, diffusion_config, dataset_config, vae)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for dit image generation')
    parser.add_argument('--config', dest='config_path',
                        default='config/celebhq.yaml', type=str)
    args = parser.parse_args()
    infer(args)
