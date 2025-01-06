import torch
import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
from dataset.celeb_dataset import CelebDataset
from torch.utils.data import DataLoader
from model.transformer import DIT
from model.vae import VAE
from scheduler.linear_scheduler import LinearNoiseScheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using mps')


def train(args):
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

    im_dataset = CelebDataset(split='train',
                              im_path=dataset_config['im_path'],
                              im_size=dataset_config['im_size'],
                              im_channels=dataset_config['im_channels'],
                              use_latents=True,
                              latent_path=os.path.join(train_config['task_name'],
                                                       train_config['vae_latent_dir_name'])
                              )

    data_loader = DataLoader(im_dataset,
                             batch_size=train_config['dit_batch_size'],
                             shuffle=True)

    # Instantiate the model
    im_size = dataset_config['im_size'] // 2 ** sum(autoencoder_model_config['down_sample'])
    model = DIT(im_size=im_size,
                im_channels=autoencoder_model_config['z_channels'],
                config=dit_model_config).to(device)
    model.train()

    if os.path.exists(os.path.join(train_config['task_name'],
                                   train_config['dit_ckpt_name'])):
        print('Loaded DiT checkpoint')
        model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                      train_config['dit_ckpt_name']),
                                         map_location=device))

    # Load VAE ONLY if latents are not to be used or are missing
    if not im_dataset.use_latents:
        print('Loading vae model as latents not present')
        vae = VAE(im_channels=dataset_config['im_channels'],
                  model_config=autoencoder_model_config).to(device)
        vae.eval()
        # Load vae if found
        if os.path.exists(os.path.join(train_config['task_name'],
                                       train_config['vae_autoencoder_ckpt_name'])):
            print('Loaded vae checkpoint')
            vae.load_state_dict(torch.load(os.path.join(
                train_config['task_name'],
                train_config['vae_autoencoder_ckpt_name']),
                map_location=device))
    # Specify training parameters
    num_epochs = train_config['dit_epochs']
    optimizer = AdamW(model.parameters(), lr=1E-5, weight_decay=0)
    criterion = torch.nn.MSELoss()

    # Run training
    if not im_dataset.use_latents:
        for param in vae.parameters():
            param.requires_grad = False

    acc_steps = train_config['dit_acc_steps']
    for epoch_idx in range(num_epochs):
        losses = []
        step_count = 0
        for im in tqdm(data_loader):
            step_count += 1
            im = im.float().to(device)
            if im_dataset.use_latents:
                mean, logvar = torch.chunk(im, 2, dim=1)
                std = torch.exp(0.5 * logvar)
                im = mean + std * torch.randn(mean.shape).to(device=im.device)
            else:
                with torch.no_grad():
                    im, _ = vae.encode(im)

            # Sample random noise
            noise = torch.randn_like(im).to(device)

            # Sample timestep
            t = torch.randint(0, diffusion_config['num_timesteps'],
                              (im.shape[0],)).to(device)

            # Add noise to images according to timestep
            noisy_im = scheduler.add_noise(im, noise, t)
            pred = model(noisy_im, t)
            loss = criterion(pred, noise)
            losses.append(loss.item())
            loss = loss / acc_steps
            loss.backward()
            if step_count % acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        optimizer.step()
        optimizer.zero_grad()
        print('Finished epoch:{} | Loss : {:.4f}'.format(
            epoch_idx + 1,
            np.mean(losses)))
        torch.save(model.state_dict(), os.path.join(train_config['task_name'],
                                                    train_config['dit_ckpt_name']))

    print('Done Training ...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for dit training')
    parser.add_argument('--config', dest='config_path',
                        default='config/celebhq.yaml', type=str)
    args = parser.parse_args()
    train(args)
