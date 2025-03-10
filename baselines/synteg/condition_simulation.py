import os
import time
import torch
import random
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from synteg import Generator, Discriminator
from torch.autograd import grad as torch_grad


SEED = 1337
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

# Argument parsing
parser = argparse.ArgumentParser(
    description='Synteg')
parser.add_argument('--dataset', type=str,
                    default='mimiciii', help="Dataset name")
parser.add_argument('--dataset_version', type=str,
                    default='1.4', help="Dataset version")
parser.add_argument('--LOG_DIR', type=str, required=True)
parser.add_argument("--embedding_dim", type=int,
                    default=112, help="Embedding dimension")
parser.add_argument("--word_embedding_dim", type=int,
                    default=80, help="Word embedding dimension")
parser.add_argument("--attention_size", type=int,
                    default=128, help="Attention size")
parser.add_argument("--ff_dim", type=int, default=128,
                    help="Feed-forward dimension")
parser.add_argument("--max_num_visit", type=int, default=48,
                    help="Maximum number of visits per patient")
parser.add_argument("--max_length_visit", type=int,
                    default=80, help="Maximum number of codes per visit")
parser.add_argument("--num_head", type=int, default=4,
                    help="Number of attention heads")
parser.add_argument("--code_vocab_dim", type=int,
                    default=6984, help="Code vocabulary dimension")
parser.add_argument("--label_vocab_dim", type=int,
                    default=25, help="Label vocabulary dimension")
parser.add_argument("--vocab_dim", type=int, default=6984 + 25 + 2,
                    help="Total vocabulary dimension (codes + labels + special tokens)")
parser.add_argument("--head_dim", type=int, default=32,
                    help="Head dimension for multi-head attention")
parser.add_argument("--lstm_dim", type=int, default=512,
                    help="LSTM hidden dimension")
parser.add_argument("--n_layer", type=int, default=3,
                    help="Number of layers in the network")
parser.add_argument("--condition_dim", type=int, default=256,
                    help="Condition dimension for conditional GAN")
parser.add_argument("--dependency_batchsize", type=int,
                    default=40, help="Batch size for dependency computations")
parser.add_argument("--args", type=int, nargs="+",
                    default=[-1, 48, 80, 4, 32], help="Arguments related to dimensions and heads")
parser.add_argument("--z_dim", type=int, default=128,
                    help="Dimension of latent vector z")
parser.add_argument("--g_dims", type=int, nargs="+",
                    help="Generator dimensions")
parser.add_argument("--d_dims", type=int, nargs="+",
                    default=[256, 256, 256, 128, 128, 128], help="Discriminator dimensions")
parser.add_argument("--gan_batchsize", type=int,
                    default=2500, help="Batch size for GAN training")
parser.add_argument("--gp_weight", type=int, default=10,
                    help="Gradient penalty weight for WGAN-GP")

args = parser.parse_args()
# Dynamically define g_dims
if args.g_dims is None:
    args.g_dims = [256, 256, 512, 512, 512, 512,
                   args.code_vocab_dim + args.label_vocab_dim + 2]
print(args)

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

condition_dataset = pickle.load(open(
    f"{args.LOG_DIR}/{args.dataset}_{args.dataset_version}_conditionDataset.pkl", 'rb'))


def get_batch(loc, batch_size):
    data = condition_dataset[loc:loc+batch_size]
    visits = [d['ehr'] for d in data]
    conditions = [d['condition'] for d in data]
    visits = torch.tensor(visits, dtype=torch.int64).to(device)
    conditions = torch.tensor(conditions).to(device)
    return (visits, conditions)


def shuffle_dataset(dataset):
    np.random.shuffle(dataset)


EPOCHS = 600
generator = Generator(args).to(device)
discriminator = Discriminator(args).to(device)
generator_optimizer = torch.optim.Adam(
    generator.parameters(), lr=4e-6, weight_decay=1e-5)
discriminator_optimizer = torch.optim.Adam(
    discriminator.parameters(), lr=2e-5, weight_decay=1e-5)
# if os.path.exists("../../save/synteg_condition_model"):
#     print("Loading previous model")
#     checkpoint = torch.load(
#         "../../save/synteg_condition_model", map_location=torch.device(device))
#     generator.load_state_dict(checkpoint['generator'])
#     generator_optimizer.load_state_dict(checkpoint['generator_optimizer'])
#     discriminator.load_state_dict(checkpoint['discriminator'])
#     discriminator_optimizer.load_state_dict(
#         checkpoint['discriminator_optimizer'])


def d_step(visits, conditions):
    discriminator.train()
    generator.eval()
    discriminator_optimizer.zero_grad()

    real = visits
    z = torch.randn((len(visits), args.z_dim)).to(device)
    epsilon = torch.rand((len(visits), 1)).to(device)

    synthetic = generator(z, conditions)
    real_output = discriminator(real, conditions)
    fake_output = discriminator(synthetic, conditions)
    w_distance = -torch.mean(real_output) + torch.mean(fake_output)

    interpolate = real + epsilon * (synthetic - real)
    interpolate_output = discriminator(interpolate, conditions)

    gradients = torch_grad(outputs=interpolate_output, inputs=interpolate,
                           grad_outputs=torch.ones(
                               interpolate_output.size()).to(device),
                           create_graph=True, retain_graph=True)[0]
    gradients = gradients.view(len(visits), -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    gradient_penalty = args.gp_weight * ((gradients_norm - 1) ** 2).mean()

    disc_loss = gradient_penalty + w_distance
    disc_loss.backward()
    discriminator_optimizer.step()
    return disc_loss, w_distance


def g_step(conditions):
    z = torch.randn((len(conditions), args.z_dim)).to(device)
    generator.train()
    discriminator.eval()
    generator_optimizer.zero_grad()
    synthetic = generator(z, conditions)
    fake_output = discriminator(synthetic, conditions)
    gen_loss = -torch.mean(fake_output)
    gen_loss.backward()
    generator_optimizer.step()


def train_step(batch):
    visits, conditions = batch  # bs * codes, bs * condition
    visits = torch.sum(F.one_hot(
        visits, num_classes=args.vocab_dim+1), dim=-2)[:, :-1]  # bs * vocab
    disc_loss, w_distance = d_step(visits, conditions)
    g_step(conditions)
    return disc_loss, w_distance


print('training start')
for e in tqdm(range(EPOCHS)):
    total_loss = 0
    total_w = 0
    step = 0
    shuffle_dataset(condition_dataset)
    for i in range(0, len(condition_dataset), args.gan_batchsize):
        batch = get_batch(i, args.gan_batchsize)
        loss, w = train_step(batch)
        total_loss += loss
        total_w += w
        step += 1
    format_str = 'epoch: %d, loss = %f, w = %f'
    print(format_str % (e, -total_loss / step, -total_w / step))
    if e % 50 == 49:
        state = {
            'generator': generator.state_dict(),
            'generator_optimizer': generator_optimizer.state_dict(),
            'discriminator': discriminator.state_dict(),
            'discriminator_optimizer': discriminator_optimizer.state_dict(),
            'epoch': e
        }
        torch.save(state, os.path.join(args.LOG_DIR, f'{args.dataset}_{args.dataset_version}_synteg_condition_model.pt'))
        print('\n------------ Save newest model ------------\n')
