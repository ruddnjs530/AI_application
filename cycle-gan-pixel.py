import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import os
from torch.nn import functional as F
from tqdm import tqdm
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def denormalize(tensor):
    tensor = tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    return torch.clamp(tensor, 0, 1)

def imshow(tensor, title=None):
    tensor = tensor.cpu().clone().squeeze(0)
    tensor = denormalize(tensor)
    image = transforms.ToPILImage()(tensor)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.show()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

def save_image(tensor, path):
    tensor = tensor.cpu().clone().squeeze(0)
    tensor = denormalize(tensor)
    image = transforms.ToPILImage()(tensor)
    image.save(path)

class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        ]

        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(True)
            ]
            in_features = out_features
            out_features = in_features * 2

        for _ in range(n_residual_blocks):
            model += [ResnetBlock(in_features)]

        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(True)
            ]
            in_features = out_features
            out_features = in_features // 2

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(64, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()
        model = [
            nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]

        model += [
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, True)
        ]

        model += [
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, True)
        ]

        model += [
            nn.Conv2d(256, 512, kernel_size=4, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, True)
        ]

        model += [nn.Conv2d(512, 1, kernel_size=4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

def real_loss(D_out):
    return torch.mean((D_out - 1) ** 2)

def fake_loss(D_out):
    return torch.mean(D_out ** 2)

def cycle_consistency_loss(real_img, reconstructed_img, lambda_weight):
    return lambda_weight * torch.mean(torch.abs(real_img - reconstructed_img))

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None, num_samples=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
        if num_samples is not None and num_samples < len(self.image_files):
            self.image_files = random.sample(self.image_files, num_samples)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

current_dir = os.path.dirname(os.path.abspath(__file__))
photo_dataset_dir = os.path.join(current_dir, "photo_dataset")
pixel_art_dataset_dir = os.path.join(current_dir, "pixel_art_dataset")

photo_dataset = CustomImageDataset(photo_dataset_dir, transform=transform, num_samples=200)
pixel_art_dataset = CustomImageDataset(pixel_art_dataset_dir, transform=transform, num_samples=200)


photo_loader = DataLoader(photo_dataset, batch_size=1, shuffle=True)
pixel_art_loader = DataLoader(pixel_art_dataset, batch_size=1, shuffle=True)

generator_AB = Generator(input_nc=3, output_nc=3).to(device)
generator_BA = Generator(input_nc=3, output_nc=3).to(device)
discriminator_A = Discriminator(input_nc=3).to(device)
discriminator_B = Discriminator(input_nc=3).to(device)

opt_disc = optim.Adam(list(discriminator_A.parameters()) + list(discriminator_B.parameters()), lr=0.0003, betas=(0.5, 0.999))
opt_gen = optim.Adam(list(generator_AB.parameters()) + list(generator_BA.parameters()), lr=0.0003, betas=(0.5, 0.999))

disc_scheduler = optim.lr_scheduler.StepLR(opt_disc, step_size=100, gamma=0.5)
gen_scheduler = optim.lr_scheduler.StepLR(opt_gen, step_size=100, gamma=0.5)

def train_fn(
    disc_A, disc_B, gen_AB, gen_BA, loader_A, loader_B, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler
):
    loop = tqdm(zip(loader_A, loader_B), leave=True, total=min(len(loader_A), len(loader_B)))

    for idx, (photo_data, pixel_data) in enumerate(loop):
        photo_img = photo_data.to(device)
        pixel_img = pixel_data.to(device)

        with torch.cuda.amp.autocast():
            fake_pixel = gen_AB(photo_img)
            D_A_real = disc_A(photo_img)
            D_A_fake = disc_A(fake_pixel.detach())
            D_A_real_loss = mse(D_A_real, torch.ones_like(D_A_real))
            D_A_fake_loss = mse(D_A_fake, torch.zeros_like(D_A_fake))
            D_A_loss = D_A_real_loss + D_A_fake_loss

            fake_photo = gen_BA(pixel_img)
            D_B_real = disc_B(pixel_img)
            D_B_fake = disc_B(fake_photo.detach())
            D_B_real_loss = mse(D_B_real, torch.ones_like(D_B_real))
            D_B_fake_loss = mse(D_B_fake, torch.zeros_like(D_B_fake))
            D_B_loss = D_B_real_loss + D_B_fake_loss

            D_loss = (D_A_loss + D_B_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        with torch.cuda.amp.autocast():
            D_A_fake = disc_A(fake_pixel)
            D_B_fake = disc_B(fake_photo)
            loss_G_A = mse(D_A_fake, torch.ones_like(D_A_fake))
            loss_G_B = mse(D_B_fake, torch.ones_like(D_B_fake))

            cycle_photo = gen_BA(fake_pixel)
            cycle_pixel = gen_AB(fake_photo)

            cycle_photo = F.interpolate(cycle_photo, size=photo_img.size()[2:])
            cycle_pixel = F.interpolate(cycle_pixel, size=pixel_img.size()[2:])

            cycle_photo_loss = l1(photo_img, cycle_photo)
            cycle_pixel_loss = l1(pixel_img, cycle_pixel)

            identity_photo = gen_BA(photo_img)
            identity_pixel = gen_AB(pixel_img)

            identity_photo = F.interpolate(identity_photo, size=photo_img.size()[2:])
            identity_pixel = F.interpolate(identity_pixel, size=pixel_img.size()[2:])

            identity_photo_loss = l1(photo_img, identity_photo)
            identity_pixel_loss = l1(pixel_img, identity_pixel)

            G_loss = (
                loss_G_A
                + loss_G_B
                + cycle_photo_loss * 10
                + cycle_pixel_loss * 10
                + identity_photo_loss * 5
                + identity_pixel_loss * 5
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        loop.set_postfix(D_loss=D_loss.item(), G_loss=G_loss.item())

def main():
    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    d_scaler = torch.cuda.amp.GradScaler()
    g_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(1, 101):
        print(f"Epoch [{epoch}/100] 시작")

        train_fn(
            discriminator_A,
            discriminator_B,
            generator_AB,
            generator_BA,
            photo_loader,
            pixel_art_loader,
            opt_disc,
            opt_gen,
            L1,
            mse,
            d_scaler,
            g_scaler,
        )

        disc_scheduler.step()
        gen_scheduler.step()

        print(f"Epoch [{epoch}/100] 종료")

    model_dir = 'cycle-gan-model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    torch.save(generator_AB.state_dict(), os.path.join(model_dir, "generator_AB_final.pth"))

    with torch.no_grad():
        photo_img = next(iter(photo_loader)).to(device)
        fake_pixel = generator_AB(photo_img)
        imshow(fake_pixel, title='Generated Pixel Art')
        imshow(photo_img, title='Original Photo')


if __name__ == "__main__":
    main()
