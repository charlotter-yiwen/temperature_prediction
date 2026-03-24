import argparse
import datetime
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# ----------------------------
# Models
# ----------------------------
class Generator(nn.Module):
    def __init__(self, cond_dim):
        super().__init__()
        self.fc_cond = nn.Sequential(
            nn.Linear(cond_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 64 * 25 * 25),
            nn.ReLU(True),
        )
        self.cnn = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(8, 1, 3, 1, 1),
        )

    def forward(self, noise_img, cond):
        cond_feat = self.fc_cond(cond)
        cond_feat = cond_feat.view(-1, 64, 25, 25)
        out = self.cnn(cond_feat)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 25 * 25, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, real_img, gen_img):
        x = torch.cat([real_img, gen_img], dim=1)
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        out = self.fc(h)
        return out


# ----------------------------
# Dataset
# ----------------------------
class TempDataset(Dataset):
    def __init__(self, params, temps):
        self.params = params.astype(np.float32)
        self.temps = temps.astype(np.float32)

    def __len__(self):
        return self.params.shape[0]

    def __getitem__(self, idx):
        cond = self.params[idx]
        img = self.temps[idx]
        img = img.reshape(1, 200, 200)
        return torch.from_numpy(img), torch.from_numpy(cond)


# ----------------------------
# Utility
# ----------------------------
def load_data(train_params_path, train_temps_path, test_params_path, test_temps_path):
    p_train = np.load(train_params_path)
    t_train = np.load(train_temps_path)
    p_test = np.load(test_params_path)
    t_test = np.load(test_temps_path)
    return p_train, t_train, p_test, t_test


def preprocess(p_train, t_train, p_test, t_test):
    # replace NaN in temps with mean
    def fill_nan(arr):
        flat = arr.reshape(arr.shape[0], -1)
        for i in range(flat.shape[0]):
            row = flat[i]
            mask = np.isnan(row)
            if np.any(mask):
                mean_val = np.nanmean(row)
                if np.isnan(mean_val):
                    mean_val = 0.0
                row[mask] = mean_val
            flat[i] = row
        return flat.reshape(arr.shape)

    t_train = fill_nan(t_train)
    t_test = fill_nan(t_test)

    # scale params
    scaler_x = StandardScaler()
    p_train_scaled = scaler_x.fit_transform(p_train)
    p_test_scaled = scaler_x.transform(p_test)

    # scale temps
    scaler_y = StandardScaler()
    t_train_flat = t_train.reshape(t_train.shape[0], -1)
    t_test_flat = t_test.reshape(t_test.shape[0], -1)
    t_train_scaled = scaler_y.fit_transform(t_train_flat).reshape(t_train.shape)
    t_test_scaled = scaler_y.transform(t_test_flat).reshape(t_test.shape)

    return p_train_scaled, t_train_scaled, p_test_scaled, t_test_scaled, scaler_x, scaler_y


def train_cgan(p_train, t_train, device, epochs=2000, batch_size=8, lr=1e-4, adv_weight=0.01):
    cond_dim = p_train.shape[1]
    dataset = TempDataset(p_train, t_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    G = Generator(cond_dim).to(device)
    D = Discriminator().to(device)
    bce_loss = nn.BCELoss().to(device)
    l1_loss = nn.L1Loss().to(device)
    optimizer_G = optim.Adam(G.parameters(), lr=lr)
    optimizer_D = optim.Adam(D.parameters(), lr=lr)

    for epoch in range(epochs):
        G.train(); D.train()
        for real_img, cond in loader:
            real_img = real_img.to(device)
            cond = cond.to(device)
            bsz = real_img.size(0)
            valid = torch.ones(bsz, 1, device=device)
            fake = torch.zeros(bsz, 1, device=device)

            # Train D
            noise_img = torch.randn_like(real_img)
            with torch.no_grad():
                gen_img_detached = G(noise_img, cond)
            d_real = D(real_img, real_img)
            d_fake = D(real_img, gen_img_detached)
            loss_d = 0.5 * (bce_loss(d_real, valid) + bce_loss(d_fake, fake))
            optimizer_D.zero_grad(); loss_d.backward(); optimizer_D.step()

            # Train G
            noise_img = torch.randn_like(real_img)
            gen_img = G(noise_img, cond)
            adv_loss = bce_loss(D(real_img, gen_img), valid)
            recon_loss = l1_loss(gen_img, real_img)
            loss_g = recon_loss + adv_weight * adv_loss
            optimizer_G.zero_grad(); loss_g.backward(); optimizer_G.step()

        if (epoch + 1) % 200 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} D:{loss_d.item():.4f} G:{loss_g.item():.4f} (recon {recon_loss.item():.4f}, adv {adv_loss.item():.4f})")

    return G


def predict(G, scaler_y, p_test_scaled, device):
    G.eval()
    with torch.no_grad():
        cond = torch.from_numpy(p_test_scaled.astype(np.float32)).to(device)
        noise = torch.randn(cond.size(0), 1, 200, 200, device=device)
        pred = G(noise, cond)  # (N,1,200,200)
    pred_np = pred.cpu().numpy().reshape(cond.size(0), -1)
    pred_inv = scaler_y.inverse_transform(pred_np).reshape(cond.size(0), 200, 200)
    return pred_np, pred_inv


def compute_r2(pred_np, temps_test):
    r2_vals = []
    for i in range(pred_np.shape[0]):
        true_flat = temps_test[i].reshape(-1)
        pred_flat = pred_np[i].reshape(-1)
        mask = np.isfinite(true_flat) & np.isfinite(pred_flat)
        if not np.any(mask):
            r2_vals.append(np.nan)
        else:
            r2_vals.append(r2_score(true_flat[mask], pred_flat[mask]))
    return np.array(r2_vals)


def main():
    parser = argparse.ArgumentParser(description='Train cGAN (CNN) on thermal fields and evaluate on test set.')
    parser.add_argument('--train-params', default='thermal_analysis_output/training data/params_training.npy')
    parser.add_argument('--train-temps', default='thermal_analysis_output/training data/temps_training.npy')
    parser.add_argument('--test-params', default='thermal_analysis_output/test data/params_testing.npy')
    parser.add_argument('--test-temps', default='thermal_analysis_output/test data/temps_testing.npy')
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--adv-weight', type=float, default=0.01)
    parser.add_argument('--pred-out', default=None, help='Optional path to save predicted flattened temps (npy)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    p_train, t_train, p_test, t_test = load_data(args.train_params, args.train_temps, args.test_params, args.test_temps)
    p_train_s, t_train_s, p_test_s, t_test_s, scaler_x, scaler_y = preprocess(p_train, t_train, p_test, t_test)

    G = train_cgan(p_train_s, t_train_s, device, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, adv_weight=args.adv_weight)

    pred_np_scaled, pred_inv = predict(G, scaler_y, p_test_s, device)

    # pred_inv: inverse-transformed (N,200,200) physical scale; pred_np_scaled is scaled domain
    r2_vals = compute_r2(pred_inv.reshape(pred_inv.shape[0], -1), t_test.reshape(t_test.shape[0], -1))
    print('Per-sample R2:', r2_vals)
    finite_mask = np.isfinite(r2_vals)
    if np.any(finite_mask):
        avg_r2 = float(np.mean(r2_vals[finite_mask]))
        print('Average R2 (finite):', avg_r2)
    else:
        avg_r2 = float('nan')
        print('Average R2: all NaN')

    if args.pred_out:
        np.save(args.pred_out, pred_inv.reshape(pred_inv.shape[0], -1))
        print('Saved predictions to', args.pred_out)

    # ---------- Write training log ----------
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_log.md')
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    n_train   = p_train.shape[0]
    n_test    = p_test.shape[0]
    n_nan     = int(np.sum(~finite_mask))
    n_valid   = int(np.sum(finite_mask))
    min_r2    = float(np.nanmin(r2_vals)) if n_valid > 0 else float('nan')
    max_r2    = float(np.nanmax(r2_vals)) if n_valid > 0 else float('nan')
    device_str = str(device)

    lines = []
    lines.append(f"\n---\n")
    lines.append(f"## cGAN-CNN Run \u2014 {timestamp}\n")
    lines.append(f"\n### Hyperparameters\n")
    lines.append(f"| Param | Value |\n|---|---|\n")
    lines.append(f"| Epochs        | {args.epochs} |\n")
    lines.append(f"| Batch size    | {args.batch_size} |\n")
    lines.append(f"| Learning rate | {args.lr} |\n")
    lines.append(f"| Adv weight    | {args.adv_weight} |\n")
    lines.append(f"| Device        | {device_str} |\n")
    lines.append(f"\n### Dataset\n")
    lines.append(f"| Item | Value |\n|---|---|\n")
    lines.append(f"| Train samples   | {n_train} |\n")
    lines.append(f"| Test samples    | {n_test}  |\n")
    lines.append(f"| Param dim       | {p_train.shape[1]} |\n")
    lines.append(f"| Temp field size | {t_train.shape[1:]} |\n")
    lines.append(f"\n### Results\n")
    lines.append(f"| Metric | Value |\n|---|---|\n")
    lines.append(f"| Average R\u00b2 (finite) | {avg_r2:.6f} |\n")
    lines.append(f"| Max R\u00b2              | {max_r2:.6f} |\n")
    lines.append(f"| Min R\u00b2              | {min_r2:.6f} |\n")
    lines.append(f"| Valid samples       | {n_valid}/{n_test} |\n")
    lines.append(f"| NaN samples         | {n_nan} |\n")
    lines.append(f"\n### Per-sample R\u00b2\n\n")
    lines.append(f"| Sample | R\u00b2 |\n|---|---|\n")
    for i, r2 in enumerate(r2_vals):
        val = f"{r2:.6f}" if np.isfinite(r2) else "NaN"
        lines.append(f"| {i+1:02d} | {val} |\n")
    lines.append(f"\n### File Paths\n")
    lines.append(f"- Train params: `{args.train_params}`\n")
    lines.append(f"- Train temps:  `{args.train_temps}`\n")
    lines.append(f"- Test params:  `{args.test_params}`\n")
    lines.append(f"- Test temps:   `{args.test_temps}`\n")
    if args.pred_out:
        lines.append(f"- Predictions saved: `{args.pred_out}`\n")

    write_header = not os.path.exists(log_path)
    with open(log_path, 'a', encoding='utf-8') as lf:
        if write_header:
            lf.write('# Training Log\n')
        lf.writelines(lines)
    print(f"Log written to {log_path}")


if __name__ == '__main__':
    main()
