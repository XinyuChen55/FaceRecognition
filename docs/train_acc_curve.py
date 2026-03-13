import pandas as pd
import matplotlib.pyplot as plt

LOG = "work_dirs/ms1mv2_r50/training.log"
STEPS_PER_EPOCH = 500000 // 128

rows=[]
with open(LOG, "r", errors="ignore") as f:
    for line in f:
        if "Accuracy-Flip" not in line:
            continue
        
        step = int(line.split("[lfw][", 1)[1].split("]Accuracy-Flip", 1)[0])
        tail = line.split("Accuracy-Flip:", 1)[1].strip()
        acc_str, std_str = tail.split("+-", 1)
        acc = float(acc_str)
        std = float(std_str)
        epoch = step // STEPS_PER_EPOCH + 1
        rows.append({"epoch": epoch, "step": step, "acc": acc, "std": std})

df = pd.DataFrame(rows).sort_values(["epoch", "step"])
per_epoch = df.groupby("epoch", as_index=False).tail(1).reset_index(drop=True)

csv_out = "work_dirs/ms1mv2_r50/train_acc_per_epoch.csv"
png_out = "work_dirs/ms1mv2_r50/train_acc_curve.png"
per_epoch.to_csv(csv_out, index=False)

plt.figure()
plt.plot(per_epoch["epoch"], per_epoch["acc"])
plt.xlabel("Epoch")
plt.ylabel("LFW Accuracy")
plt.title("LFW Val Accuracy per Epoch")
plt.savefig(png_out, dpi=200, bbox_inches="tight")

loss_rows = []
with open(LOG, "r", errors="ignore") as f:
    for line in f:
        if "2026-03-11" not in line or "Loss" not in line:
            continue
        loss = float(line.split("Loss", 1)[1].strip().split()[0])
        step = int(line.split("Global Step:", 1)[1].strip().split()[0])
        epoch = int(line.split("Epoch:", 1)[1].split("Global", 1)[0].strip())
        loss_rows.append({"epoch":epoch, "step": step, "loss": loss})

df_loss = pd.DataFrame(loss_rows).sort_values("step")

csv_loss = "work_dirs/ms1mv2_r50/loss_curve.csv"
png_loss = "work_dirs/ms1mv2_r50/loss_curve.png"
df_loss.to_csv(csv_loss, index=False)

plt.figure()
plt.plot(df_loss["step"], df_loss["loss"])
plt.xlabel("Global step")
plt.ylabel("Training Loss")
plt.title("Training Loss Curve")
plt.savefig(png_loss, dpi=200, bbox_inches="tight")