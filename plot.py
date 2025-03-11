import json
import matplotlib.pyplot as plt

data = json.load(open("checkpoints/phi/trainer_state.json"))

train_steps = []
train_losses = []
eval_steps = []
eval_losses = []
for d in data["log_history"]:
    if "loss" in d: 
        train_steps.append(d["step"])
        train_losses.append(d["loss"])
    elif "eval_loss" in d:
        eval_steps.append(d["step"])
        eval_losses.append(d["eval_loss"])
        

plt.figure(figsize=(8, 5))
plt.plot(train_steps, train_losses, marker='o', linestyle='-', color='b', label="Train")
plt.plot(eval_steps, eval_losses, marker='o', linestyle='-', color='r', label="Eval")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Step vs. Loss")
plt.legend()
plt.grid(True)

plt.savefig("loss_plot.png")
