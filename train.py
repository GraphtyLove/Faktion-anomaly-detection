from gan.model import DCGAN
import torch

print("GPU available: ", torch.cuda.is_available())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)
nb_gpu = 1 if device.type == 'cuda' else 0
print("number of GPU: ", nb_gpu)
print("Defining model...")
model = DCGAN(ngpu=nb_gpu)
print("Start training mode...")
model.fit(
    epochs=500,
    batch_size=128,
    dataset_path="data/dice/",
    save_samples=True
)
print("Model saved.")
model.save_metrics("/output/metrics.json")
torch.save(model, "/output/model.pth")
