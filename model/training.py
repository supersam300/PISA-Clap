import torch
import torch.nn as nn
import torch.optim as optim
import os

class PfaultClassifier(nn.Module):
    def __init__(self, ebbeding_dim=512):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(ebbeding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128,2)
        )
    
    def forward(self, x):
        return self.network(x)
    


def trainmodel(embeddings , labels, epochs = 50):
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    embeddings = embeddings.squeeze(1).to(device)
    labels = labels.long().to(device)
    labels_cpu = labels.detach().cpu()
    normal_count = int((labels_cpu == 0).sum().item())
    abnormal_count = int((labels_cpu == 1).sum().item())
    total_count = max(normal_count + abnormal_count, 1)

    normal_count_safe = max(normal_count, 1)
    abnormal_count_safe = max(abnormal_count, 1)
    w_normal = total_count / (2.0 * normal_count_safe)
    w_abnormal = total_count / (2.0 * abnormal_count_safe)
    class_weights = torch.tensor([w_normal, w_abnormal], dtype=torch.float32, device=device)

    print(f"class counts (normal=0, abnormal=1): normal={normal_count}, abnormal={abnormal_count}")
    print(f"class weights: normal={w_normal:.4f}, abnormal={w_abnormal:.4f}")

    model = PfaultClassifier().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(embeddings)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
    
    torch.save(model.state_dict(), "pump_classifier.pth")
    print("the padwan is now a jedi")

    return model

NORMAL_DIR = "data/normal"
ABNORMAL_DIR = "data/abnormal"

def load_dataset():
    try:
        from .clap import load_audio_48k_mono, get_audio_embedding
    except ImportError:
        from clap import load_audio_48k_mono, get_audio_embedding
    all_embbedings = []
    all_labels = []
    print("extracting features from normal sounds")
    for filename in os.listdir(NORMAL_DIR):
        if filename.endswith(".wav"):
            filepath = os.path.join(NORMAL_DIR,filename)
            audio_array, _ = load_audio_48k_mono(filepath)
            embbedings = get_audio_embedding(audio_array)
            all_embbedings.append(embbedings)
            all_labels.append(0)
    print("extracting features from abnormal sounds")
    for filename in os.listdir(ABNORMAL_DIR):
        if filename.endswith("wav"):
            filepath = os.path.join(ABNORMAL_DIR,filename)
            audio_array,_ = load_audio_48k_mono(filepath)
            embeddings = get_audio_embedding(audio_array)
            all_embbedings.append(embeddings)
            all_labels.append(1)
    print(f"extraction is complete, processed {len(all_embbedings)} samples")
    embeddings_tensor = torch.stack(all_embbedings)
    labels_tensor = torch.tensor(all_labels)
    return embeddings_tensor, labels_tensor
    
if __name__ == "__main__":
    if not os.path.exists(NORMAL_DIR) or not os.path.exists(ABNORMAL_DIR):
        print(f"Please create the '{NORMAL_DIR}' and '{ABNORMAL_DIR}' folders and add your .wav files.")
    else:
        embeddings, labels = load_dataset()
        
        print(f"Final Embeddings Shape: {embeddings.shape}") 
        print(f"Final Labels Shape: {labels.shape}")         
        
        print("\nStarting Training Phase...")
        trained_model = trainmodel(embeddings, labels, epochs=50)    


    

    
    