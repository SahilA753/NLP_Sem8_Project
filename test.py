import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import os

# ===== 1. Dataset Class for CSV data =====
class EmotionCauseDataset(Dataset):
    def __init__(self, csv_file, feature_dir):
        """
        Args:
            csv_file: Path to the CSV file with dialogue data
            feature_dir: Directory containing feature files
        """
        self.df = pd.read_csv(csv_file)
        self.feature_dir = feature_dir
        
        # Process the dataframe to group by dialogue_id
        self.dialogues = self.process_dialogues()
        
        # Determine feature dimensions
        self.feature_dims = self.get_feature_dimensions()
        print(f"Feature dimensions: {self.feature_dims}")
        
    def get_feature_dimensions(self):
        """Determine the feature dimensions by examining the first available file for each modality"""
        dims = {}
        
        # Get a dialogue ID and utterance ID from the data
        dialogue_id = self.dialogues[0]['dialogue_id']
        
        # Check text feature dimension
        text_path = os.path.join(self.feature_dir, 'text', f'text_feature_{dialogue_id}_0.npy')
        if os.path.exists(text_path):
            text_feat = np.load(text_path)
            dims['text'] = text_feat.shape[0]
        else:
            dims['text'] = 768  # Default
            
        # Check audio feature dimension
        audio_path = os.path.join(self.feature_dir, 'audio', f'audio_feature_{dialogue_id}_0.npy')
        if os.path.exists(audio_path):
            audio_feat = np.load(audio_path)
            dims['audio'] = audio_feat.shape[0]
        else:
            dims['audio'] = 1024  # Default, updated based on error
            
        # Check video feature dimension
        video_path = os.path.join(self.feature_dir, 'video', f'video_feature_{dialogue_id}_0.npy')
        if os.path.exists(video_path):
            video_feat = np.load(video_path)
            dims['video'] = video_feat.shape[0]
        else:
            dims['video'] = 2048  # Default
            
        return dims
        
    def process_dialogues(self):
        """Process the dataframe to group utterances by dialogue_id"""
        dialogues = []
        
        # Group by Dialogue_ID
        for dialogue_id, group in self.df.groupby('Dialogue_ID'):
            utterances = []
            emotion_target = []
            cause_target = []
            
            # Sort by Utterance_ID to ensure correct sequence
            group = group.sort_values('Utterance_ID')
            
            for _, row in group.iterrows():
                utterances.append(row['Utterance'])
                
                # Set emotion target (1 if labeled with any emotion, 0 otherwise)
                if row['Label'] != 'neutral' and row['Emotion'] != 'neutral':
                    emotion_target.append(1)
                else:
                    emotion_target.append(0)
                
                # Set cause target (1 if labeled as cause, 0 otherwise)
                if row['Label'] == 'cause':
                    cause_target.append(1)
                else:
                    cause_target.append(0)
            
            dialogues.append({
                'dialogue_id': int(dialogue_id),
                'utterances': utterances,
                'emotion_target': emotion_target,
                'cause_target': cause_target,
                'num_utterances': len(utterances)
            })
        
        return dialogues
    
    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, idx):
        dialogue = self.dialogues[idx]
        dialogue_id = dialogue['dialogue_id']
        num_utterances = dialogue['num_utterances']
        
        # Load features for each utterance
        text_features = []
        audio_features = []
        video_features = []
        
        for utt_id in range(num_utterances):
            # Load text features
            text_path = os.path.join(self.feature_dir, 'text', f'text_feature_{dialogue_id}_{utt_id}.npy')
            if os.path.exists(text_path):
                text_feat = np.load(text_path)
                # Ensure consistent dimensions
                if text_feat.shape != (self.feature_dims['text'],):
                    text_feat = np.zeros(self.feature_dims['text'])
            else:
                text_feat = np.zeros(self.feature_dims['text'])
            
            # Load audio features
            audio_path = os.path.join(self.feature_dir, 'audio', f'audio_feature_{dialogue_id}_{utt_id}.npy')
            if os.path.exists(audio_path):
                audio_feat = np.load(audio_path)
                # Ensure consistent dimensions
                if audio_feat.shape != (self.feature_dims['audio'],):
                    audio_feat = np.zeros(self.feature_dims['audio'])
            else:
                audio_feat = np.zeros(self.feature_dims['audio'])
            
            # Load video features
            video_path = os.path.join(self.feature_dir, 'video', f'video_feature_{dialogue_id}_{utt_id}.npy')
            if os.path.exists(video_path):
                video_feat = np.load(video_path)
                # Ensure consistent dimensions
                if video_feat.shape != (self.feature_dims['video'],):
                    video_feat = np.zeros(self.feature_dims['video'])
            else:
                video_feat = np.zeros(self.feature_dims['video'])
            
            text_features.append(text_feat)
            audio_features.append(audio_feat)
            video_features.append(video_feat)
        
        # Convert lists to tensors
        text_feat_tensor = torch.tensor(np.array(text_features), dtype=torch.float)
        audio_feat_tensor = torch.tensor(np.array(audio_features), dtype=torch.float)
        video_feat_tensor = torch.tensor(np.array(video_features), dtype=torch.float)
        emotion_target_tensor = torch.tensor(dialogue['emotion_target'], dtype=torch.float)
        cause_target_tensor = torch.tensor(dialogue['cause_target'], dtype=torch.float)
        
        return {
            'dialogue_id': dialogue_id,
            'text_feat': text_feat_tensor,
            'audio_feat': audio_feat_tensor,
            'video_feat': video_feat_tensor,
            'emotion_target': emotion_target_tensor,
            'cause_target': cause_target_tensor,
            'num_utterances': num_utterances
        }

# ===== 2. Model Architecture with auto dimensions =====
class MultiModalTransformer(nn.Module):
    def __init__(self, text_dim=768, audio_dim=1024, video_dim=2048, 
                 hidden_dim=512, num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()
        
        # Feature dimension for each modality
        self.text_dim = text_dim
        self.audio_dim = audio_dim
        self.video_dim = video_dim
        self.combined_dim = text_dim + audio_dim + video_dim
        
        # Projection layers to map each modality to a common space
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        
        # Feature fusion layer
        self.fusion_layer = nn.Linear(3 * hidden_dim, hidden_dim)
        
        # Cross-modal transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.emotion_classifier = nn.Linear(hidden_dim, 1)
        self.cause_classifier = nn.Linear(hidden_dim, 1)
    
    def forward(self, text_feat, audio_feat, video_feat):
        """
        Args:
            text_feat: [batch_size, seq_len, text_dim]
            audio_feat: [batch_size, seq_len, audio_dim]
            video_feat: [batch_size, seq_len, video_dim]
        
        Returns:
            emotion_logits: [batch_size, seq_len]
            cause_logits: [batch_size, seq_len]
        """
        batch_size, seq_len = text_feat.shape[0], text_feat.shape[1]
        
        # Project each modality
        text_hidden = self.text_proj(text_feat)    # [B, seq_len, hidden_dim]
        audio_hidden = self.audio_proj(audio_feat) # [B, seq_len, hidden_dim]
        video_hidden = self.video_proj(video_feat) # [B, seq_len, hidden_dim]
        
        # Concat features from different modalities for each utterance
        concat_features = torch.cat([text_hidden, audio_hidden, video_hidden], dim=2)  # [B, seq_len, 3*hidden_dim]
        
        # Fuse modalities
        fused_features = self.fusion_layer(concat_features)  # [B, seq_len, hidden_dim]
        fused_features = F.relu(fused_features)
        
        # Apply transformer encoder for contextualization
        context_features = self.transformer(fused_features)  # [B, seq_len, hidden_dim]
        
        # Get predictions for each utterance
        emotion_logits = self.emotion_classifier(context_features).squeeze(-1)  # [B, seq_len]
        cause_logits = self.cause_classifier(context_features).squeeze(-1)      # [B, seq_len]
        
        return emotion_logits, cause_logits

# ===== 3. Training Functions (updated) =====
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    
    # Loss functions - handle class imbalance with pos_weight
    emotion_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0], device=device))
    cause_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0], device=device))  # Higher weight for cause detection
    
    for batch in dataloader:
        # Move batch to device
        text_feat = batch['text_feat'].to(device)          # [B, seq_len, text_dim]
        audio_feat = batch['audio_feat'].to(device)        # [B, seq_len, audio_dim]
        video_feat = batch['video_feat'].to(device)        # [B, seq_len, video_dim]
        emotion_target = batch['emotion_target'].to(device).float()  # [B, seq_len]
        cause_target = batch['cause_target'].to(device).float()      # [B, seq_len]
        
        # Forward pass
        emotion_logits, cause_logits = model(text_feat, audio_feat, video_feat)
        
        # Calculate loss
        emotion_loss = emotion_criterion(emotion_logits, emotion_target)
        cause_loss = cause_criterion(cause_logits, cause_target)
        
        # Combined loss
        loss = emotion_loss + cause_loss
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    # Instead of lists for concatenation, we'll use these counters
    total_emotion_correct = 0
    total_cause_correct = 0
    total_pair_correct = 0
    total_samples = 0
    
    # For F1 calculations
    all_emotion_preds = []
    all_emotion_targets = []
    all_cause_preds = []
    all_cause_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            text_feat = batch['text_feat'].to(device)
            audio_feat = batch['audio_feat'].to(device)
            video_feat = batch['video_feat'].to(device)
            emotion_target = batch['emotion_target'].to(device)
            cause_target = batch['cause_target'].to(device)
            
            # Forward pass
            emotion_logits, cause_logits = model(text_feat, audio_feat, video_feat)
            
            # Convert logits to predictions (binary)
            emotion_preds = (torch.sigmoid(emotion_logits) > 0.5).float()
            cause_preds = (torch.sigmoid(cause_logits) > 0.5).float()
            
            # For each dialogue in the batch
            batch_size = emotion_target.size(0)
            for i in range(batch_size):
                # Get number of actual utterances for this dialogue (to avoid padding)
                num_utterances = batch['text_feat'][i].sum(dim=1).nonzero().size(0)
                if num_utterances == 0:  # Handle edge case where a sample might have all zeros
                    num_utterances = emotion_target[i].size(0)
                
                # Get predictions and targets for this dialogue
                e_pred = emotion_preds[i, :num_utterances]
                e_target = emotion_target[i, :num_utterances]
                c_pred = cause_preds[i, :num_utterances]
                c_target = cause_target[i, :num_utterances]
                
                # Add to lists for F1 calculation
                all_emotion_preds.append(e_pred.cpu().numpy())
                all_emotion_targets.append(e_target.cpu().numpy())
                all_cause_preds.append(c_pred.cpu().numpy())
                all_cause_targets.append(c_target.cpu().numpy())
                
                # Count correct predictions
                emotion_correct = (e_pred == e_target).sum().item()
                cause_correct = (c_pred == c_target).sum().item()
                pair_correct = ((e_pred == e_target) & (c_pred == c_target)).sum().item()
                
                # Update counters
                total_emotion_correct += emotion_correct  
                total_cause_correct += cause_correct
                total_pair_correct += pair_correct
                total_samples += num_utterances
    
    # Flatten lists for overall metrics
    all_emotion_preds = np.concatenate([p.flatten() for p in all_emotion_preds])
    all_emotion_targets = np.concatenate([t.flatten() for t in all_emotion_targets])
    all_cause_preds = np.concatenate([p.flatten() for p in all_cause_preds])
    all_cause_targets = np.concatenate([t.flatten() for t in all_cause_targets])
    
    # Calculate metrics
    emotion_f1 = f1_score(all_emotion_targets, all_emotion_preds, zero_division=0)
    emotion_precision = precision_score(all_emotion_targets, all_emotion_preds, zero_division=0)
    emotion_recall = recall_score(all_emotion_targets, all_emotion_preds, zero_division=0)
    
    cause_f1 = f1_score(all_cause_targets, all_cause_preds, zero_division=0)
    cause_precision = precision_score(all_cause_targets, all_cause_preds, zero_division=0)
    cause_recall = recall_score(all_cause_targets, all_cause_preds, zero_division=0)
    
    # Calculate accuracies
    emotion_accuracy = total_emotion_correct / total_samples if total_samples > 0 else 0
    cause_accuracy = total_cause_correct / total_samples if total_samples > 0 else 0
    pair_accuracy = total_pair_correct / total_samples if total_samples > 0 else 0
    
    metrics = {
        "emotion_f1": emotion_f1,
        "emotion_precision": emotion_precision,
        "emotion_recall": emotion_recall,
        "emotion_accuracy": emotion_accuracy,
        "cause_f1": cause_f1,
        "cause_precision": cause_precision,
        "cause_recall": cause_recall,
        "cause_accuracy": cause_accuracy,
        "pair_accuracy": pair_accuracy
    }
    
    return metrics

# ===== 4. Custom Collate Function =====
def collate_fn(batch):
    """
    Custom collate function to handle variable sequence lengths
    """
    # Get max sequence length in this batch
    max_len = max([item['text_feat'].shape[0] for item in batch])
    
    # Initialize batch tensors
    batch_size = len(batch)
    text_dim = batch[0]['text_feat'].shape[1]
    audio_dim = batch[0]['audio_feat'].shape[1]
    video_dim = batch[0]['video_feat'].shape[1]
    
    batch_text = torch.zeros(batch_size, max_len, text_dim)
    batch_audio = torch.zeros(batch_size, max_len, audio_dim)
    batch_video = torch.zeros(batch_size, max_len, video_dim)
    batch_emotion = torch.zeros(batch_size, max_len)
    batch_cause = torch.zeros(batch_size, max_len)
    
    # Fill in batch tensors
    for i, item in enumerate(batch):
        seq_len = item['text_feat'].shape[0]
        batch_text[i, :seq_len] = item['text_feat']
        batch_audio[i, :seq_len] = item['audio_feat']
        batch_video[i, :seq_len] = item['video_feat']
        batch_emotion[i, :seq_len] = item['emotion_target']
        batch_cause[i, :seq_len] = item['cause_target']
    
    return {
        'dialogue_id': [item['dialogue_id'] for item in batch],
        'text_feat': batch_text,
        'audio_feat': batch_audio,
        'video_feat': batch_video,
        'emotion_target': batch_emotion,
        'cause_target': batch_cause
    }

# ===== 5. Train/Val Split Function =====
def split_data(csv_file, val_ratio=0.1):
    """
    Split dialogues into training and validation sets
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Get unique dialogue IDs
    dialogue_ids = df['Dialogue_ID'].unique()
    
    # Shuffle dialogue IDs
    np.random.shuffle(dialogue_ids)
    print("Dialogue ID LENGTH:,",len(dialogue_ids))

    # Split dialogues
    split_idx = int(len(dialogue_ids) * (1 - val_ratio))
    train_dialogue_ids = dialogue_ids[:split_idx]
    val_dialogue_ids = dialogue_ids[split_idx:]
    
    return train_dialogue_ids, val_dialogue_ids

# ===== 6. Main Training Loop =====
def train_model(csv_file, feature_dir, training_params):
    # Create full dataset to detect feature dimensions
    full_dataset = EmotionCauseDataset(csv_file, feature_dir)
    
    # Get detected feature dimensions
    feature_dims = full_dataset.feature_dims
    print(feature_dims)
    # Update model parameters based on detected dimensions
    model_params = {
        'text_dim': feature_dims.get('text', 768),
        'audio_dim': feature_dims.get('audio', 1024),
        'video_dim': feature_dims.get('video', 2048),
        'hidden_dim': 256,
        'num_heads': 4,
        'num_layers': 3,
        'dropout': 0.2
    }
    
    print(f"Using model parameters: {model_params}")
    
    # Split data into train and validation sets
    train_dialogue_ids, val_dialogue_ids = split_data(csv_file, val_ratio=0.1)
    
    # Filter datasets for train and validation
    train_dataset = torch.utils.data.Subset(
        full_dataset, 
        [i for i, d in enumerate(full_dataset.dialogues) if d['dialogue_id'] in train_dialogue_ids]
    )
    
    val_dataset = torch.utils.data.Subset(
        full_dataset, 
        [i for i, d in enumerate(full_dataset.dialogues) if d['dialogue_id'] in val_dialogue_ids]
    )
    
    print(f"Training on {len(train_dataset)} dialogues, validating on {len(val_dataset)} dialogues")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=training_params['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=training_params['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = MultiModalTransformer(**model_params).to(device)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_params['learning_rate'],
        weight_decay=training_params['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=2, 
        verbose=True
    )
    
    # Training loop
    best_val_f1 = 0
    output_dir = training_params.get('output_dir', './models')
    os.makedirs(output_dir, exist_ok=True)
    
    for epoch in range(training_params['num_epochs']):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, device)
        
        # Update learning rate based on validation loss
        val_f1 = (val_metrics['emotion_f1'] + val_metrics['cause_f1']) / 2
        scheduler.step(1 - val_f1)  # Scheduler expects loss (1-F1)
        
        # Print progress
        print(f"Epoch {epoch+1}/{training_params['num_epochs']}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Emotion - F1: {val_metrics['emotion_f1']:.4f}, P: {val_metrics['emotion_precision']:.4f}, R: {val_metrics['emotion_recall']:.4f}")
        print(f"  Cause  - F1: {val_metrics['cause_f1']:.4f}, P: {val_metrics['cause_precision']:.4f}, R: {val_metrics['cause_recall']:.4f}")
        print(f"  Pair Accuracy: {val_metrics['pair_accuracy']:.4f}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'model_params': model_params  # Save model params for easy loading
            }, os.path.join(output_dir, f"best_model_epoch{epoch+1}.pt"))
            print(f"  Saved new best model with F1: {val_f1:.4f}")
    
    # Save final model
    torch.save({
        'epoch': training_params['num_epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_params': model_params
    }, os.path.join(output_dir, "final_model.pt"))
    
    return model, model_params

# ===== 7. Inference Function =====
def predict_dialogue(model, csv_file, feature_dir, dialogue_id, model_params, device=None):
    """
    Predict emotion and cause for a specific dialogue
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    
    # Read CSV file
    df = pd.read_csv(csv_file)
    dialogue_df = df[df['Dialogue_ID'] == dialogue_id].sort_values('Utterance_ID')
    
    if dialogue_df.empty:
        print(f"Dialogue ID {dialogue_id} not found in the CSV file")
        return None
    
    # Get utterances
    utterances = dialogue_df['Utterance'].tolist()
    num_utterances = len(utterances)
    
    # Load features
    text_features = []
    audio_features = []
    video_features = []
    
    for utt_id in range(num_utterances):
        # Load text features
        text_path = os.path.join(feature_dir, 'text', f'text_feature_{dialogue_id}_{utt_id}.npy')
        if os.path.exists(text_path):
            text_feat = np.load(text_path)
        else:
            text_feat = np.zeros(model_params['text_dim'])
        
        # Load audio features
        audio_path = os.path.join(feature_dir, 'audio', f'audio_feature_{dialogue_id}_{utt_id}.npy')
        if os.path.exists(audio_path):
            audio_feat = np.load(audio_path)
        else:
            audio_feat = np.zeros(model_params['audio_dim'])
        
        # Load video features
        video_path = os.path.join(feature_dir, 'video', f'video_feature_{dialogue_id}_{utt_id}.npy')
        if os.path.exists(video_path):
            video_feat = np.load(video_path)
        else:
            video_feat = np.zeros(model_params['video_dim'])
        
        text_features.append(text_feat)
        audio_features.append(audio_feat)
        video_features.append(video_feat)
    
    # Convert to tensors
    text_feat_tensor = torch.tensor(np.array(text_features), dtype=torch.float).unsqueeze(0).to(device)
    audio_feat_tensor = torch.tensor(np.array(audio_features), dtype=torch.float).unsqueeze(0).to(device)
    video_feat_tensor = torch.tensor(np.array(video_features), dtype=torch.float).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Forward pass
        emotion_logits, cause_logits = model(text_feat_tensor, audio_feat_tensor, video_feat_tensor)
        
        # Get probabilities and predictions
        emotion_prob = torch.sigmoid(emotion_logits).squeeze(0).cpu().numpy()
        cause_prob = torch.sigmoid(cause_logits).squeeze(0).cpu().numpy()
        
        emotion_pred = (emotion_prob > 0.5).astype(int)
        cause_pred = (cause_prob > 0.5).astype(int)
    
    # Combine results
    results = []
    for i in range(num_utterances):
        results.append({
            'Utterance_ID': i,
            'Utterance': utterances[i],
            'Speaker': dialogue_df.iloc[i]['Speaker'] if 'Speaker' in dialogue_df.columns else "Unknown",
            # 'Original_Emotion': dialogue_df.iloc[i]['Emotion'] if 'Emotion' in dialogue_df.columns else "Unknown",
            'Original_Label': dialogue_df.iloc[i]['Label'] if 'Label' in dialogue_df.columns else "Unknown",
            'Emotion_Prob': emotion_prob[i],
            'Emotion_Pred': "Anger" if emotion_pred[i] == 1 else "Neutral",
            'Cause_Prob': cause_prob[i],
            'Cause_Pred': "Cause" if cause_pred[i] == 1 else "Neutral"
        })
    
    return results

# ===== 8. Main Execution =====
if __name__ == "__main__":
    # Set paths
    csv_file = "makkari.csv"  # Path to your CSV file
    feature_dir = "/home/klad/Desktop/Friends/features"  # Root directory for features
    
    # Define training parameters
    training_params = {
        'batch_size': 8,
        'learning_rate': 2e-5,
        'weight_decay': 1e-4,
        'num_epochs': 20,
        'output_dir': './models'
    }
    
    # Train the model
    print("Starting model training...")
    model, model_params = train_model(csv_file, feature_dir, training_params)
    
    print("Training completed!")
    
    # Example: Predict for a specific dialogue
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Try to find the best model
        best_model_files = [f for f in os.listdir('./models') if f.startswith('best_model_epoch')]
        if best_model_files:
            best_model_path = os.path.join('./models', sorted(best_model_files)[-1])  # Get latest best model
            print(f"Loading model from {best_model_path}")
            
            # Load model
            checkpoint = torch.load(best_model_path)
            saved_model_params = checkpoint['model_params']
            
            model = MultiModalTransformer(**saved_model_params).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Predict for dialogue ID 6 (as shown in your example)
            results = predict_dialogue(model, csv_file, feature_dir, 40, saved_model_params, device)
            
            if results:
                print("\nPrediction results for Dialogue ID 6:")
                for result in results:
                    print(f"Utterance {result['Utterance_ID']} ({result['Speaker']}): {result['Utterance']}")
                    print(f"Label={result['Original_Label']}")
                    print(f"  Emotion: {result['Emotion_Pred']} (Prob: {result['Emotion_Prob']:.4f})")
                    print(f"  Cause: {result['Cause_Pred']} (Prob: {result['Cause_Prob']:.4f})")
                    print("-" * 50)
        else:
            print("No best model found. Using the final trained model.")
    except Exception as e:
        print(f"Error loading best model: {e}")