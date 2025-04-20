import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import os

# ===== 1. Updated Dataset Class for CSV data with unified labels =====
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
        """Process the dataframe to group utterances by dialogue_id with unified label encoding"""
        dialogues = []
        
        # Group by Dialogue_ID
        for dialogue_id, group in self.df.groupby('Dialogue_ID'):
            utterances = []
            unified_target = []
            
            # Sort by Utterance_ID to ensure correct sequence
            group = group.sort_values('Utterance_ID')
            
            for _, row in group.iterrows():
                utterances.append(row['Utterance'])
                
                # Unified label encoding: 0=neutral, 1=anger, 2=cause
                if row['Label'] == 'cause':
                    unified_target.append(2)  # cause
                elif row['Label'] != 'neutral' and row['Emotion'] != 'neutral':
                    unified_target.append(1)  # anger
                else:
                    unified_target.append(0)  # neutral
            
            dialogues.append({
                'dialogue_id': int(dialogue_id),
                'utterances': utterances,
                'unified_target': unified_target,
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
        unified_target_tensor = torch.tensor(dialogue['unified_target'], dtype=torch.long)  # Using long for class indices
        
        return {
            'dialogue_id': dialogue_id,
            'text_feat': text_feat_tensor,
            'audio_feat': audio_feat_tensor,
            'video_feat': video_feat_tensor,
            'unified_target': unified_target_tensor,
            'num_utterances': num_utterances
        }

# ===== 2. Updated Model Architecture with unified classifier =====
class MultiModalTransformer(nn.Module):
    def __init__(self, text_dim=768, audio_dim=1024, video_dim=2048, 
                 hidden_dim=512, num_heads=8, num_layers=4, dropout=0.1, num_classes=3):
        super().__init__()
        
        # Feature dimension for each modality
        self.text_dim = text_dim
        self.audio_dim = audio_dim
        self.video_dim = video_dim
        self.combined_dim = text_dim + audio_dim + video_dim
        self.num_classes = num_classes
        
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
        
        # Unified classifier for all three classes (neutral, anger, cause)
        self.unified_classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, text_feat, audio_feat, video_feat):
        """
        Args:
            text_feat: [batch_size, seq_len, text_dim]
            audio_feat: [batch_size, seq_len, audio_dim]
            video_feat: [batch_size, seq_len, video_dim]
        
        Returns:
            logits: [batch_size, seq_len, num_classes]
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
        logits = self.unified_classifier(context_features)  # [B, seq_len, num_classes]
        
        return logits

# ===== 3. Updated Training Functions =====
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    
    # Loss function - CrossEntropyLoss with class weights to handle imbalance
    # Approximate weights: neutral is most common, cause is most rare
    class_weights = torch.tensor([1.0, 3.0, 5.0], device=device)  # [neutral, anger, cause]
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    for batch in dataloader:
        # Move batch to device
        text_feat = batch['text_feat'].to(device)          # [B, seq_len, text_dim]
        audio_feat = batch['audio_feat'].to(device)        # [B, seq_len, audio_dim]
        video_feat = batch['video_feat'].to(device)        # [B, seq_len, video_dim]
        unified_target = batch['unified_target'].to(device)  # [B, seq_len]
        num_utterances = batch['num_utterances']  # List of actual sequence lengths
        
        # Forward pass
        logits = model(text_feat, audio_feat, video_feat)  # [B, seq_len, num_classes]
        
        # Reshape for loss calculation
        B, S, C = logits.shape
        logits_flat = logits.view(-1, C)  # [B*S, num_classes]
        targets_flat = unified_target.view(-1)  # [B*S]
        
        # Calculate standard classification loss
        ce_loss = criterion(logits_flat, targets_flat)
        
        # Calculate dialogue-level constraint loss
        dialogue_loss = 0
        probs = F.softmax(logits, dim=2)  # [B, seq_len, num_classes]
        
        for i in range(B):
            seq_len = num_utterances[i]
            dialogue_probs = probs[i, :seq_len]  # [seq_len, num_classes]
            
            # Get maximum probability for each class across the dialogue
            max_class_probs = torch.max(dialogue_probs, dim=0)[0]  # [num_classes]
            
            # We want at least one anger (class 1) and one cause (class 2) prediction
            # Penalize if max probability for these classes is too low
            min_prob_threshold = 0.3
            anger_penalty = F.relu(min_prob_threshold - max_class_probs[1])
            cause_penalty = F.relu(min_prob_threshold - max_class_probs[2])
            
            dialogue_loss += anger_penalty + cause_penalty
        
        dialogue_loss = dialogue_loss / B  # Average over batch
        
        # Combine losses with a weight factor
        lambda_dialogue = 0.5  # Weight for dialogue constraint loss
        total_batch_loss = ce_loss + lambda_dialogue * dialogue_loss
        
        # Backpropagation
        optimizer.zero_grad()
        total_batch_loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += total_batch_loss.item()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    total_dialogues = 0
    dialogue_accuracies = []
    
    # For detailed metrics
    all_dialogue_preds = []
    all_dialogue_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            text_feat = batch['text_feat'].to(device)
            audio_feat = batch['audio_feat'].to(device)
            video_feat = batch['video_feat'].to(device)
            unified_target = batch['unified_target'].to(device)
            
            # Forward pass
            logits = model(text_feat, audio_feat, video_feat)  # [B, seq_len, num_classes]
            
            # Get predictions
            preds = torch.argmax(logits, dim=2)  # [B, seq_len]
            
            # For each dialogue in the batch
            batch_size = unified_target.size(0)
            for i in range(batch_size):
                # Get number of actual utterances for this dialogue
                num_utterances = batch['num_utterances'][i]
                
                # Get predictions and targets for this dialogue
                pred = preds[i, :num_utterances]
                target = unified_target[i, :num_utterances]
                
                # Calculate accuracy for this dialogue
                correct_predictions = (pred == target).sum().item()
                dialogue_accuracy = correct_predictions / num_utterances
                dialogue_accuracies.append(dialogue_accuracy)
                
                # Check for presence of anger and cause
                pred_has_anger = (pred == 1).any()
                pred_has_cause = (pred == 2).any()
                target_has_anger = (target == 1).any()
                target_has_cause = (target == 2).any()
                
                # Store dialogue-level predictions for metrics
                all_dialogue_preds.append([pred_has_anger.cpu().item(), pred_has_cause.cpu().item()])
                all_dialogue_targets.append([target_has_anger.cpu().item(), target_has_cause.cpu().item()])
                
                total_dialogues += 1
    
    # Calculate average dialogue accuracy
    mean_dialogue_accuracy = sum(dialogue_accuracies) / len(dialogue_accuracies) if dialogue_accuracies else 0
    
    # Convert to numpy arrays for metric calculation
    all_dialogue_preds = np.array(all_dialogue_preds)
    all_dialogue_targets = np.array(all_dialogue_targets)
    
    # Calculate dialogue-level F1 scores for anger and cause detection
    anger_f1 = f1_score(all_dialogue_targets[:, 0], all_dialogue_preds[:, 0])
    cause_f1 = f1_score(all_dialogue_targets[:, 1], all_dialogue_preds[:, 1])
    macro_f1 = (anger_f1 + cause_f1) / 2
    
    metrics = {
        "mean_dialogue_accuracy": mean_dialogue_accuracy,
        "anger_detection_f1": anger_f1,
        "cause_detection_f1": cause_f1,
        "macro_f1": macro_f1,
        "num_dialogues": total_dialogues
    }
    
    return metrics

# ===== 4. Updated Custom Collate Function =====
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
    batch_unified = torch.zeros(batch_size, max_len, dtype=torch.long)  # Using long for class indices
    batch_num_utts = []
    
    # Fill in batch tensors
    for i, item in enumerate(batch):
        seq_len = item['text_feat'].shape[0]
        batch_text[i, :seq_len] = item['text_feat']
        batch_audio[i, :seq_len] = item['audio_feat']
        batch_video[i, :seq_len] = item['video_feat']
        batch_unified[i, :seq_len] = item['unified_target']
        batch_num_utts.append(seq_len)
    
    return {
        'dialogue_id': [item['dialogue_id'] for item in batch],
        'text_feat': batch_text,
        'audio_feat': batch_audio,
        'video_feat': batch_video,
        'unified_target': batch_unified,
        'num_utterances': batch_num_utts
    }

# ===== 5. Train/Val Split Function (unchanged) =====
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
    print("Dialogue ID LENGTH:", len(dialogue_ids))

    # Split dialogues
    split_idx = int(len(dialogue_ids) * (1 - val_ratio))
    train_dialogue_ids = dialogue_ids[:split_idx]
    val_dialogue_ids = dialogue_ids[split_idx:]
    
    return train_dialogue_ids, val_dialogue_ids

# ===== 6. Updated Main Training Loop =====
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
        'dropout': 0.2,
        'num_classes': 3  # Unified classes: neutral, anger, cause
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
        
        # Update learning rate based on validation loss (using macro F1)
        val_f1 = val_metrics['macro_f1']
        scheduler.step(1 - val_f1)  # Scheduler expects loss (1-F1)
        
        # Print progress
        print(f"Epoch {epoch+1}/{training_params['num_epochs']}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Mean Dialogue Accuracy: {val_metrics['mean_dialogue_accuracy']:.4f}")
        print(f"  Anger Detection F1: {val_metrics['anger_detection_f1']:.4f}")
        print(f"  Cause Detection F1: {val_metrics['cause_detection_f1']:.4f}")
        print(f"  Macro F1: {val_metrics['macro_f1']:.4f}")
        print(f"  Number of Dialogues: {val_metrics['num_dialogues']}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'model_params': model_params
            }, os.path.join(output_dir, f"unified_best_model_epoch{epoch+1}.pt"))
            print(f"  Saved new best model with F1: {val_f1:.4f}")
    
    # Save final model
    torch.save({
        'epoch': training_params['num_epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_params': model_params
    }, os.path.join(output_dir, "unified_final_model.pt"))
    
    return model, model_params

# ===== 7. Updated Inference Function =====
def predict_dialogue(model, csv_file, feature_dir, dialogue_id, model_params, device=None):
    """
    Predict unified class (neutral, anger, cause) for a specific dialogue with dialogue-level analysis
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
        logits = model(text_feat_tensor, audio_feat_tensor, video_feat_tensor)
        
        # Get probabilities and predictions
        probs = F.softmax(logits, dim=2).squeeze(0).cpu().numpy()  # [seq_len, num_classes]
        pred_class = np.argmax(probs, axis=1)  # [seq_len]
    
    # Class names mapping
    class_names = {0: "Neutral", 1: "Anger", 2: "Cause"}
    
    # Dialogue-level analysis
    anger_probs = probs[:, 1]  # Probabilities for anger class
    cause_probs = probs[:, 2]  # Probabilities for cause class
    
    # Find utterances with highest confidence for anger and cause
    max_anger_idx = np.argmax(anger_probs)
    max_cause_idx = np.argmax(cause_probs)
    
    # Get dialogue-level predictions
    has_anger = np.any(pred_class == 1)
    has_cause = np.any(pred_class == 2)
    
    # Prepare results
    dialogue_results = {
        'dialogue_id': dialogue_id,
        'has_anger': has_anger,
        'has_cause': has_cause,
        'anger_confidence': anger_probs[max_anger_idx],
        'cause_confidence': cause_probs[max_cause_idx],
        'most_likely_anger_utterance': {
            'utterance_id': max_anger_idx,
            'text': utterances[max_anger_idx],
            'confidence': anger_probs[max_anger_idx]
        },
        'most_likely_cause_utterance': {
            'utterance_id': max_cause_idx,
            'text': utterances[max_cause_idx],
            'confidence': cause_probs[max_cause_idx]
        },
        'utterances': []
    }
    
    # Add detailed utterance-level results
    for i in range(num_utterances):
        # Get probabilities for each class
        class_probs = {class_names[j]: probs[i, j] for j in range(3)}
        
        dialogue_results['utterances'].append({
            'utterance_id': i,
            'text': utterances[i],
            'speaker': dialogue_df.iloc[i]['Speaker'] if 'Speaker' in dialogue_df.columns else "Unknown",
            'original_label': dialogue_df.iloc[i]['Label'] if 'Label' in dialogue_df.columns else "Unknown",
            'predicted_class': class_names[pred_class[i]],
            'class_probabilities': class_probs
        })
    
    return dialogue_results

def print_dialogue_results(results):
    """
    Helper function to print dialogue prediction results in a readable format
    """
    if results is None:
        print("No results to display")
        return
    
    print("\n" + "="*80)
    print(f"Dialogue Analysis (ID: {results['dialogue_id']})")
    print("="*80)
    
    # Print dialogue-level predictions
    print("\nDialogue-Level Predictions:")
    print(f"Contains Anger: {'Yes' if results['has_anger'] else 'No'} (Confidence: {results['anger_confidence']:.3f})")
    print(f"Contains Cause: {'Yes' if results['has_cause'] else 'No'} (Confidence: {results['cause_confidence']:.3f})")
    
    # Print most likely anger and cause utterances
    print("\nMost Likely Anger Utterance:")
    print(f"ID: {results['most_likely_anger_utterance']['utterance_id']}")
    print(f"Text: {results['most_likely_anger_utterance']['text']}")
    print(f"Confidence: {results['most_likely_anger_utterance']['confidence']:.3f}")
    
    print("\nMost Likely Cause Utterance:")
    print(f"ID: {results['most_likely_cause_utterance']['utterance_id']}")
    print(f"Text: {results['most_likely_cause_utterance']['text']}")
    print(f"Confidence: {results['most_likely_cause_utterance']['confidence']:.3f}")
    
    # Print detailed utterance-level results
    print("\nDetailed Utterance Analysis:")
    print("-"*80)
    for utt in results['utterances']:
        print(f"\nUtterance {utt['utterance_id']} ({utt['speaker']}):")
        print(f"Text: {utt['text']}")
        print(f"Original Label: {utt['original_label']}")
        print(f"Predicted Class: {utt['predicted_class']}")
        print("Class Probabilities:")
        for class_name, prob in utt['class_probabilities'].items():
            print(f"  {class_name}: {prob:.3f}")
    print("="*80)

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
    print("Starting unified model training...")
    model, model_params = train_model(csv_file, feature_dir, training_params)
    
    print("Training completed!")
    
    # Example: Predict for a specific dialogue
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Try to find the best model
        best_model_files = [f for f in os.listdir('./models') if f.startswith('unified_best_model_epoch')]
        if best_model_files:
            best_model_path = os.path.join('./models', sorted(best_model_files)[-1])  # Get latest best model
            print(f"Loading model from {best_model_path}")
            
            # Load model
            checkpoint = torch.load(best_model_path)
            saved_model_params = checkpoint['model_params']
            
            model = MultiModalTransformer(**saved_model_params).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Predict for dialogue ID 40 (as shown in your example)
            results = predict_dialogue(model, csv_file, feature_dir, 893, saved_model_params, device)
            
            # Print results in a readable format
            print_dialogue_results(results)
        else:
            print("No best model found. Using the final trained model.")
    except Exception as e:
        print(f"Error loading best model: {e}")