# Integration Examples {#integration_examples}

## 🔧 Easy Integration Guide

DNM framework is designed as a drop-in replacement for traditional training loops, making integration straightforward and seamless.

## 🚀 Quick Integration Examples

### Basic Integration

```python
# Replace this traditional training loop...
import torch.nn as nn
import torch.optim as optim

model = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(64, 128, 3, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(128, 10)
)

optimizer = optim.Adam(model.parameters())
# ... traditional training loop ...

# With this one-line DNM integration:
from neuroexapt.core.dnm_framework import train_with_dnm

result = train_with_dnm(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    target_accuracy=95.0
)
```

### Custom Training Loop Integration

```python
from neuroexapt.core.dnm_framework import DNMFramework
from neuroexapt.core.morphogenesis import MorphogenesisEngine

# Initialize DNM components
dnm = DNMFramework(
    bottleneck_threshold=0.02,
    morphogenesis_patience=8
)
morphogenesis_engine = MorphogenesisEngine()

# Your existing training loop with DNM integration
for epoch in range(num_epochs):
    # Regular training step
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    # Evaluate model
    val_accuracy = evaluate_model(model, val_loader)
    
    # DNM morphogenesis check
    if dnm.should_trigger_morphogenesis(model, val_accuracy, epoch):
        print(f"🧬 Triggering morphogenesis at epoch {epoch}")
        model = morphogenesis_engine.evolve_model(model, train_loader)
        optimizer = optim.Adam(model.parameters())  # Reinitialize optimizer
        
    print(f"Epoch {epoch}: Accuracy = {val_accuracy:.2f}%")
```

## 🏢 Framework-Specific Integrations

### PyTorch Lightning Integration

```python
import pytorch_lightning as pl
from neuroexapt.integrations.lightning import DNMLightningModule

class MyLightningModel(DNMLightningModule):
    def __init__(self, model, target_accuracy=95.0):
        super().__init__(
            model=model,
            target_accuracy=target_accuracy,
            dnm_config={
                'bottleneck_threshold': 0.02,
                'enable_aggressive_growth': False
            }
        )
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        
        # DNM automatically handles morphogenesis
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss

# Training
trainer = pl.Trainer(max_epochs=100)
model = MyLightningModel(your_model, target_accuracy=95.0)
trainer.fit(model, train_dataloader, val_dataloader)
```

### Hugging Face Transformers Integration

```python
from transformers import Trainer, TrainingArguments
from neuroexapt.integrations.transformers import DNMTrainer

# Define your model and tokenizer
model = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# DNM-enhanced training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
)

# DNM configuration
dnm_config = {
    'target_accuracy': 92.0,
    'enable_attention_growth': True,  # Specifically for transformers
    'enable_layer_addition': True,
    'morphogenesis_patience': 3
}

# Use DNMTrainer instead of standard Trainer
trainer = DNMTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    dnm_config=dnm_config
)

trainer.train()
```

### Keras/TensorFlow Integration

```python
import tensorflow as tf
from neuroexapt.integrations.tensorflow import DNMCallback

# Build your Keras model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Add DNM callback
dnm_callback = DNMCallback(
    target_accuracy=0.95,
    bottleneck_threshold=0.02,
    morphogenesis_patience=8,
    enable_neuron_division=True,
    enable_connection_growth=True
)

# Train with DNM
history = model.fit(
    train_dataset,
    epochs=100,
    validation_data=val_dataset,
    callbacks=[dnm_callback]
)
```

## 🎯 Domain-Specific Integrations

### Computer Vision Pipeline

```python
from torchvision import transforms, datasets
from neuroexapt.tasks.vision import DNMVisionClassifier

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Create vision-specific DNM classifier
vision_classifier = DNMVisionClassifier(
    num_classes=1000,
    backbone='resnet50',
    pretrained=True,
    dnm_config={
        'enable_attention_growth': True,
        'enable_multiscale_features': True,
        'target_accuracy': 85.0
    }
)

# Train with automatic data augmentation and morphogenesis
result = vision_classifier.fit(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    auto_augmentation=True,
    auto_batch_size=True
)
```

### Natural Language Processing

```python
from neuroexapt.tasks.nlp import DNMTextClassifier
from transformers import AutoTokenizer

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Create NLP-specific DNM classifier
nlp_classifier = DNMTextClassifier(
    model_name='bert-base-uncased',
    num_classes=2,
    max_length=512,
    dnm_config={
        'enable_attention_growth': True,
        'enable_layer_addition': True,
        'target_accuracy': 92.0
    }
)

# Prepare data
train_texts = ["Sample text 1", "Sample text 2", ...]
train_labels = [0, 1, ...]

# Train with automatic tokenization and morphogenesis
result = nlp_classifier.fit(
    train_texts=train_texts,
    train_labels=train_labels,
    val_texts=val_texts,
    val_labels=val_labels,
    tokenizer=tokenizer
)
```

### Time Series Forecasting

```python
from neuroexapt.tasks.timeseries import DNMTimeSeriesForecaster

# Time series specific DNM configuration
ts_forecaster = DNMTimeSeriesForecaster(
    sequence_length=100,
    prediction_horizon=10,
    features_dim=5,
    dnm_config={
        'enable_temporal_attention': True,
        'enable_multiscale_temporal': True,
        'target_mse': 0.01
    }
)

# Train on time series data
result = ts_forecaster.fit(
    train_sequences=train_data,
    val_sequences=val_data,
    auto_scaling=True
)
```

## 🛠 Advanced Integration Patterns

### Multi-GPU Training

```python
import torch.distributed as dist
from neuroexapt.distributed import DNMDistributedTrainer

# Initialize distributed training
dist.init_process_group(backend='nccl')

# Create distributed DNM trainer
distributed_trainer = DNMDistributedTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    dnm_config={
        'sync_morphogenesis': True,  # Synchronize morphogenesis across GPUs
        'target_accuracy': 95.0
    }
)

# Train across multiple GPUs
result = distributed_trainer.train()
```

### Hyperparameter Optimization Integration

```python
import optuna
from neuroexapt.optimization import DNMHyperoptTrainer

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    bottleneck_threshold = trial.suggest_float('bottleneck_threshold', 0.01, 0.1)
    
    # Create DNM trainer with suggested hyperparameters
    trainer = DNMHyperoptTrainer(
        model=create_model(),
        train_loader=create_train_loader(batch_size),
        val_loader=val_loader,
        learning_rate=lr,
        dnm_config={
            'bottleneck_threshold': bottleneck_threshold,
            'target_accuracy': 95.0
        }
    )
    
    # Train and return validation accuracy
    result = trainer.train(max_epochs=50)
    return result.final_accuracy

# Optimize hyperparameters
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

### Production Deployment

```python
from neuroexapt.deployment import DNMModelServer
import torch.jit

# Convert trained model to production format
production_model = torch.jit.script(trained_dnm_model)

# Create production server with DNM optimizations
server = DNMModelServer(
    model=production_model,
    optimization_config={
        'enable_quantization': True,
        'enable_pruning': True,
        'target_latency_ms': 50,
        'max_memory_mb': 1000
    }
)

# Deploy model
server.start(host='0.0.0.0', port=8080)

# Example inference endpoint
@server.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = server.predict(data['input'])
    return {'prediction': prediction}
```

## 🔍 Monitoring and Observability

### MLflow Integration

```python
import mlflow
from neuroexapt.monitoring import DNMMLflowLogger

# Initialize MLflow tracking
mlflow.set_experiment("DNM_Training")

with mlflow.start_run():
    # Create DNM logger
    dnm_logger = DNMMLflowLogger()
    
    # Train with automatic logging
    result = train_with_dnm(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        logger=dnm_logger,
        target_accuracy=95.0
    )
    
    # Log final results
    mlflow.log_metric("final_accuracy", result.final_accuracy)
    mlflow.log_metric("morphogenesis_count", result.morphogenesis_count)
    mlflow.log_metric("training_efficiency", result.training_efficiency)
```

### Weights & Biases Integration

```python
import wandb
from neuroexapt.monitoring import DNMWandbLogger

# Initialize Weights & Biases
wandb.init(project="dnm-experiments")

# Create W&B logger
wandb_logger = DNMWandbLogger()

# Train with automatic logging and visualization
result = train_with_dnm(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    logger=wandb_logger,
    target_accuracy=95.0
)

# Log architecture evolution visualization
wandb_logger.log_architecture_evolution(result.morphogenesis_history)
```

## 📋 Migration Checklist

### From Traditional Training

- [ ] Replace training loop with `train_with_dnm()`
- [ ] Set target accuracy instead of fixed epochs
- [ ] Configure DNM parameters for your task
- [ ] Update model saving/loading to handle dynamic architectures
- [ ] Adjust evaluation metrics if needed

### From Other AutoML Solutions

- [ ] Remove search space definitions
- [ ] Replace architecture search with DNM framework
- [ ] Update resource allocation (DNM is more efficient)
- [ ] Integrate existing preprocessing pipelines
- [ ] Migrate performance monitoring

### Production Considerations

- [ ] Test morphogenesis behavior in staging
- [ ] Set up model versioning for evolved architectures
- [ ] Configure monitoring for production morphogenesis
- [ ] Plan rollback strategy if needed
- [ ] Update inference infrastructure for optimized models

---

*For more integration examples and patterns, see our [GitHub examples repository](https://github.com/neuroexapt/neuroexapt/tree/main/examples).*