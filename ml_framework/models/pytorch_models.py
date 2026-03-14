"""
PyTorch深度学习模型
"""

import numpy as np
from ml_framework.model_base import BaseModel


class MLPModel(BaseModel):
    """
    MLP (多层感知机) 回归模型
    """
    
    def __init__(self, input_dim: int, hidden_dims=[128, 64, 32], 
                 dropout_rate=0.3, **kwargs):
        super().__init__('MLP_PyTorch', **kwargs)
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.batch_size = kwargs.get('batch_size', 256)
        self.epochs = kwargs.get('epochs', 100)
        self.lr = kwargs.get('learning_rate', 0.001)
        self.patience = kwargs.get('patience', 10)
        
        self._build_model()
    
    def _build_model(self):
        """构建PyTorch模型"""
        try:
            import torch
            import torch.nn as nn
            
            class MLPNet(nn.Module):
                def __init__(self, input_dim, hidden_dims, dropout_rate):
                    super().__init__()
                    layers = []
                    prev_dim = input_dim
                    for hidden_dim in hidden_dims:
                        layers.append(nn.Linear(prev_dim, hidden_dim))
                        layers.append(nn.ReLU())
                        layers.append(nn.BatchNorm1d(hidden_dim))
                        layers.append(nn.Dropout(dropout_rate))
                        prev_dim = hidden_dim
                    layers.append(nn.Linear(prev_dim, 1))
                    self.net = nn.Sequential(*layers)
                
                def forward(self, x):
                    return self.net(x)
            
            self.model = MLPNet(self.input_dim, self.hidden_dims, self.dropout_rate)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            
        except ImportError:
            raise ImportError("请安装 torch: pip install torch")
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        import torch
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
        
        # 准备数据
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train).unsqueeze(1)
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        criterion = torch.nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': []}
        
        print(f"🚀 训练 MLP (device: {self.device})...")
        
        for epoch in range(self.epochs):
            # 训练
            self.model.train()
            train_losses = []
            for batch_X, batch_y in train_loader:
                # 跳过 batch size 为 1 的情况（BatchNorm 需要至少 2 个样本）
                if batch_X.size(0) <= 1:
                    continue
                
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            history['train_loss'].append(avg_train_loss)
            
            # 验证
            if X_val is not None and y_val is not None:
                val_loss = self._validate(X_val, y_val)
                history['val_loss'].append(val_loss)
                scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.best_state = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.patience:
                    print(f"   早停于 epoch {epoch+1}")
                    break
            
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1}: train_loss={avg_train_loss:.6f}")
        
        # 加载最佳模型
        if hasattr(self, 'best_state'):
            self.model.load_state_dict(self.best_state)
        
        self.is_fitted = True
        return history
    
    def _validate(self, X_val, y_val):
        import torch
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_val).to(self.device)
            y_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)
            outputs = self.model(X_tensor)
            loss = torch.nn.MSELoss()(outputs, y_tensor)
        return loss.item()
    
    def predict(self, X):
        import torch
        if not self.is_fitted:
            raise ValueError("模型未训练")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
        return outputs.cpu().numpy().flatten()
    
    def save(self, path: str):
        import torch
        torch.save({
            'model_state': self.model.state_dict(),
            'config': self.get_params()
        }, path)
    
    def load(self, path: str):
        import torch
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.is_fitted = True


class LSTMModel(MLPModel):
    """
    LSTM 时序模型
    """
    
    def __init__(self, input_dim: int, hidden_dim=64, num_layers=2, 
                 dropout_rate=0.3, **kwargs):
        # 覆盖父类的初始化
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.config = kwargs
        
        self.batch_size = kwargs.get('batch_size', 256)
        self.epochs = kwargs.get('epochs', 100)
        self.lr = kwargs.get('learning_rate', 0.001)
        self.patience = kwargs.get('patience', 10)
        
        self._build_lstm_model()
    
    def _build_lstm_model(self):
        """构建LSTM模型"""
        try:
            import torch
            import torch.nn as nn
            
            class LSTMNet(nn.Module):
                def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate):
                    super().__init__()
                    self.lstm = nn.LSTM(
                        input_size=input_dim,
                        hidden_size=hidden_dim,
                        num_layers=num_layers,
                        dropout=dropout_rate if num_layers > 1 else 0,
                        batch_first=True
                    )
                    self.fc = nn.Linear(hidden_dim, 1)
                
                def forward(self, x):
                    lstm_out, (h_n, c_n) = self.lstm(x)
                    last_hidden = h_n[-1]
                    out = self.fc(last_hidden)
                    return out
            
            self.model = LSTMNet(self.input_dim, self.hidden_dim, 
                                self.num_layers, self.dropout_rate)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.name = 'LSTM_PyTorch'
            
        except ImportError:
            raise ImportError("请安装 torch: pip install torch")
