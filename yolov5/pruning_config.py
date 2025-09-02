# Pruning Configuration for Lab-5
# เทคนิค Model Pruning สำหรับการลดขนาดโมเดลและปรับปรุงประสิทธิภาพ

# Pruning Methods และ Parameters
PRUNING_METHODS = {
    'magnitude': {
        'amount': 0.15,  # ลดลงจาก 0.2 สำหรับ fine-tuning
        'description': 'L1 magnitude-based unstructured pruning'
    },
    'structured': {
        'amount': 0.10,  # Conservative สำหรับ structured pruning
        'description': 'L2 norm structured pruning (remove channels)'
    },
    'random': {
        'amount': 0.12,  # Random pruning สำหรับเปรียบเทียบ
        'description': 'Random unstructured pruning'
    },
    'gradual': {
        'initial_amount': 0.05,
        'final_amount': 0.20,
        'frequency': 10,  # ทุก 10 epochs
        'description': 'Gradual pruning throughout training'
    }
}

# Layer-wise Pruning Strategy
LAYER_PRUNING_CONFIG = {
    'backbone_layers': 0.10,    # Backbone pruning น้อยกว่า
    'neck_layers': 0.15,        # FPN/PANet layers
    'head_layers': 0.20,        # Detection head pruning มากขึ้น
    'skip_first_last': True,    # ไม่ prune layer แรกและสุดท้าย
}

# Fine-tuning after Pruning
POST_PRUNING_CONFIG = {
    'recovery_epochs': 20,      # epochs สำหรับ recovery หลัง pruning
    'reduced_lr': 0.001,        # learning rate ต่ำสำหรับ recovery
    'gradual_unfreeze': True,   # ค่อยๆ unfreeze layers
}
