
DATA_PATH = "data/formation_e_dataset.csv"

MAT2VEC_PATH = "mat2vec/training/models/pretrained_embeddings"  

MAT2VEC_EMBEDDINGS_PATH = "data/met2vec.json"

MODEL_CONFIG = {
    "d_model": 256,           
    "n_layers": 6,            
    "n_heads": 8,             
    "d_ff": None,             
    "dropout": 0.15,           
    "d_f": 64,                
    "epsilon": 0.05,          
    "n_max": 10,              
    "element_head_n_layers": 2,  
    "element_head_hidden": None,  
}


TRAIN_CONFIG = {
    "batch_size": 1024,              
    "learning_rate": 8e-5,         
    "weight_decay": 5e-3,          
    "num_epochs": 2000,             
    "early_stopping_patience": 500, 
    "train_ratio": 0.8,            
    "val_ratio": 0.1,              
    "test_ratio": 0.1,             
    
    

    "num_workers": 16,              
    "pin_memory": True,             
    "prefetch_factor": 4,          
    "persistent_workers": True,     
    

    "use_warmup": True,             
    "warmup_epochs": 20,            
    "lr_scheduler": "cosine",      
    "lr_min": 1e-7,                 
}


LOSS_TYPE = "MSE"  

