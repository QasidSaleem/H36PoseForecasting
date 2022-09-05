from utils import prepare_2d_data
from torch.utils.data import Dataset

class Human2dJoints(Dataset):
    """Dataset class Human3.6m 2d Joints"""
    def __init__(
        self,
        data_file,
        mode="train",
        s_fname="scaler.joblib",
        s_type="normalize",
        n_seqs=20,
    ):
        self.sequences_matrix = prepare_2d_data(
           data_file=data_file,
           mode=mode,
           s_fname=s_fname,
           s_type=s_type
        )
        self.n_seqs=n_seqs
        

    def __len__(self):
        return len(self.sequences_matrix) // self.n_seqs
    
    def __getitem__(self, idx):
        return self.sequences_matrix[idx*self.n_seqs:(idx+1)*self.n_seqs]