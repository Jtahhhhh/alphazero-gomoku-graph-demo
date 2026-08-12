from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class GomokuState:
    board: np.ndarray
    to_play: int=1
    last_move: int=-1
    win_length: int=4
    @classmethod
    def initial(cls,size=6,win_length=4): return cls(np.zeros((size,size),np.int8),1,-1,win_length)
    @property
    def size(self): return self.board.shape[0]
    def legal_actions(self): return np.flatnonzero(self.board.reshape(-1)==0)
    def play(self,action):
        action=int(action)
        if action<0 or action>=self.size**2 or self.board.reshape(-1)[action]!=0: raise ValueError(f"illegal action {action}")
        board=self.board.copy(); board.reshape(-1)[action]=self.to_play
        return GomokuState(board,-self.to_play,action,self.win_length)
    def winner(self):
        n,k=self.size,self.win_length
        for r in range(n):
            for c in range(n):
                p=self.board[r,c]
                if not p: continue
                for dr,dc in ((0,1),(1,0),(1,1),(1,-1)):
                    er,ec=r+(k-1)*dr,c+(k-1)*dc
                    if 0<=er<n and 0<=ec<n and all(self.board[r+i*dr,c+i*dc]==p for i in range(k)): return int(p)
        return 0
    def terminal(self): return self.winner()!=0 or len(self.legal_actions())==0
    def outcome_for(self,player): return int(self.winner()==player)-int(self.winner()==-player)
    def features(self):
        n=self.size; rr,cc=np.meshgrid(np.arange(n),np.arange(n),indexing="ij")
        last=np.zeros_like(self.board,dtype=np.float32)
        if self.last_move>=0: last.reshape(-1)[self.last_move]=1
        return np.stack([(self.board==self.to_play),(self.board==-self.to_play),last,np.full_like(self.board,self.to_play,dtype=np.float32),rr/max(1,n-1),cc/max(1,n-1)],axis=0).astype(np.float32)
