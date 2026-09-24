import torch

tensor0 = torch.tensor(1) # Tensor de ordem 0 (um elemento)
tensor1 = torch.tensor([1, 2, 3]) # Tensor de ordem 1 (vetor)
tensor2 = torch.tensor([[1, 2],  # Tensor de ordem 2 (matriz bidimensional)
                        [3, 4]])
tensor3 = torch.tensor([[1 , 2, 3],
                        [4, 5, 6]])

print(tensor3.dtype)

print(tensor3)

print(tensor3.shape)

print(tensor3.view(3, 2)) # Troca o número de colunas com número de linhas e vice-versa.

print(tensor3.T) # Faz a transposta do tensor

print(tensor3.matmul(tensor3.T)) # Matmul multiplica dois tensores.