import torch
from scipy.special import digamma

def knn_entropy_torch(data, k=3):
    """Estimate the entropy of a continuous random variable using k-NN."""
    n_samples, n_features = data.shape
    distances = torch.cdist(data, data, p=2)
    distances.fill_diagonal_(float('inf'))
    distances_k, _ = torch.topk(distances, k=k, largest=False, dim=1)
    distances_k = distances_k[:, -1]  # k-th nearest neighbor distances
    mean_log_distance = torch.mean(torch.log(distances_k))
    entropy = (
        digamma(n_samples) - digamma(k) +
        n_features * mean_log_distance +
        n_features * torch.log(torch.tensor(2.0))
    )
    return entropy.item()

def conditional_entropy_torch(data, labels, k=3):
    """Estimate the conditional entropy H(X | Y) using k-NN."""
    unique_labels = torch.unique(labels)
    n_samples = len(labels)
    conditional_entropy = 0.0

    for label in unique_labels:
        # Select samples for the current class
        class_data = data[labels == label]
        class_prob = len(class_data) / n_samples  # P(Y = y)
        
        # Compute entropy for the current class
        class_entropy = knn_entropy_torch(class_data, k=k)
        conditional_entropy += class_prob * class_entropy  # Weighted entropy

    return conditional_entropy

# Example: Generate synthetic data
torch.manual_seed(42)
data = torch.cat([
    torch.distributions.MultivariateNormal(
        loc=torch.tensor([i, i]),
        covariance_matrix=torch.eye(2)
    ).sample((20,))
    for i in range(5)
])  # 5 classes, each with 20 samples

labels = torch.tensor([i for i in range(5) for _ in range(20)])  # 5 classes

# Estimate conditional entropy H(X | Y)
cond_entropy = conditional_entropy_torch(data, labels, k=5)
print("Estimated Conditional Entropy H(X | Y):", cond_entropy)
