import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionMechanism(nn.Module):
    def __init__(self, in_channels):
        super(AttentionMechanism, self).__init__()
        reduced_channels = max(1, in_channels // 8)  # Ensure at least 1 channel
        self.query_conv = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x1, x2):
        # Assume x1 and x2 have shape [batch_size, channels, height, width]
        batch_size, c, h, w = x1.size()

        # Compute query and key from both input tensors
        query1 = self.query_conv(x1).view(batch_size, -1, h * w).permute(0, 2, 1)  # [batch_size, h*w, c//8]
        key1 = self.key_conv(x1).view(batch_size, -1, h * w)  # [batch_size, c//8, h*w]
        value1 = self.value_conv(x1).view(batch_size, c, h * w).permute(0, 2, 1)  # [batch_size, h*w, c]

        query2 = self.query_conv(x2).view(batch_size, -1, h * w).permute(0, 2, 1)  # [batch_size, h*w, c//8]
        key2 = self.key_conv(x2).view(batch_size, -1, h * w)  # [batch_size, c//8, h*w]
        value2 = self.value_conv(x2).view(batch_size, c, h * w).permute(0, 2, 1)  # [batch_size, h*w, c]

        # Compute attention scores
        attention1 = torch.bmm(query1, key2)  # [batch_size, h*w, h*w]
        attention1 = F.softmax(attention1, dim=-1)  # [batch_size, h*w, h*w]
        out1 = torch.bmm(attention1, value2).permute(0, 2, 1).view(batch_size, c, h, w)  # [batch_size, c, h, w]

        attention2 = torch.bmm(query2, key1)  # [batch_size, h*w, h*w]
        attention2 = F.softmax(attention2, dim=-1)  # [batch_size, h*w, h*w]
        out2 = torch.bmm(attention2, value1).permute(0, 2, 1).view(batch_size, c, h, w)  # [batch_size, c, h, w]

        # Combine the outputs using a learnable gamma parameter
        out = self.gamma * out1 + (1 - self.gamma) * out2
        # out = self.gamma * x1 + (1 - self.gamma) * x2
        return out

def patch_attention(x1, x2, model, patch_size=64):
    # Split the input tensors into patches
    batch_size, c, h, w = x1.size()
    num_patches_h = h // patch_size
    num_patches_w = w // patch_size

    results = torch.zeros_like(x1).to(x1.device)

    for i in range(num_patches_h):
        for j in range(num_patches_w):
            start_h, end_h = i * patch_size, (i + 1) * patch_size
            start_w, end_w = j * patch_size, (j + 1) * patch_size

            patch_x1 = x1[:, :, start_h:end_h, start_w:end_w]
            patch_x2 = x2[:, :, start_h:end_h, start_w:end_w]

            # Forward pass through the attention mechanism
            patch_out = model(patch_x1, patch_x2)

            # Store the result in the corresponding position
            results[:, :, start_h:end_h, start_w:end_w] = patch_out

    return results

if __name__ == '__main__':

    # Example usage
    batch_size = 5
    in_channels = 5
    height, width = 224, 224

    x1 = torch.randn(batch_size, in_channels, height, width).cuda()
    x2 = torch.randn(batch_size, in_channels, height, width).cuda()

    attention_mechanism = AttentionMechanism(in_channels).cuda()
    output = patch_attention(x1, x2, attention_mechanism, patch_size=64)

    print(output.shape)  # Should print: torch.Size([5, 5, 224, 224])