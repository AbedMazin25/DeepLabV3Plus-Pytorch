import torch
from torch import nn
from torch.nn import functional as F

from .utils import _SimpleSegmentationModel


__all__ = ["DeepLabV3"]


def PSO_select_channels(output, num_channels=9, num_particles=9, num_iterations=100):
    num_filters = output.shape[1]  # Number of channels in the layer

    # Debugging information
    # print("Number of filters (num_filters):", num_filters)
    # print("Number of channels to select (num_channels):", num_channels)

    # Check if the number of channels is valid
    if num_filters < num_channels:
        raise ValueError("Number of filters is less than the number of channels to select.")

    # Adjust the number of channels if necessary
    num_channels = min(num_channels, num_filters)

    # Initial random values for particles
    particles = torch.randint(0, num_filters, (num_particles, num_channels), dtype=torch.float32)  # Initial positions
    velocities = torch.zeros_like(particles)  # Initial velocities

    # Objective function: Minimize the sum of selected pixel values
    def objective_function(particle):
        selected_pixels = output[:, particle.long(), :, :].sum(dim=(1, 2, 3))  # Select pixels based on particle positions
        return selected_pixels.sum()  # Total sum of the selected pixel values

    # PSO parameters
    inertia_weight = 0.7
    cognitive_constant = 1.5
    social_constant = 1.5

    # Best particle positions and global best position
    best_particle_positions = particles.clone()
    best_particle_scores = torch.tensor([objective_function(p) for p in particles])
    global_best_position = particles[best_particle_scores.argmin()]
    global_best_score = best_particle_scores.min()

    for iteration in range(num_iterations):  # Number of PSO iterations
        for i, particle in enumerate(particles):
            # Update particle velocities
            velocities[i] = (inertia_weight * velocities[i]
                             + cognitive_constant * torch.rand(1) * (best_particle_positions[i] - particle)
                             + social_constant * torch.rand(1) * (global_best_position - particle))

            # Update particle positions
            particles[i] = particles[i] + velocities[i]
            particles[i] = torch.clamp(particles[i], 0, num_filters - 1)  # Clamp particle positions

            # Calculate fitness and update best positions
            current_fitness = objective_function(particles[i])
            if current_fitness < best_particle_scores[i]:
                best_particle_positions[i] = particles[i]
                best_particle_scores[i] = current_fitness
            if current_fitness < global_best_score:
                global_best_position = particles[i]
                global_best_score = current_fitness

    return global_best_position.long()  # Return the optimal channel positions

class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass

class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential( 
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        low_level_feature = self.project( feature['low_level'] )
        output_feature = self.aspp(feature['out'])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
        return self.classifier( torch.cat( [ low_level_feature, output_feature ], dim=1 ) )
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class DeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHead, self).__init__()

        self.classifier = nn.Sequential(
            ASPP(in_channels, aspp_dilate),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        return self.classifier( feature['out'] )

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                            stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d( in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=in_channels ),
            # PointWise Conv
            nn.Conv2d( in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )
        
        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        
        # Standard ASPP layers
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))
        
        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        
        # Image pooling layer
        modules.append(ASPPPooling(in_channels, out_channels))
        
        # Max pooling convolution for PSO
        self.max_pooling_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # PSO-based channel selection convolution
        self.selected_conv = nn.Sequential(
            nn.Conv2d(9, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Combine results into a single tensor
        self.project = nn.Sequential(
            nn.Conv2d(6 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        self.convs = nn.ModuleList(modules)

    def forward(self, x):
        # Process through all standard ASPP layers
        res = [conv(x) for conv in self.convs]
        
        # Add max pooling convolution
        x_max_pooled = self.max_pooling_conv(x)
        
        # Use PSO to select the best 9 channels
        best_channels = PSO_select_channels(x_max_pooled)  # PSO-based selection
        x_selected = x_max_pooled[:, best_channels, :, :]  # Extract selected channels
        
        # Apply selected convolution to the chosen channels
        x_selected_conv = self.selected_conv(x_selected)
        
        # Append selected channels to the results
        res.append(x_selected_conv)
        
        # Concatenate all processed features
        res = torch.cat(res, dim=1)
        
        return self.project(res)



def convert_to_separable_conv(module):
    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0]>1:
        new_module = AtrousSeparableConvolution(module.in_channels,
                                      module.out_channels, 
                                      module.kernel_size,
                                      module.stride,
                                      module.padding,
                                      module.dilation,
                                      module.bias)
    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child))
    return new_module
