import torchvision
import torch.nn as nn
import torch

class ResNet18(nn.Module):
    output_size = 512

    def __init__(self, pretrained=False, mapping_feat_dim=None):
        super(ResNet18, self).__init__()
        pretrained = torchvision.models.resnet18(pretrained=pretrained)

        for module_name in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc']:
            self.add_module(module_name, getattr(pretrained, module_name))
        
        if mapping_feat_dim is None or self.fc.in_features == mapping_feat_dim:
            self.mapping = nn.Identity()
        else:
            self.mapping = nn.Linear(self.fc.in_features, mapping_feat_dim)
            self.fc = nn.Linear(mapping_feat_dim, self.fc.out_features)
    
    def forward(self, x, get_feat=False, get_ha=False, get_ha_x=False):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        b1 = self.layer1(x)
        b2 = self.layer2(b1)
        b3 = self.layer3(b2)
        b4 = self.layer4(b3)
        pool = self.avgpool(b4)
        pool = torch.flatten(pool, 1)
        feat = self.mapping(pool)
        out = self.fc(feat)

        ha = [b1, b2, b3, b4]

        if get_ha_x:
            ha.insert(0, x)

        if get_ha:
            if get_feat:
                return ha, feat, out
            else:
                return ha, out
        
        if get_feat:
            return feat, out
        else:
            return out
        # feats = {}
        # feats["feats"] = [x, b1, b2, b3, b4]
        # feats["pooled_feat"] = feat

        return out, feats


class ResNet50(nn.Module):
    output_size = 2048

    def __init__(self, pretrained=False, mapping_feat_dim=None):
        super(ResNet50, self).__init__()
        pretrained = torchvision.models.resnet50(pretrained=pretrained)
        
        for module_name in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc']:
            self.add_module(module_name, getattr(pretrained, module_name))

        if mapping_feat_dim is None or self.fc.in_features == mapping_feat_dim:
            self.mapping = nn.Identity()
        else:
            self.mapping = nn.Linear(self.fc.in_features, mapping_feat_dim)
            self.fc = nn.Linear(mapping_feat_dim, self.fc.out_features)

    def forward(self, x, get_feat=False, get_ha=False, get_ha_x=False):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        b1 = self.layer1(x)
        b2 = self.layer2(b1)
        b3 = self.layer3(b2)
        b4 = self.layer4(b3)
        pool = self.avgpool(b4)
        pool = torch.flatten(pool, 1)
        feat = self.mapping(pool)
        out = self.fc(feat)
        
        ha = [b1, b2, b3, b4]
        if get_ha_x:
            ha.insert(0, x)

        if get_ha:
            if get_feat:
                return ha, feat, out
            else:
                return ha, out
        
        if get_feat:
            return feat, out
        else:
            return out

class ResNeXt50_32x4d(nn.Module):
    output_size = 2048  # same as resnet50

    def __init__(self, pretrained=False, mapping_feat_dim=None):
        super(ResNeXt50_32x4d, self).__init__()
        pretrained_model = torchvision.models.resnext50_32x4d(pretrained=pretrained)

        for module_name in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc']:
            self.add_module(module_name, getattr(pretrained_model, module_name))

        if mapping_feat_dim is None or self.fc.in_features == mapping_feat_dim:
            self.mapping = nn.Identity()
        else:
            self.mapping = nn.Linear(self.fc.in_features, mapping_feat_dim)
            self.fc = nn.Linear(mapping_feat_dim, self.fc.out_features)

    def forward(self, x, get_feat=False, get_ha=False, get_ha_x=False):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        b1 = self.layer1(x)
        b2 = self.layer2(b1)
        b3 = self.layer3(b2)
        b4 = self.layer4(b3)
        pool = self.avgpool(b4)
        pool = torch.flatten(pool, 1)
        feat = self.mapping(pool)
        out = self.fc(feat)
        
        ha = [b1, b2, b3, b4]
        if get_ha_x:
            ha.insert(0, x)

        if get_ha:
            if get_feat:
                return ha, feat, out
            else:
                return ha, out
        
        if get_feat:
            return feat, out
        else:
            return out
