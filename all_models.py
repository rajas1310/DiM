import timm

arch_list = ['resnet18', 'resnet26', 'resnet34',
 'efficientnet_b0', 'efficientnet_b1', 'efficientnetv2_s', 
 'convnext_tiny']


def get_model(arch_name, num_classes): 
  model = timm.create_model(arch_name, num_classes=num_classes)
  return model
