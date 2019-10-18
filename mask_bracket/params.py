from model.u_net import get_unet_256, get_unet_512, get_unet_128

input_size = 512

max_epochs = 50
batch_size = 32

threshold = 0.5

if input_size == 256:
  model_factory = get_unet_256
elif input_size == 128:
  model_factory = get_unet_128
else:
  model_factory = get_unet_512
