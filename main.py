import pytorch_influence_functions as ptif

# Supplied by the user:
model = ptif.F
trainloader, testloader = ptif.get_my_dataloaders()

ptif.init_logging()
config = ptif.get_default_config()

influences, harmful, helpful = ptif.calc_img_wise(config, model, trainloader, testloader)

# do someting with influences/harmful/helpful