import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable


transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Scale((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))  #  numbers here need to be adjusted in future
])


def getVGG_16bn(path_vgg):
    # if the pre_trained vgg16 model not in path_vgg, download it from the url below
    state_dict = torch.utils.model_zoo.load_url('https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',path_vgg)
    model = torchvision.models.vgg16_bn()
    model.load_state_dict(state_dict)
    # ingore the classifier
    model_2 = list(model.children())[0] 
    return model_2

# dtype determine to use cpu or gpu
def get_conv_feature_for_image(image, model, dtype=torch.cuda.FloatTensor):
    im = transform(image)
    im = im.view(1,*im.shape)
    feature = model(Variable(im).type(dtype))
    return feature.data
    
 
       
    