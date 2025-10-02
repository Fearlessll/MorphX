from prognosis.Networks.resnet import resnet10
import torch
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional

class FusionNet(nn.Module):
    def __init__(self, patch_first_covd_param: List[int], patch_input_channel_num: int,
                 macro_first_covd_param: List[int], macro_input_channel_num: int,
                 macro_best_ckpt_path: str = None, output_use_sigmoid: bool = False):
        super(FusionNet, self).__init__()


        macro_net = resnet10(first_covd_param=macro_first_covd_param, input_channel_num=macro_input_channel_num)
        if macro_best_ckpt_path is not None:
            best_seg_ckpt = torch.load(macro_best_ckpt_path, map_location=torch.device('cpu'))
            macro_net.load_state_dict(best_seg_ckpt['model_state_dict'])
        self.macro_net = macro_net

        self.output_use_sigmoid = output_use_sigmoid

        self.classifier = nn.Sequential(nn.BatchNorm1d(1024 + 168),
                                        nn.Dropout(0.25),
                                        nn.Linear(1024 + 168, 64),
                                        # nn.Dropout(0.25),
                                        nn.Linear(64, 32),
                                        nn.BatchNorm1d(32),
                                        nn.ReLU(),
                                        nn.Linear(32, 1))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x_patch, x_macro):

        macro_vec, _ = self.macro_net(x_macro)
        features = torch.cat([macro_vec, x_patch], dim=1)
        hazard = self.classifier(features)
        if self.output_use_sigmoid == True:
            hazard = self.sigmoid(hazard)
        return hazard

    def __hasattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return True
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return True
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return True
        return False


#
# from prognosis.Networks.resnet import resnet9
# from prognosis.Networks.milmodel import MILModel
# import torch
# import torch.nn as nn
# from typing import Type, Any, Callable, Union, List, Optional
#
# class FusionNet(nn.Module):
#     def __init__(self, patch_first_covd_param: List[int], patch_input_channel_num: int,
#                  macro_first_covd_param: List[int], macro_input_channel_num: int,
#                  macro_best_ckpt_path: str = None, output_use_sigmoid: bool = False):
#         super(FusionNet, self).__init__()
#         # patch_net = MILModel(num_classes=1, pretrained=False, mil_mode='mean')
#         # # patch_net = resnet9(first_covd_param=patch_first_covd_param, input_channel_num=patch_input_channel_num)
#         # self.patch_net = patch_net
#         #self.patchnet = patch_net
#
#         macro_net = resnet9(first_covd_param=macro_first_covd_param, input_channel_num=macro_input_channel_num)
#         if macro_best_ckpt_path is not None:
#             best_seg_ckpt = torch.load(macro_best_ckpt_path, map_location=torch.device('cpu'))
#             macro_net.load_state_dict(best_seg_ckpt['model_state_dict'])
#         self.macro_net = macro_net
#         #self.segnet = macro_net
#
#         self.output_use_sigmoid = output_use_sigmoid
#
#         # self.classifier = nn.Sequential(nn.Linear(1024, 64),
#         #                                 nn.Dropout(0.25),
#         #                                 nn.Linear(64, 32),
#         #                                 nn.BatchNorm1d(32),
#         #                                 nn.ReLU(),
#         #                                 nn.Linear(32, 1))
#         self.fc_macro = nn.Sequential(nn.Linear(1024, 64),
#                                        # nn.BatchNorm1d(64),
#                                        # nn.ReLU(),
#                                        #  nn.Linear(64, 32),
#                                       )
#         self.fc_micro = nn.Sequential(nn.Linear(168, 64),
#                                        # nn.BatchNorm1d(64),
#                                        # nn.ReLU(),
#                                        #  nn.Linear(64, 32),
#                                       )
#
#         self.classifier = nn.Sequential(#nn.BatchNorm1d(1024 + 168),
#             # nn.BatchNorm1d(64),
#             # nn.ReLU(),
#                                         nn.Linear(64, 32),
#                                         nn.BatchNorm1d(32),
#                                         nn.ReLU(),
#                                         nn.Linear(32, 1))
#
#         # self.classifier = nn.Sequential(  # nn.Dropout(0.25),
#         #     nn.BatchNorm1d(1024 + 168),
#         #     #nn.ReLU(),
#         #     #nn.Dropout(0.2),
#         #     nn.Linear(1024 + 168, 64),
#         #     nn.BatchNorm1d(64),
#         #     #nn.ReLU(),
#         #     # nn.Dropout(0.25),
#         #     nn.Linear(64, 32),
#         #     nn.BatchNorm1d(32),
#         #     #nn.ReLU(),
#         #     nn.Linear(32, 1))
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x_patch, x_macro):
#         #patch_vec, _ = self.patch_net(x_patch)
#         #patch_vec = self.patch_net(x_patch)
#         macro_vec, _ = self.macro_net(x_macro)
#         macro_fc = self.fc_macro(macro_vec)
#         micro_fc = self.fc_micro(x_patch)
#         # patch_vec, _ = self.patchnet(x_patch)
#         # macro_vec, _ = self.segnet(x_macro)
#         #features = torch.cat([macro_fc, micro_fc], dim=1)
#         features = macro_fc+micro_fc
#         hazard = self.classifier(features)
#         if self.output_use_sigmoid == True:
#             hazard = self.sigmoid(hazard)
#         return hazard
#
#     def __hasattr__(self, name):
#         if '_parameters' in self.__dict__:
#             _parameters = self.__dict__['_parameters']
#             if name in _parameters:
#                 return True
#         if '_buffers' in self.__dict__:
#             _buffers = self.__dict__['_buffers']
#             if name in _buffers:
#                 return True
#         if '_modules' in self.__dict__:
#             modules = self.__dict__['_modules']
#             if name in modules:
#                 return True
#         return False
