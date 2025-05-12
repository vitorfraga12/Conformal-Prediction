from cp_func import *

apsk8_inner_radius = 2.0/torch.sqrt(2.0+(1+torch.sqrt(torch.tensor(3.0)))**2)
print(apsk8_inner_radius)
apsk8_outer_radius = torch.sqrt(2.0 - apsk8_inner_radius ** 2)
print(apsk8_outer_radius)