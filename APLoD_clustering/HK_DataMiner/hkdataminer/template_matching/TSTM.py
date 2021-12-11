import argparse
import numpy as np
from sklearn.decomposition import PCA
import sh
import mrcfile
from Select_angle import cal_var_byPCA
from gen_2D_image import gen_image
from two_stage_matching import matching_algorithm
# training parameters

parser = argparse.ArgumentParser()
parser.add_argument('--vol_size', type=int, default=128, help='3D conformation size is vol_size*vol_size*vol_size')
parser.add_argument('--n_components', type=int, default=3, help='pca components')
parser.add_argument('--conf_size', type=float, default=99, help='3D conformations number: conf_size')
parser.add_argument('--datatype', type=str, default='sim', help='dataset type: simulation or real')
parser.add_argument('--path1', type=str, default='./select_angle', help='path of select angle')
parser.add_argument('--path2', type=str, default='./data', help='path of data')
parser.add_argument('--path3', type=str, default='./gen_2D_image/', help='path of gen 2d image')
parser.add_argument('--path4', type=str, default='./two_stage_matching/', help='path of matching stage')
opt = parser.parse_args()

def main():
  ###Select angle
  select_anlge = cal_var_byPCA.Select_anlge_step(n_components=opt.n_component, vol_size=opt,vol_size, conf_size=opt.conf_size, outputname=opt.path1)
  input = select_angle.load_3Dmrc(opt.path2 + '/test_data_'+ opt.datatype)  #load 3D mrcs
  pc_eigenvalue = select_angle.PCA(input)
  select_angle.project(pc_eigenvalue)
  select_anlge.cal_contribution_from_each_pc_original_weighted()
  select_angle.find_best_anlge()
  
  ###gen_2D_images
  gen_image.gen_image(opt.datatype, opt.path3)
  
  
 ### two_stage_matching
  matching_algorithm.two_stage_matching(opt.datatype, opt.path4)
  
if __name__ == '__main__':
    main()
  
  


