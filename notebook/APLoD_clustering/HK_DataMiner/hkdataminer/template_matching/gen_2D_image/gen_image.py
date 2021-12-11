####edit by Hanlin Gu on 28/8/2020
#####Aimed to generate the 2D images from 3D volumes by xmipp, same as the paper, we choose the red range viewing angles.
'''
input : 3D volumes (templates and experimental dataset)
output: template 2D images projected in red range, experimental 2D images uniformly distributed in sphere
'''
import sh

def gen_image(datatype, path3):
    sh.bash(path3 + 'project_' + datatype + '.sh')
    sh.bash(path3 + 'project_templates_' + datatype + '.sh')

