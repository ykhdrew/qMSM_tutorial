import mdtraj as md
import os
import operator
import numpy as np
from functools import reduce 
class TrajReader:
    '''
    def __init__(self, trajlistName, atomlistName, trajDir, trajExt, File_TOP):

        self.trajlistName = trajlistName
        self.atomlistName = atomlistName
        self.trajDir = trajDir
        self.trajExt = trajExt
        self.File_TOP = File_TOP
        self.homedir = self.get_homedir()
        self.trajlist_list = self.get_trajlist(trajlistName, self.homedir)
        self.atom_indices = self.get_atom_indices( atomlistName, self.homedir)
        #self.framelist = self.get_framelist()
    '''

    def walk_dir(self,input_traj_dir, input_traj_ext,topdown=True):
        frame_list = []
        for root, dirs, files in os.walk(input_traj_dir, topdown):
            for name in files:
                if os.path.splitext(name)[1] == input_traj_ext:
                    frame_list.append( os.path.join( root, name ))
        return frame_list

    def get_trajlist(self,trajlist_filename, trajlist_dir):
        trajlist_file = open( trajlist_filename )
        #trajlist_dir = self.get_homedir()
        trajlist_list = []
        for line in trajlist_file:
            list = trajlist_dir + '/' + line.rstrip("\n")
            list = list.strip() # remove the spaces in the end line, thanks to Yang Xi for reporting this bug, Stephen 20141208
            trajlist_list.append( list )
        trajlist_file.close()
        return trajlist_list

    def get_homedir(self):
        return os.getcwd()

    def get_atom_indices(self, indices_filename, indices_dir):
        if indices_filename != None:
            atom_indices = np.loadtxt(indices_filename, dtype=np.int32).tolist()
            return atom_indices
        else:
            return None

    def get_framefile_list(self, trajlist_list):
        framefile_list = []
        #trajlist_list = self.trajlist_list
        Ext = '.' + self.trajExt
        for trajlist in trajlist_list:
            framefile_list.extend(self.walk_dir(trajlist, Ext))
        return framefile_list

class XTCReader(TrajReader):
    def __init__(self, trajlistName, atomlistName, homedir, trajExt, File_TOP, nSubSample=None):
        self.trajlistName = trajlistName
        self.atomlistName = atomlistName
        self.trajDir = homedir
        self.trajExt = trajExt
        self.File_TOP = File_TOP
        self.homedir = homedir
        self.nSubSample = nSubSample
        #self.homedir = self.get_homedir()
        self.trajlist_list = self.get_trajlist(trajlistName, self.homedir)
        self.atom_indices = self.get_atom_indices( atomlistName, self.homedir)
        self.framefile_list = self.get_framefile_list(self.trajlist_list)
        self.trajs, self.traj_len = self.read_trajs(self.framefile_list)


    def read_trajs(self, framelist):
        trajs = []
        traj_len = []
        print("Reading trajs...")
        for frame in framelist:
            print('Reading: ', frame)
            #traj = md.load(frame, top=self.File_TOP, atom_indices=self.atom_indices)
            traj = md.load(frame, discard_overlapping_frames=True, top=self.File_TOP, #atom_indices=self.atom_indices,
                           stride=self.nSubSample)
            #traj = traj[:-1] #remove last one
            trajs.append(traj)
            traj_len.append(len(traj))

        len_trajs = len(trajs)
        whole_trajs= reduce(operator.add, (trajs[i] for i in range(len_trajs)))
        print("Done.")
        print(len_trajs, "trajs,", len(whole_trajs), "frames.")
        #print "debug output: len_trajs", len_trajs, "len_whole_trajs", len(whole_trajs)

        return whole_trajs, traj_len

    def get_phipsi(self, trajs, phi, psi):
        #phi = [6, 8, 14, 16]
        #psi = [4, 6, 8, 14]
        PHI_INDICES = []
        PSI_INDICES = []
        for i in range(len(phi)):
            PHI_INDICES.append(self.atom_indices.index(phi[i]))
            PSI_INDICES.append(self.atom_indices.index(psi[i]))
        #len_trajs = len(trajs)
        print("PSI:", PSI_INDICES)
        print("PHI:", PHI_INDICES)
        phi_angles = md.compute_dihedrals(trajs, [PHI_INDICES]) * 180.0 / np.pi
        psi_angles = md.compute_dihedrals(trajs, [PSI_INDICES]) * 180.0 / np.pi
        #phi_psi=np.column_stack((phi_angles, psi_angles))
        #return phi_psi
        return phi_angles, psi_angles

class DCDReader(TrajReader):
    def __init__(self, trajlistName, atomlistName, homedir, trajExt, File_TOP, nSubSample):
        self.trajlistName = trajlistName
        self.atomlistName = atomlistName
        self.trajDir = homedir
        self.trajExt = trajExt
        self.File_TOP = File_TOP
        self.homedir = homedir
        self.nSubSample = nSubSample
        #self.homedir = self.get_homedir()
        self.trajlist_list = self.get_trajlist(trajlistName, self.homedir)
        self.atom_indices = self.get_atom_indices( atomlistName, self.homedir)

    def read_trajs(self, framelist):
        #data = []
        trajs = []
        for frame in framelist:
            #framedata = []
            #print 'Reading: ', frame
            traj = md.load_dcd(frame, self.File_TOP, stride=self.nSubSample)
            trajs.append(traj)

        return trajs

class AmberReader(TrajReader):
    def __init__(self, trajlistName, atomlistName, homedir, trajExt, File_TOP, nSubSample):
        self.trajlistName = trajlistName
        self.atomlistName = atomlistName
        self.trajDir = homedir
        self.trajExt = trajExt
        self.File_TOP = File_TOP
        self.homedir = homedir
        self.nSubSample = nSubSample
        #self.homedir = self.get_homedir()
        self.trajlist_list = self.get_trajlist(trajlistName, self.homedir)
        self.atom_indices = self.get_atom_indices( atomlistName, self.homedir)

    def read_trajs(self, framelist):
        #data = []
        trajs = []
        for frame in framelist:
            #framedata = []
            print('Reading: ', frame)
            traj = md.load_netcdf(frame, self.File_TOP, stride=self.nSubSample)
            trajs.append(traj)

        return trajs

class VectorReader(TrajReader):
    def __init__(self, trajlistName, atomlistName=None, homedir='.', trajExt='txt', File_TOP=None, stride=None, framefile=None):
        self.trajlistName = trajlistName
        #self.atomlistName = atomlistName
        self.trajDir = homedir
        self.trajExt = trajExt
        #self.File_TOP = File_TOP
        self.homedir = homedir
        self.stride = stride
        #self.homedir = self.get_homedir()
        if framefile is not None:
            self.framefile_list = self.get_framefile(framefile)
        else:
            self.trajlist_list = self.get_trajlist(trajlistName, self.homedir)
            self.framefile_list = self.get_framefile_list(self.trajlist_list)
        #self.atom_indices = self.get_atom_indices( atomlistName, self.homedir)
        self.trajs, self.traj_len = self.read_trajs(self.framefile_list)

    def read_trajs(self, framelist):
        #data = []
        trajs = []
        traj_len = []
        for frame in framelist:
            #framedata = []
            print('Reading: ', frame)
            traj = np.loadtxt(frame, dtype='float32')
            #traj = traj[:-1] #remove last one
            if self.stride is not None:
                len_traj = len(traj)
                traj = traj[0:len_traj:self.stride]
            len_traj = len(traj)
            trajs.extend(traj)
            traj_len.append(len_traj)
        print("Total Points:", len(trajs))
        return np.asarray(trajs), traj_len

