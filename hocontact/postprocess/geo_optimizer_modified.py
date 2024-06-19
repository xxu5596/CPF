from pprint import pprint

import numpy as np
import open3d as o3d
import torch
from manopth.anchorlayer import AnchorLayer
from manopth.axislayer import AxisLayer
from manopth.manolayer import ManoLayer
from manopth.quatutils import normalize_quaternion, quaternion_to_angle_axis
from manopth.rodrigues_layer import batch_rodrigues
from termcolor import colored
from tqdm import trange

from hocontact.models.honet import ManoAdaptor
from hocontact.postprocess.geo_loss import FieldLoss, ObjectLoss, HandLoss
from hocontact.utils import netutils
from hocontact.visualize.vis_contact_info import create_vertex_color

###########################added by xingyu
import sys
sys.path.append('/home/xingyu/project/ContactOpt')
sys.path.append('/home/xingyu/project/CPF')
import pytorch3d
import time
from pytorch3d.structures import Meshes
from contactopt.loader import *
from manopth.manolayer import ManoLayer
from manopth import rodrigues_layer
import hocontact.postprocess.diffcontact as calculate_contact
import hocontact.postprocess.util as util

from contactopt.hand_object import HandObject
from contactopt.visualize import show_optimization

class GeOptimizer:
    def __init__(
        self,
        device,
        lr=1e-2,
        n_iter=2500,
        verbose=False,
        mano_root="assets/mano",
        anchor_path="assets/anchor",
        fhb=False,
        load_fhb_path="assets/mano/fhb_skel_centeridx9.pkl",
        compensate_tsl=False,
        # values to initialize coef_val
        lambda_contact_loss=10.0,
        lambda_repulsion_loss=0.5,
        repulsion_query=0.030,
        repulsion_threshold=0.080,
    ):
        self.device = device
        self.lr = lr
        self.n_iter = n_iter

        # options
        self.verbose = verbose
        self.runtime_vis = None
        self.compensate_tsl = compensate_tsl

        # layers and loss utils
        self.mano_layer = ManoLayer(
            joint_rot_mode="quat",
            root_rot_mode="quat",
            use_pca=False,
            mano_root=mano_root,
            center_idx=9,
            flat_hand_mean=True,
            return_transf=True,
            return_full_pose=True,
        ).to(self.device)
        self.fhb = fhb
        if fhb:
            self.adaptor = ManoAdaptor(self.mano_layer, load_fhb_path).to(self.device)
            netutils.rec_freeze(self.adaptor)
        self.anchor_layer = AnchorLayer(anchor_path).to(self.device)
        self.axis_layer = AxisLayer().to(self.device)

        # opt val dict, const val dict
        self.opt_val = {}
        self.const_val = {}
        self.ctrl_val = {}
        self.coef_val = {
            "lambda_contact_loss": lambda_contact_loss,
            "lambda_repulsion_loss": lambda_repulsion_loss,
            "repulsion_query": repulsion_query,
            "repulsion_threshold": repulsion_threshold,
        }

        # creating slots for optimizer and scheduler
        self.optimizer = None
        self.optimizing = True
        self.scheduler = None

    def set_opt_val(
        self,
        # static val
        vertex_contact,  # TENSOR[NVERT, ] {0, 1}
        contact_region,  # TENSOR[NVERT, 1], int
        anchor_id,  # TENSOR[NVERT, 4]: int
        anchor_elasti,  # TENSOR[NVERT, 4]
        anchor_padding_mask,  # TENSOR[NVERT, 4] {0, 1}
        hand_region_assignment=None,  # TENSOR[NHANDVERT, ]
        hand_palm_vertex_mask=None,  # TENSOR[NHANDVERT, ] {0, 1}
        # dynamic val: hand
        hand_shape_gt=None,  # TENSOR[10, ]
        hand_tsl_gt=None,  # TENSOR[3, ]
        hand_pose_gt=None,  # (LIST[NPROV, ]: int {0..16}, TENSOR[NPROV, 4])
        hand_shape_init=None,  # TENSOR[10, ]
        hand_tsl_init=None,  # TENSOR[3, ]
        hand_pose_init=None,  # (LIST[NPROV, ]: int {0..16}, TENSOR[NPROV, 4])    (16,4)

        
        # dynamic val: obj
        obj_verts_3d_gt=None,
        obj_normals_gt=None,
        obj_verts_3d_can=None,
        obj_normals_can=None,
        obj_rot_init=None,
        obj_tsl_init=None,


        # hand tsl compensate
        hand_compensate_root=None,  # TENSOR[3, ]
        # runtime viz
        runtime_vis=None,


        # contactopt hand/obj contact:
        hand_contact_target=None, ###added by xingyu
        obj_contact_target= None, ###  added by xingyu
        data = None,  ## added by xingyu
        it = None,
        is_thin = None

    ):
        # ====== clear memory
        self.is_thin = is_thin
        self.data = data ## store the attributes of sample into self.data for further application
        self.hand_contact_target = hand_contact_target
        self.obj_contact_target = obj_contact_target

    
        self.opt_val = {}
        self.const_val = {}
        self.ctrl_val = {
            "optimize_hand_shape": False,
            "optimize_hand_tsl": False,
            "optimize_hand_pose": False,
            "optimize_obj": False,
            "fhb": self.fhb,
        }

        # ============ process static values >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # region
        vertex_contact = vertex_contact.long()
        anchor_id = anchor_id.long()
        anchor_padding_mask = anchor_padding_mask.long()

        # boolean index contact_region, anchor_id, anchor_elasti && anchor_padding_mask
        obj_contact_region = contact_region[vertex_contact == 1]  # TENSOR[NCONT, ]
        anchor_id = anchor_id[vertex_contact == 1, :]  # TENSOR[NCONT, 4]
        anchor_elasti = anchor_elasti[vertex_contact == 1, :]  # TENSOR[NCONT, 4]
        anchor_padding_mask = anchor_padding_mask[vertex_contact == 1, :]  # TENSOR[NCONT, 4]

        # boolean mask indexing anchor_id, anchor_elasti && obj_vert_id
        indexed_anchor_id = anchor_id[anchor_padding_mask == 1]  # TENSOR[NVALID, ]
        self.const_val["indexed_anchor_id"] = indexed_anchor_id
        self.const_val["indexed_anchor_elasti"] = anchor_elasti[anchor_padding_mask == 1]  # TENSOR[NVALID, ]

        vertex_id = torch.arange(anchor_id.shape[0])[:, None].repeat_interleave(
            anchor_padding_mask.shape[1], dim=1
        )  # TENSOR[NCONT, 4]
        self.const_val["indexed_vertex_id"] = vertex_id[anchor_padding_mask == 1]  # TENSOR[NVALID, ]

        tip_anchor_mask = torch.zeros(indexed_anchor_id.shape[0]).bool().to(self.device)
        tip_anchor_list = [2, 3, 4, 9, 10, 11, 15, 16, 17, 22, 23, 24, 29, 30, 31]
        for tip_anchor_id in tip_anchor_list:
            tip_anchor_mask = tip_anchor_mask | (self.const_val["indexed_anchor_id"] == tip_anchor_id)
        self.const_val["indexed_elasti_k"] = torch.where(
            tip_anchor_mask, torch.Tensor([1.0]).to(self.device), torch.Tensor([0.1]).to(self.device)
        ).to(self.device)

        # hand faces & edges
        self.const_val["hand_faces"] = self.mano_layer.th_faces
        self.const_val["static_verts"] = self.get_static_hand_verts()
        self.const_val["hand_edges"] = HandLoss.get_edge_idx(self.const_val["hand_faces"])
        self.const_val["static_edge_len"] = HandLoss.get_edge_len(
            self.const_val["static_verts"], self.const_val["hand_edges"]
        )
        # endregion
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        #self.data ============ dynamic val: hand >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # region
        # ====== hand_shape
        if hand_shape_gt is not None and hand_shape_init is not None:
            raise RuntimeError("both hand_shape gt and init are provided")
        elif hand_shape_gt is not None and hand_shape_init is None:
            self.const_val["hand_shape_gt"] = hand_shape_gt
            self.ctrl_val["optimize_hand_shape"] = False
        elif hand_shape_gt is None and hand_shape_init is not None:
            self.opt_val["hand_shape_var"] = hand_shape_init.detach().clone().requires_grad_(True)
            self.const_val["hand_shape_init"] = hand_shape_init
            self.ctrl_val["optimize_hand_shape"] = True
        else:
            # hand_tsl_gt is None and hand_tsl_init is None:
            self.opt_val["hand_shape_var"] = torch.zeros(10, dtype=torch.float, requires_grad=True, device=self.device)
            self.const_val["hand_shape_init"] = torch.zeros(10, dtype=torch.float, device=self.device)
            self.ctrl_val["optimize_hand_shape"] = True

        # ====== hand_tsl
        if hand_tsl_gt is not None and hand_tsl_init is not None:
            raise RuntimeError("both hand_tsl gt and init are provided")
        elif hand_tsl_gt is not None and hand_tsl_init is None:
            self.const_val["hand_tsl_gt"] = hand_tsl_gt
            self.ctrl_val["optimize_hand_tsl"] = False
        elif hand_tsl_gt is None and hand_tsl_init is not None:
            self.opt_val["hand_tsl_var"] = hand_tsl_init.detach().clone().requires_grad_(True)
            self.const_val["hand_tsl_init"] = hand_tsl_init
            self.ctrl_val["optimize_hand_tsl"] = True
        else:
            # hand_tsl_gt is None and hand_tsl_init is None:
            self.opt_val["hand_tsl_var"] = torch.zeros(3, dtype=torch.float, requires_grad=True, device=self.device)
            self.const_val["hand_tsl_init"] = torch.zeros(3, dtype=torch.float, device=self.device)
            self.ctrl_val["optimize_hand_tsl"] = True

        # ====== hand pose
        # this is complex! need special care
        if hand_pose_gt is not None and hand_pose_init is not None:# hand/hand_obj
            # full gt and init provided
            gt_pose_idx, gt_pose_val = hand_pose_gt  #(1,4) first row
            init_pose_idx, init_pose_val = hand_pose_init  #(15,4)  from info/intermediate (result of picr)
            # make sure they are complementary
            if len(set(gt_pose_idx).intersection(set(init_pose_idx))) > 0:
                raise RuntimeError("repeat hand_pose gt & init provided")
            if set(gt_pose_idx).union(set(init_pose_idx)) != set(range(16)):
                raise RuntimeError("hand_pose: not enough gt & init")
            self.const_val["hand_pose_gt_idx"] = gt_pose_idx
            self.const_val["hand_pose_gt_val"] = gt_pose_val
            self.const_val["hand_pose_var_idx"] = init_pose_idx
            self.opt_val["hand_pose_var_val"] = init_pose_val.detach().clone().requires_grad_(True) #(15,4)
            self.const_val["hand_pose_init_val"] = init_pose_val    
            self.ctrl_val["optimize_hand_pose"] = True
        elif hand_pose_gt is not None and hand_pose_init is None:#obj
            gt_pose_idx, gt_pose_val = hand_pose_gt
            self.const_val["hand_pose_gt_idx"] = gt_pose_idx
            self.const_val["hand_pose_gt_val"] = gt_pose_val
            if set(gt_pose_idx) == set(range(16)):
                # full gt provided
                self.const_val["hand_pose_var_idx"] = []
                self.opt_val["hand_pose_var_val"] = torch.zeros((0, 4), dtype=torch.float, device=self.device)
                self.ctrl_val["optimize_hand_pose"] = False
            else:
                # partial gt provided    (eg. only the first row)
                var_pose_idx = self.get_var_pose_idx(gt_pose_idx)
                n_var_pose = len(var_pose_idx)
                init_val = np.array([[0.9999, 0.0, -0.0101, 0.0]] * n_var_pose).astype(np.float32)
                self.const_val["hand_pose_var_idx"] = var_pose_idx
                self.opt_val["hand_pose_var_val"] = torch.tensor(
                    init_val, dtype=torch.float, requires_grad=True, device=self.device
                )
                init_val_true = np.array([[1.0, 0.0, 0.0, 0.0]] * n_var_pose).astype(np.float32)
                self.const_val["hand_pose_init_val"] = torch.tensor(                      
                    init_val_true, dtype=torch.float, device=self.device
                )
                self.ctrl_val["optimize_hand_pose"] = True
        elif hand_pose_gt is None and hand_pose_init is not None:
            # full init provided
            init_pose_idx, init_pose_val = hand_pose_init
            if set(init_pose_idx) != set(range(16)):  #0~15
                raise RuntimeError("hand_pose: not enough init")
            self.const_val["hand_pose_gt_idx"] = []
            self.const_val["hand_pose_gt_val"] = torch.zeros((0, 4), dtype=torch.float).to(self.device)
            self.const_val["hand_pose_var_idx"] = init_pose_idx
            self.const_val["hand_pose_gt_val"] = torch.zeros((0, 4), dtype=torch.float).to(self.device)
            self.const_val["hand_pose_var_idx"] = list(range(16))
            n_var_pose = 16
            init_val = np.array([[0.9999, 0.0, -0.0101, 0.0]] * n_var_pose).astype(np.float32)
            self.opt_val["hand_pose_var_val"] = torch.tensor(
                init_val, dtype=torch.float, requires_grad=True, device=self.device
            )
            init_val_true = np.array([[1.0, 0.0, 0.0, 0.0]] * n_var_pose).astype(np.float32)
            self.const_val["hand_pose_init_val"] = torch.tensor(init_val_true, dtype=torch.float, device=self.device)
            self.ctrl_val["optimize_hand_pose"] = True
        # endregion
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # ============ dynamic val: obj >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # region
        if obj_verts_3d_gt is not None and obj_verts_3d_can is not None:
            raise RuntimeError("optimize obj mode conflict: both provided")
        elif obj_verts_3d_gt is None and obj_verts_3d_can is None:
            raise RuntimeError("optimize obj mode conflict: neither proided")
        elif obj_verts_3d_gt is not None and obj_verts_3d_can is None:
            self.const_val["obj_verts_3d_gt"] = obj_verts_3d_gt[vertex_contact == 1, :]
            self.const_val["obj_normals_gt"] = obj_normals_gt[vertex_contact == 1, :]
            self.ctrl_val["optimize_obj"] = False
            self.const_val["full_obj_verts_3d"] = obj_verts_3d_gt
            self.const_val["full_obj_normals"] = obj_normals_gt
        else:
            # obj_verts_3d_gt is None and obj_verts_3d_can is not None
            self.const_val["obj_verts_3d_can"] = obj_verts_3d_can[vertex_contact == 1, :]
            self.const_val["obj_normals_can"] = obj_normals_can[vertex_contact == 1, :]
            self.ctrl_val["optimize_obj"] = True
            self.const_val["full_obj_verts_3d"] = obj_verts_3d_can
            self.const_val["full_obj_normals"] = obj_normals_can
            # check if init value is provided
            if obj_rot_init is None and obj_tsl_init is None:
                # both not provided
                self.opt_val["obj_rot_var"] = torch.tensor(
                    [0.001, 0.001, 0.001], dtype=torch.float, requires_grad=True, device=self.device
                )
                self.opt_val["obj_tsl_var"] = torch.zeros(3, dtype=torch.float, requires_grad=True, device=self.device)
                self.const_val["obj_rot_init"] = torch.tensor(
                    [0.000, 0.000, 0.000], dtype=torch.float, device=self.device
                )
                self.const_val["obj_tsl_init"] = torch.zeros(3, dtype=torch.float, device=self.device)
            elif obj_rot_init is not None and obj_tsl_init is not None:
                self.opt_val["obj_rot_var"] = obj_rot_init.detach().clone().requires_grad_(True)
                self.opt_val["obj_tsl_var"] = obj_tsl_init.detach().clone().requires_grad_(True)
                self.const_val["obj_rot_init"] = obj_rot_init
                self.const_val["obj_tsl_init"] = obj_tsl_init
            else:
                raise RuntimeError("incomplete init for optimize obj")
        # endregion
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # ============ compensate tsl (when fhb adapt layer is not used) >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        if self.compensate_tsl and (hand_tsl_gt is not None or hand_tsl_init is not None):
            if hand_compensate_root is None:
                raise RuntimeError("if need to compensate hand root tsl, correct root pos is requried")
            _, curr_joints, _ = self.recover_hand()
            compensate_offset = hand_compensate_root - curr_joints[0, ...]
            if not self.ctrl_val["optimize_hand_tsl"]:
                self.const_val["hand_tsl_gt"] = hand_tsl_gt + compensate_offset
            else:
                self.opt_val["hand_tsl_var"] = (hand_tsl_init + compensate_offset).detach().clone().requires_grad_(True)
                self.const_val["hand_tsl_init"] = hand_tsl_init + compensate_offset
        
        # if it >=0: #set the default value for opt_vector
        #     batch_size = self.data['hand_pose_aug'].shape[0]    
        #     device = self.data['hand_pose_aug'].device
        #     ncomps = 15
        #     opt_vector = torch.zeros((batch_size, ncomps + 6 + 3), device=device)   # 3 hand rot, 3 hand trans, 3 obj rot
        #     opt_vector.requires_grad = True
        #     self.opt_val["opt_vector"] = opt_vector
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # ============ construct optimizer & scheduler >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # region
        # ====== optimizer
        if (
            self.ctrl_val["optimize_hand_shape"]
            or self.ctrl_val["optimize_hand_tsl"]
            or self.ctrl_val["optimize_hand_pose"]
            or self.ctrl_val["optimize_obj"]
        ):
            # dispatch lr to different param
            param = []
            if self.ctrl_val["optimize_hand_shape"]:
                param.append({"params": [self.opt_val["hand_shape_var"]]})
            if self.ctrl_val["optimize_hand_tsl"]:
                param.append({"params": [self.opt_val["hand_tsl_var"]], "lr": 0.1 * self.lr})
            if self.ctrl_val["optimize_hand_pose"]:
                param.append({"params": [self.opt_val["hand_pose_var_val"]]})
            if self.ctrl_val["optimize_obj"]:
                param.append({"params": [self.opt_val["obj_rot_var"]]})
                param.append({"params": [self.opt_val["obj_tsl_var"]], "lr": 0.1 * self.lr})

            # added by xingyu
            #param.append({"params":[self.opt_val["opt_vector"]]})

            self.optimizer = torch.optim.Adam(param, lr=self.lr)
            self.optimizing = True
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, min_lr=1e-5, mode="min", factor=0.5, patience=20, verbose=False
            )
        else:
            self.optimizing = False
        # endregion
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # ====== runtime viz
        self.runtime_vis = runtime_vis

        # ====== verbose
        if self.verbose:
            print("Optimizing: ", self.optimizing)
            pprint(self.ctrl_val)
            pprint(list(self.opt_val.keys()))
            pprint(list(self.const_val.keys()))
            pprint(self.coef_val)

    @staticmethod
    def get_var_pose_idx(sel_pose_idx):
        # gt has 16 pose
        all_pose_idx = set(range(16))
        sel_pose_idx_set = set(sel_pose_idx)
        var_pose_idx = all_pose_idx.difference(sel_pose_idx_set)
        return list(var_pose_idx)

    def get_static_hand_verts(self):
        init_val_pose = np.array([[1.0, 0.0, 0.0, 0.0]] * 16).astype(np.float32)
        vec_pose = torch.tensor(init_val_pose).reshape(-1).unsqueeze(0).float().to(self.device)
        vec_shape = torch.zeros(1, 10).float().to(self.device)
        v, j, t, _ = self.mano_layer(vec_pose, vec_shape)
        v = v.squeeze(0)
        return v

    @staticmethod
    def assemble_pose_vec(gt_idx, gt_pose, var_idx, var_pose):
        idx_tensor = torch.cat((torch.Tensor(gt_idx).long(), torch.Tensor(var_idx).long()))
        pose_tensor = torch.cat((gt_pose, var_pose), dim=0)
        pose_tensor = pose_tensor[torch.argsort(idx_tensor)]
        return pose_tensor

    @staticmethod
    def transf_vectors(vectors, tsl, rot):
        """
        vectors: [K, 3], tsl: [3, ], rot: [3, ]
        return: [K, 3]
        """
        rot_matrix = batch_rodrigues(rot.unsqueeze(0)).squeeze(0).reshape((3, 3))
        vec = (rot_matrix @ vectors.T).T
        vec = vec + tsl
        return vec
     

    def loss_fn(self, opt_val, const_val, ctrl_val, coef_val,hand_contact_target,obj_contact_target,n_iter=250, lr=0.01, w_cont_hand=2, w_cont_obj=1,
                  save_history=False, ncomps=15, w_cont_asym=2, w_opt_trans=0.3, w_opt_pose=1, w_opt_rot=1,
                  caps_top=0.0005, caps_bot=-0.001, caps_rad=0.001, caps_on_hand=False,
                  contact_norm_method=0, w_pen_cost=600, w_obj_rot=0, pen_it=0):

        #loss_contactopt = self.ho_contact(opt_val, const_val,obj_contact_target,hand_contact_target)
        batch_size = 1   #data['hand_pose_aug'].shape[0]    here batch size become 18 in the 2. iteration,should be 1  
        device = const_val['hand_pose_gt_val'].device
        # print('12',data['hand_pose_aug'].shape)  #1,18?
        # print('10',data['hand_beta_aug'].shape)  # 1,10
        # print('11',data['hand_mTc_aug'].shape)   #1,4,4
        # print('11',data['hand_mTc_aug'].dtype)   #float 32 needed

        # opt_vector= self.opt_val["opt_vector"]

        mano_model = ManoLayer(mano_root='mano/models', use_pca=True, ncomps=ncomps, side='right', flat_hand_mean=False).to(device)
        
        #obj_mesh = Meshes(verts=[torch.Tensor(self.obj_verts)], faces=[torch.Tensor(self.obj_faces)])



        # optimizer = torch.optim.Adam([opt_vector], lr=lr, amsgrad=True)  # AMSgrad helps
        loss_criterion = torch.nn.L1Loss(reduction='none')  # Benchmarked, L1 performs best vs MSE/SmoothL1

        opt_state = []
        #is_thin = mesh_is_thin(data['mesh_aug'].num_verts_per_mesh())
        #mano_pose_out = torch.cat([opt_vector[:, 0:3] * w_opt_rot, opt_vector[:, 3:ncomps+3] * w_opt_pose], dim=1)

        #print('9',mano_pose_out)# tensor cuda grad

        # rebuild ho contact from contactopt in each iteration to compute the actual contact value
        # in such case the two projects are combined

        #print(const_val['hand_pose_gt_val'],opt_val['hand_pose_var_val'])
        #tensor cuda grad
        
        quaternion= torch.cat((const_val['hand_pose_gt_val'],opt_val['hand_pose_var_val']),dim=0)  #(16,4)
        #print('1',quaternion) #tensor cuda grad
        axiang = quaternion_to_angle_axis(quaternion).detach().cpu().numpy()  #(16,3)
        #print('2',axiang)# array 2d
        axiang_flatten = axiang.flatten()  #(48,)
        #print('3',axiang_flatten)# 1d array
        hand_pose_curr = util.fit_pca_to_axang(axiang_flatten,opt_val["hand_shape_var"].detach().cpu().numpy())
        #print('4',data['hand_pose_aug']) #1d array (18,)
        hand_pose_curr = torch.tensor(hand_pose_curr).to(device)
        #print('5',data['hand_pose_aug'])# tensor cuda
        hand_shape_curr = opt_val["hand_shape_var"]
        #print('6',data['hand_beta_aug'].shape) #tensor cuda grad     now [10] should be reshape ([1,10])
        hand_shape_curr = hand_shape_curr.unsqueeze(0) #.reshape(1, -1)
        hand_mTc =np.eye(4)
        hand_mTc[:3, 3] = opt_val["hand_tsl_var"].detach().cpu().numpy()
        #print('7',hand_mTc) #2d array
        hand_mTc_curr = torch.tensor(hand_mTc).to(device).unsqueeze(0).float()  #float32 instead of float64/double
    
        
        
        hand_pose_curr= hand_pose_curr.unsqueeze(0) #  (18,) PCA
        #tform_out = util.translation_to_tform(opt_vector[:, ncomps+3:ncomps+6] * w_opt_trans) #  (batch, 3) to (batch, 4, 4)
        
        #print('dtype')
        #print(mano_pose_out.shape, data['hand_beta_aug'].shape)  #torch.Size([1, 18]) torch.Size([1, 10])
        #print(data['hand_mTc_aug'].shape, tform_out.shape)#torch.Size([4, 4]) torch.Size([1, 4, 4])
        #print(data['hand_mTc_aug'].dtype, tform_out.dtype)

        hand_verts, hand_joints = util.forward_mano(mano_model, hand_pose_curr, hand_shape_curr, [hand_mTc_curr])   # 2.2ms
        # data['hand_mTc_aug'] ----(batch,4,4)?  tsl
        if contact_norm_method != 0 and not caps_on_hand:
            with torch.no_grad():   # We need to calculate hand normals if using more complicated methods
                mano_mesh_curr = Meshes(verts=hand_verts, faces=mano_model.th_faces.repeat(batch_size, 1, 1))
                hand_normals_curr = mano_mesh_curr.verts_normals_padded()
        else:
            hand_normals_curr = torch.zeros(hand_verts.shape, device=device)
        #print(data['obj_normals_aug'].shape)
        # torch.Size([1, 2361, 3])

        obj_normals_curr=const_val['full_obj_normals']
        #print(data['obj_normals_aug'].unsqueeze(0).shape, data['obj_sampled_idx'].shape)
                    # torch.Size([1,2361, 3]) torch.Size([1, 2048])
        SAMPLE_VERTS_NUM =2048
        obj_verts_curr = const_val["full_obj_verts_3d"]
        ##to be debugged:                            opt_vector[:, ncomps+6:]                                       
        obj_rot_mat = rodrigues_layer.batch_rodrigues(opt_val["obj_rot_var"].reshape((-1,3)))
        obj_rot_mat = obj_rot_mat.view(batch_size, 3, 3)

        if w_obj_rot > 0:
            obj_verts_curr = util.apply_rot(obj_rot_mat, obj_verts_curr, around_centroid=True)
            obj_normals_curr = util.apply_rot(obj_rot_mat, obj_normals_curr)

        sample_idx = torch.Tensor(np.random.randint(0, len(obj_verts_curr), SAMPLE_VERTS_NUM)).long()
        
        if sample_idx.numel() > 1:  #2048
            sampled_obj_normals_curr = util.batched_index_select(obj_normals_curr.unsqueeze(0), 1, sample_idx.unsqueeze(0).to(device))
        else:   # If we're optimizing over all verts
            sampled_obj_normals_curr = obj_normals_curr
        #print('obj_normals_sampled',obj_normals_sampled)
        # print(data['obj_verts_aug'].shape)
        # print(const_val["obj_verts_3d_can"].shape)
        
        sampled_obj_verts_curr= obj_verts_curr[sample_idx, :]
        sampled_obj_verts_curr = sampled_obj_verts_curr.unsqueeze(0)
        #print(data['obj_sampled_verts_aug'].shape) # torch.Size([1, 2048, 3])
        # obj_verts = data['obj_sampled_verts_aug']
        # obj_normals = sampled_obj_normals_curr

        # obj_rot_mat = rodrigues_layer.batch_rodrigues(opt_vector[:, ncomps+6:])
        # obj_rot_mat = obj_rot_mat.view(batch_size, 3, 3)

        # if w_obj_rot > 0:
        #     obj_verts = util.apply_rot(obj_rot_mat, obj_verts, around_centroid=True)
        #     obj_normals = util.apply_rot(obj_rot_mat, obj_normals)
        # print(hand_verts.shape, hand_normals.shape, obj_verts.shape, obj_normals.shape)
        # torch.Size([1, 778, 3]) torch.Size([1, 778, 3]) torch.Size([1, 2048, 3]) torch.Size([1, 2048, 3])
        
        contact_obj, contact_hand = calculate_contact.calculate_contact_capsule(hand_verts, hand_normals_curr, sampled_obj_verts_curr, sampled_obj_normals_curr,
                            caps_top=caps_top, caps_bot=caps_bot, caps_rad=caps_rad, caps_on_hand=caps_on_hand, contact_norm_method=contact_norm_method)

        contact_obj_sub = obj_contact_target - contact_obj
        contact_obj_weighted = contact_obj_sub + torch.nn.functional.relu(contact_obj_sub) * w_cont_asym  # Loss for 'missing' contact higher
        loss_contact_obj = loss_criterion(contact_obj_weighted, torch.zeros_like(contact_obj_weighted)).mean(dim=(1, 2))

        contact_hand_sub = hand_contact_target - contact_hand
        contact_hand_weighted = contact_hand_sub + torch.nn.functional.relu(contact_hand_sub) * w_cont_asym  # Loss for 'missing' contact higher
        loss_contact_hand = loss_criterion(contact_hand_weighted, torch.zeros_like(contact_hand_weighted)).mean(dim=(1, 2))

        loss_contactopt = loss_contact_obj * w_cont_obj + loss_contact_hand * w_cont_hand

        # if w_pen_cost > 0 and it >= pen_it:
        if w_pen_cost > 0:
            pen_cost = calculate_contact.calculate_penetration_cost(hand_verts, hand_normals_curr, sampled_obj_verts_curr, sampled_obj_normals_curr, self.is_thin, contact_norm_method)
            loss_contactopt += pen_cost.mean(dim=1) * w_pen_cost


        var_hand_pose_assembled = self.assemble_pose_vec(
            const_val["hand_pose_gt_idx"],
            const_val["hand_pose_gt_val"],
            const_val["hand_pose_var_idx"],
            opt_val["hand_pose_var_val"],  #(15,3)
        )

        # dispatch hand var
        vec_pose = var_hand_pose_assembled.unsqueeze(0)
        if ctrl_val["optimize_hand_shape"]:
            vec_shape = opt_val["hand_shape_var"].unsqueeze(0)
        else:
            vec_shape = const_val["hand_shape_gt"].unsqueeze(0)
        if ctrl_val["optimize_hand_tsl"]:
            vec_tsl = opt_val["hand_tsl_var"].unsqueeze(0)
        else:
            vec_tsl = const_val["hand_tsl_gt"].unsqueeze(0)

        # rebuild hand
        rebuild_verts, rebuild_joints, rebuild_transf, rebuild_full_pose = self.mano_layer(vec_pose, vec_shape)
        # skel adaption
        if ctrl_val["fhb"]:
            adapt_joints, _ = self.adaptor(rebuild_verts)
            adapt_joints = adapt_joints.transpose(1, 2)
            rebuild_joints = rebuild_joints - adapt_joints[:, 9].unsqueeze(1)
            rebuild_verts = rebuild_verts - adapt_joints[:, 9].unsqueeze(1)
        rebuild_joints = rebuild_joints + vec_tsl
        rebuild_verts = rebuild_verts + vec_tsl
        rebuild_transf = rebuild_transf + torch.cat(
            [
                torch.cat([torch.zeros(3, 3).to(self.device), vec_tsl.view(3, -1)], dim=1),
                torch.zeros(1, 4).to(self.device),
            ],
            dim=0,
        )
        rebuild_verts_squeezed = rebuild_verts.squeeze(0)

        # rebuild anchor
        rebuild_anchor = self.anchor_layer(rebuild_verts)
        rebuild_anchor = rebuild_anchor.contiguous()  # TENSOR[1, 32, 3]
        rebuild_anchor = rebuild_anchor.squeeze(0)  # TENSOR[32, 3]
        anchor_pos = rebuild_anchor[const_val["indexed_anchor_id"]]  # TENSOR[NVALID, 3]

        # dispatch obj var
        if ctrl_val["optimize_obj"]:
            obj_verts = self.transf_vectors(
                const_val["obj_verts_3d_can"],
                opt_val["obj_tsl_var"],
                opt_val["obj_rot_var"],
            )
            full_obj_verts = self.transf_vectors(
                const_val["full_obj_verts_3d"],
                opt_val["obj_tsl_var"],
                opt_val["obj_rot_var"],
            )
            full_obj_normals = self.transf_vectors(
                const_val["full_obj_normals"],
                torch.zeros(3, dtype=torch.float, device=self.device),
                opt_val["obj_rot_var"],
            )
        else:
            obj_verts = const_val["obj_verts_3d_gt"]
            full_obj_verts = const_val["full_obj_verts_3d"]
            full_obj_normals = const_val["full_obj_normals"]

        # contact loss
        contact_loss = FieldLoss.contact_loss(
            anchor_pos,
            obj_verts[const_val["indexed_vertex_id"]],
            const_val["indexed_anchor_elasti"],
            const_val["indexed_elasti_k"],
        )
        # repulsion loss
        repulsion_loss = FieldLoss.full_repulsion_loss(
            rebuild_verts_squeezed,
            full_obj_verts,
            full_obj_normals,
            query=coef_val["repulsion_query"],
            threshold=coef_val["repulsion_threshold"],
        )

        if ctrl_val["optimize_hand_pose"]:
            # get hand loss
            quat_norm_loss = HandLoss.pose_quat_norm_loss(var_hand_pose_assembled)
            var_hand_pose_normalized = normalize_quaternion(var_hand_pose_assembled)
            pose_reg_loss = HandLoss.pose_reg_loss(
                var_hand_pose_normalized[const_val["hand_pose_var_idx"]], const_val["hand_pose_init_val"]
            )

            b_axis, u_axis, l_axis = self.axis_layer(rebuild_joints, rebuild_transf)

            angle_axis = quaternion_to_angle_axis(var_hand_pose_normalized.reshape((16, 4)))
            angle_axis = angle_axis[1:, :]  # ignore global rot [15, 3]
            axis = angle_axis / torch.norm(angle_axis, dim=1, keepdim=True)
            angle = torch.norm(angle_axis, dim=1, keepdim=False)
            # limit angle
            angle_limit_loss = HandLoss.rotation_angle_loss(angle)

            joint_b_axis_loss = HandLoss.joint_b_axis_loss(b_axis, axis)
            joint_u_axis_loss = HandLoss.joint_u_axis_loss(u_axis, axis)
            joint_l_limit_loss = HandLoss.joint_l_limit_loss(l_axis, axis)

            edge_loss = HandLoss.edge_len_loss(
                rebuild_verts_squeezed, const_val["hand_edges"], const_val["static_edge_len"]
            )
        else:
            quat_norm_loss = torch.Tensor([0.0]).to(self.device)
            pose_reg_loss = torch.Tensor([0.0]).to(self.device)
            angle_limit_loss = torch.Tensor([0.0]).to(self.device)
            joint_b_axis_loss = torch.Tensor([0.0]).to(self.device)
            joint_u_axis_loss = torch.Tensor([0.0]).to(self.device)
            joint_l_limit_loss = torch.Tensor([0.0]).to(self.device)
            edge_loss = torch.Tensor([0.0]).to(self.device)
            # pose_reg_loss_to_zero = torch.Tensor([0.0]).to(self.device)

        if ctrl_val["optimize_hand_shape"]:
            shape_reg_loss = HandLoss.shape_reg_loss(opt_val["hand_shape_var"], const_val["hand_shape_init"])
        else:
            shape_reg_loss = torch.Tensor([0.0]).to(self.device)

        if ctrl_val["optimize_hand_tsl"]:
            hand_tsl_loss = HandLoss.hand_tsl_loss(opt_val["hand_tsl_var"], const_val["hand_tsl_init"])
        else:
            hand_tsl_loss = torch.Tensor([0.0]).to(self.device)

        if ctrl_val["optimize_obj"]:
            obj_transf_loss = ObjectLoss.obj_transf_loss(
                opt_val["obj_tsl_var"], opt_val["obj_rot_var"], const_val["obj_tsl_init"], const_val["obj_rot_init"]
            )
        else:
            obj_transf_loss = torch.Tensor([0.0]).to(self.device)

        loss = (
            # ============= HAND ANATOMICAL LOSS
            1.0 * quat_norm_loss
            + 1.0 * angle_limit_loss
            + 1.0 * edge_loss
            + 0.1 * joint_b_axis_loss
            + 0.1 * joint_u_axis_loss
            + 0.1 * joint_l_limit_loss
            # ============= ELAST POTENTIONAL ENERGY
            + coef_val["lambda_contact_loss"] * contact_loss  #10
            + coef_val["lambda_repulsion_loss"] * repulsion_loss   #0.5
            # ============= OFFSET LOSS
            + 1.0 * pose_reg_loss
            + 1.0 * shape_reg_loss
            + 1.0 * hand_tsl_loss
            + 1.0 * obj_transf_loss

            + 1.0 * loss_contactopt
        )

        # debug: runtime viz
        if self.runtime_vis:
            if self.ctrl_val["optimize_obj"]:
                full_obj_verts = self.transf_vectors(
                    self.const_val["full_obj_verts_3d"],
                    self.opt_val["obj_tsl_var"].detach(),
                    self.opt_val["obj_rot_var"].detach(),
                )
            else:
                full_obj_verts = self.const_val["full_obj_verts_3d"]

            if not ctrl_val["optimize_hand_pose"]:
                b_axis, u_axis, l_axis = self.axis_layer(rebuild_joints, rebuild_transf)  # mend this up
            self.runtime_show(rebuild_verts, b_axis, u_axis, l_axis, rebuild_transf, full_obj_verts)

        return (
            loss,
            {
                "quat_norm_loss": quat_norm_loss.detach().cpu().item(),
                "angle_limit_loss": angle_limit_loss.detach().cpu().item(),
                "edge_loss": edge_loss.detach().cpu().item(),
                "joint_b_axis_loss": joint_b_axis_loss.detach().cpu().item(),
                "joint_u_axis_loss": joint_u_axis_loss.detach().cpu().item(),
                "joint_l_limit_loss": joint_l_limit_loss.detach().cpu().item(),
                "contact_loss": contact_loss.detach().cpu().item(),
                "repulsion_loss": repulsion_loss.detach().cpu().item(),
                "pose_reg_loss": pose_reg_loss.detach().cpu().item(),
                # where is shape_reg_loss??
                "hand_tsl_loss": hand_tsl_loss.detach().cpu().item(),
                "obj_transf_loss": obj_transf_loss.detach().cpu().item(),
                'ho_contact_loss': loss_contactopt.detach().cpu().item(),
            },
        )

    def optimize(self, progress=False):
        if progress:
            bar = trange(self.n_iter, position=4)
            bar_hand = trange(0, position=2, bar_format="{desc}")
            bar_contact = trange(0, position=1, bar_format="{desc}")
            bar_axis = trange(0, position=0, bar_format="{desc}")

            bar_contactopt = trange(0, position=3, bar_format="{desc}")
        else:
            bar = range(self.n_iter)     # no visual effect  250 iter

        loss = torch.Tensor([1000.0]).to(self.device)
        loss_dict = {}
        it = 0
        for _ in bar:
            
            if self.optimizing:
                self.optimizer.zero_grad()
                                          
            loss, loss_dict = self.loss_fn(self.opt_val, self.const_val, self.ctrl_val, self.coef_val,self.hand_contact_target,self.obj_contact_target)

            if self.optimizing:
                loss.backward()
                self.optimizer.step()
                self.scheduler.step(loss)

            if progress:
                bar.set_description("TOTAL LOSS {:4e}".format(loss.item()))
                try:
                    bar_hand.set_description(
                        colored("HAND_REGUL_LOSS: ", "yellow")
                        + "QN={:.3e} PR={:.3e} EG={:.3e}".format(
                            loss_dict["quat_norm_loss"],  # QN
                            loss_dict["pose_reg_loss"],  # PR
                            loss_dict["edge_loss"],  # Edge
                        )
                    )
                except:
                    pass
                try:
                    bar_contact.set_description(
                        colored("HO_CONTACT_LOSS_CPF: ", "blue")
                        + "Conta={:.3e}, Repul={:.3e}, OT={:.3e}".format(
                            loss_dict["contact_loss"],  # Conta
                            loss_dict["repulsion_loss"],  # Repul
                            loss_dict["obj_transf_loss"],  # OT
                        )
                    )
                except:
                    pass
                try:
                    bar_axis.set_description(
                        colored("ANGLE_LOSS: ", "cyan")
                        + "AL={:.3e} JB={:.3e} JU={:.3e} JL={:.3e}".format(
                            loss_dict["angle_limit_loss"],  # AL
                            loss_dict["joint_b_axis_loss"],  # JB
                            loss_dict["joint_u_axis_loss"],  # JU
                            loss_dict["joint_l_limit_loss"],  # JL
                        )
                    )
                except:
                    pass
                
                try:
                    bar_contactopt.set_description(colored('HO_CONTACT_LOSS_CONTACTOPT:','green')
                                                   + '{:3e}'.format(loss_dict['ho_contact_loss']))
                except Exception as e:
                    print(f"Unexpected error: {e}")
                    pass
            it += 1
        return loss.item(), loss_dict

    def recover_hand(self, squeeze_out=True):
        vars_hand_pose_assembled = self.assemble_pose_vec(
            self.const_val["hand_pose_gt_idx"],
            self.const_val["hand_pose_gt_val"],
            self.const_val["hand_pose_var_idx"],
            self.opt_val["hand_pose_var_val"],
        ).detach()
        vars_hand_pose_normalized = normalize_quaternion(vars_hand_pose_assembled)
        vec_pose = vars_hand_pose_normalized.unsqueeze(0)
        if self.ctrl_val["optimize_hand_shape"]:
            vec_shape = self.opt_val["hand_shape_var"].detach().unsqueeze(0)
        else:
            vec_shape = self.const_val["hand_shape_gt"].unsqueeze(0)
        if self.ctrl_val["optimize_hand_tsl"]:
            vec_tsl = self.opt_val["hand_tsl_var"].detach().unsqueeze(0)
        else:
            vec_tsl = self.const_val["hand_tsl_gt"].unsqueeze(0)

        device = vec_pose.device
        rebuild_verts, rebuild_joints, rebuild_transf, rebuild_full_pose = self.mano_layer(vec_pose, vec_shape)
        # skel adaption
        if self.ctrl_val["fhb"]:
            adapt_joints, _ = self.adaptor(rebuild_verts)
            adapt_joints = adapt_joints.transpose(1, 2)
            rebuild_joints = rebuild_joints - adapt_joints[:, 9].unsqueeze(1)
            rebuild_verts = rebuild_verts - adapt_joints[:, 9].unsqueeze(1)
        rebuild_verts = rebuild_verts + vec_tsl
        rebuild_joints = rebuild_joints + vec_tsl
        rebuild_transf = rebuild_transf + torch.cat(
            [
                torch.cat((torch.zeros((3, 3), device=device), vec_tsl.T), dim=1),
                torch.zeros((1, 4), device=device),
            ],
            dim=0,
        )
        if squeeze_out:
            rebuild_verts, rebuild_joints, rebuild_transf = (
                rebuild_verts.squeeze(0),
                rebuild_joints.squeeze(0),
                rebuild_transf.squeeze(0),
            )
        return rebuild_verts, rebuild_joints, rebuild_transf

    def recover_hand_pose(self):
        vars_hand_pose_assembled = self.assemble_pose_vec(
            self.const_val["hand_pose_gt_idx"],
            self.const_val["hand_pose_gt_val"],
            self.const_val["hand_pose_var_idx"],
            self.opt_val["hand_pose_var_val"],
        ).detach()
        vars_hand_pose_normalized = normalize_quaternion(vars_hand_pose_assembled)
        return vars_hand_pose_normalized

    def recover_obj(self):
        if self.ctrl_val["optimize_obj"]:
            obj_verts = self.transf_vectors(
                self.const_val["full_obj_verts_3d"],
                self.opt_val["obj_tsl_var"].detach(),
                self.opt_val["obj_rot_var"].detach(),
            )
        else:
            obj_verts = self.const_val["full_obj_verts_3d"]
        return obj_verts

    def obj_rot_np(self):
        if self.ctrl_val["optimize_obj"]:
            res = self.opt_val["obj_rot_var"].detach().cpu().numpy()
            return res
        else:
            raise RuntimeError("not optimizing obj, cannot get obj_rot")

    def obj_tsl_np(self):
        if self.ctrl_val["optimize_obj"]:
            res = self.opt_val["obj_tsl_var"].detach().cpu().numpy()
            return res
        else:
            raise RuntimeError("not optimizing obj, cannot get obj_tsl")

    def runtime_show(self, hand_verts, b_axis, u_axis, l_axis, hand_transf, obj_verts):
        has_rot = False
        b_axis = b_axis.detach().cpu().squeeze(0).numpy()
        u_axis = u_axis.detach().cpu().squeeze(0).numpy()
        l_axis = l_axis.detach().cpu().squeeze(0).numpy()
        hand_transf = hand_transf.detach().cpu().squeeze(0).numpy()
        b_rot_ms = []
        u_rot_ms = []
        l_rot_ms = []

        while True:
            self.runtime_vis["hand_mesh"].vertices = o3d.utility.Vector3dVector(
                np.array(hand_verts.detach().cpu().squeeze(0))
            )
            self.runtime_vis["hand_mesh"].compute_vertex_normals()
            self.runtime_vis["obj_mesh"].vertices = o3d.utility.Vector3dVector(
                np.array(obj_verts.detach().cpu().squeeze(0))
            )
            self.runtime_vis["obj_mesh"].compute_vertex_normals()
            if not has_rot:
                for i in range(16):
                    if not i:
                        continue
                    b_rot = caculate_align_mat(b_axis[i - 1])
                    b_rot_ms.append(b_rot)
                    self.runtime_vis["b_axis"][i] = self.runtime_vis["b_axis"][i].rotate(b_rot, center=(0, 0, 0))
                    self.runtime_vis["b_axis"][i] = self.runtime_vis["b_axis"][i].rotate(
                        hand_transf[i][:3, :3], center=(0, 0, 0)
                    )
                    self.runtime_vis["b_axis"][i] = self.runtime_vis["b_axis"][i].translate(hand_transf[i][:3, 3].T)
                    self.runtime_vis["window"].update_geometry(self.runtime_vis["b_axis"][i])

                    u_rot = caculate_align_mat(u_axis[i - 1])
                    u_rot_ms.append(u_rot)
                    self.runtime_vis["up_axis"][i] = self.runtime_vis["up_axis"][i].rotate(u_rot, center=(0, 0, 0))
                    self.runtime_vis["up_axis"][i] = self.runtime_vis["up_axis"][i].rotate(
                        hand_transf[i][:3, :3], center=(0, 0, 0)
                    )
                    self.runtime_vis["up_axis"][i] = self.runtime_vis["up_axis"][i].translate(hand_transf[i][:3, 3].T)
                    self.runtime_vis["window"].update_geometry(self.runtime_vis["up_axis"][i])

                    l_rot = caculate_align_mat(l_axis[i - 1])
                    l_rot_ms.append(l_rot)
                    self.runtime_vis["l_axis"][i] = self.runtime_vis["l_axis"][i].rotate(l_rot, center=(0, 0, 0))
                    self.runtime_vis["l_axis"][i] = self.runtime_vis["l_axis"][i].rotate(
                        hand_transf[i][:3, :3], center=(0, 0, 0)
                    )
                    self.runtime_vis["l_axis"][i] = self.runtime_vis["l_axis"][i].translate(hand_transf[i][:3, 3].T)
                    self.runtime_vis["window"].update_geometry(self.runtime_vis["l_axis"][i])

                has_rot = True
            self.runtime_vis["window"].update_geometry(self.runtime_vis["hand_mesh"])
            self.runtime_vis["window"].update_geometry(self.runtime_vis["obj_mesh"])
            self.runtime_vis["window"].update_renderer()
            if not self.runtime_vis["window"].poll_events():
                break

        for i in range(16):
            if not i:
                continue
            self.runtime_vis["b_axis"][i] = self.runtime_vis["b_axis"][i].translate(-hand_transf[i][:3, 3].T)
            self.runtime_vis["b_axis"][i] = self.runtime_vis["b_axis"][i].rotate(
                hand_transf[i][:3, :3].T, center=(0, 0, 0)
            )
            self.runtime_vis["b_axis"][i] = self.runtime_vis["b_axis"][i].rotate(b_rot_ms[i - 1].T, center=(0, 0, 0))

            self.runtime_vis["up_axis"][i] = self.runtime_vis["up_axis"][i].translate(-hand_transf[i][:3, 3].T)
            self.runtime_vis["up_axis"][i] = self.runtime_vis["up_axis"][i].rotate(
                hand_transf[i][:3, :3].T, center=(0, 0, 0)
            )
            self.runtime_vis["up_axis"][i] = self.runtime_vis["up_axis"][i].rotate(u_rot_ms[i - 1].T, center=(0, 0, 0))

            self.runtime_vis["l_axis"][i] = self.runtime_vis["l_axis"][i].translate(-hand_transf[i][:3, 3].T)
            self.runtime_vis["l_axis"][i] = self.runtime_vis["l_axis"][i].rotate(
                hand_transf[i][:3, :3].T, center=(0, 0, 0)
            )
            self.runtime_vis["l_axis"][i] = self.runtime_vis["l_axis"][i].rotate(l_rot_ms[i - 1].T, center=(0, 0, 0))
        return


def caculate_align_mat(vec):
    vec = vec / np.linalg.norm(vec)
    z_unit_Arr = np.array([0, 0, 1])

    z_mat = np.array(
        [
            [0, -z_unit_Arr[2], z_unit_Arr[1]],
            [z_unit_Arr[2], 0, -z_unit_Arr[0]],
            [-z_unit_Arr[1], z_unit_Arr[0], 0],
        ]
    )

    z_c_vec = np.matmul(z_mat, vec)
    z_c_vec_mat = np.array(
        [
            [0, -z_c_vec[2], z_c_vec[1]],
            [z_c_vec[2], 0, -z_c_vec[0]],
            [-z_c_vec[1], z_c_vec[0], 0],
        ]
    )

    if np.dot(z_unit_Arr, vec) == -1:
        qTrans_Mat = -np.eye(3, 3)
    elif np.dot(z_unit_Arr, vec) == 1:
        qTrans_Mat = np.eye(3, 3)
    else:
        qTrans_Mat = np.eye(3, 3) + z_c_vec_mat + np.matmul(z_c_vec_mat, z_c_vec_mat) / (1 + np.dot(z_unit_Arr, vec))

    return qTrans_Mat


def init_runtime_viz(
    hand_verts_gt,
    hand_verts_init,
    obj_verts_gt,
    hand_faces,
    obj_verts_cur,
    obj_faces_cur,
    contact_info,
    cam_extr=None,
):
    hand_mesh_gt = o3d.geometry.TriangleMesh()
    hand_mesh_gt.triangles = o3d.utility.Vector3iVector(hand_faces)
    hand_mesh_gt.vertices = o3d.utility.Vector3dVector(hand_verts_gt)
    hand_mesh_gt.vertex_colors = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 1.0]] * len(hand_verts_gt)))
    hand_mesh_gt.compute_vertex_normals()
    obj_mesh_gt = o3d.geometry.TriangleMesh()
    obj_mesh_gt.triangles = o3d.utility.Vector3iVector(obj_faces_cur)
    obj_mesh_gt.vertices = o3d.utility.Vector3dVector(obj_verts_gt)
    obj_mesh_gt.vertex_colors = o3d.utility.Vector3dVector(np.array([[1.0, 0.0, 0.0]] * len(obj_verts_gt)))
    obj_mesh_gt.compute_vertex_normals()

    hand_mesh_cur = o3d.geometry.TriangleMesh()
    hand_mesh_cur.triangles = o3d.utility.Vector3iVector(hand_faces)
    obj_mesh = o3d.geometry.TriangleMesh()
    obj_mesh.triangles = o3d.utility.Vector3iVector(obj_faces_cur)
    obj_mesh.vertices = o3d.utility.Vector3dVector(obj_verts_cur)
    obj_colors = create_vertex_color(contact_info, "contact_region")
    obj_mesh.compute_vertex_normals()
    obj_mesh.vertex_colors = o3d.utility.Vector3dVector(obj_colors)
    vis_cur = o3d.visualization.VisualizerWithKeyCallback()
    vis_cur.create_window(window_name="Runtime Hand", width=1080, height=1080)
    vis_cur.add_geometry(obj_mesh)
    vis_cur.add_geometry(hand_mesh_cur)
    vis_cur.add_geometry(hand_mesh_gt)
    vis_cur.add_geometry(obj_mesh_gt)
    back_axis_list = []
    up_axis_list = []
    left_axis_list = []
    for i in range(16):
        b = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.0015,
            cone_radius=0.002,
            cylinder_height=0.05,
            cone_height=0.008,
            resolution=20,
            cylinder_split=4,
            cone_split=1,
        )
        b.paint_uniform_color([45 / 255.0, 220 / 255.0, 190 / 255.0])
        b.compute_vertex_normals()
        vis_cur.add_geometry(b)
        back_axis_list.append(b)

        u = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.0015,
            cone_radius=0.002,
            cylinder_height=0.04,
            cone_height=0.008,
            resolution=20,
            cylinder_split=4,
            cone_split=1,
        )
        u.paint_uniform_color([250 / 255.0, 100 / 255.0, 100 / 255.0])
        u.compute_vertex_normals()
        vis_cur.add_geometry(u)
        up_axis_list.append(u)

        l = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.0015,
            cone_radius=0.002,
            cylinder_height=0.04,
            cone_height=0.008,
            resolution=20,
            cylinder_split=4,
            cone_split=1,
        )
        l.paint_uniform_color([230 / 255.0, 120 / 255.0, 60 / 255.0])
        l.compute_vertex_normals()
        vis_cur.add_geometry(l)
        left_axis_list.append(l)

    vis_cur.poll_events()
    runtime_vis = {
        "hand_mesh_gt": hand_mesh_gt,
        "hand_mesh": hand_mesh_cur,
        "obj_mesh": obj_mesh,
        "obj_mesh_gt": obj_mesh_gt,
        "b_axis": back_axis_list,
        "up_axis": up_axis_list,
        "l_axis": left_axis_list,
        "window": vis_cur,
    }
    if cam_extr is not None:
        ctl = runtime_vis["window"].get_view_control()
        parameters = ctl.convert_to_pinhole_camera_parameters()
        parameters.extrinsic = cam_extr
        ctl.convert_from_pinhole_camera_parameters(parameters)
        # ro = runtime_vis["window"].get_render_option()
        # ro.load_from_json("./ro.json")

    def hide_gt(vis):
        vis.remove_geometry(hand_mesh_gt, reset_bounding_box=False)
        vis.remove_geometry(obj_mesh_gt, reset_bounding_box=False)
        vis.add_geometry(obj_mesh, reset_bounding_box=False)
        vis.add_geometry(hand_mesh_cur, reset_bounding_box=False)

        vis.update_renderer()
        vis.poll_events()

    def show_gt(vis):
        vis.add_geometry(hand_mesh_gt, reset_bounding_box=False)
        vis.add_geometry(obj_mesh_gt, reset_bounding_box=False)
        vis.remove_geometry(obj_mesh, reset_bounding_box=False)
        vis.remove_geometry(hand_mesh_cur, reset_bounding_box=False)

        vis.update_renderer()
        vis.poll_events()

    def axis_on(vis):
        for i in range(16):
            vis.add_geometry(back_axis_list[i], reset_bounding_box=False)
            vis.add_geometry(up_axis_list[i], reset_bounding_box=False)
            vis.add_geometry(left_axis_list[i], reset_bounding_box=False)
        vis.update_renderer()
        vis.poll_events()

    def axis_off(vis):
        for i in range(16):
            vis.remove_geometry(back_axis_list[i], reset_bounding_box=False)
            vis.remove_geometry(up_axis_list[i], reset_bounding_box=False)
            vis.remove_geometry(left_axis_list[i], reset_bounding_box=False)
        vis.update_renderer()
        vis.poll_events()

    vis_cur.register_key_callback(ord("A"), hide_gt)
    vis_cur.register_key_callback(ord("Z"), show_gt)
    vis_cur.register_key_callback(ord("U"), axis_on)
    vis_cur.register_key_callback(ord("I"), axis_off)

    return runtime_vis


def update_runtime_viz(
    runtime_vis,
    hand_verts_gt,
    hand_verts_curr,
    obj_verts_gt,
    obj_verts_curr,
    hand_faces,
    obj_faces_cur,
):
    runtime_vis["hand_mesh"].vertices = o3d.utility.Vector3dVector(hand_verts_curr)
    runtime_vis["hand_mesh"].triangles = o3d.utility.Vector3iVector(hand_faces)
    runtime_vis["hand_mesh"].compute_vertex_normals()
    runtime_vis["hand_mesh_gt"].vertices = o3d.utility.Vector3dVector(hand_verts_gt)
    runtime_vis["hand_mesh_gt"].triangles = o3d.utility.Vector3iVector(hand_faces)
    runtime_vis["hand_mesh_gt"].vertex_colors = o3d.utility.Vector3dVector(
        np.array([[0.0, 0.0, 1.0]] * len(hand_verts_gt))
    )
    runtime_vis["hand_mesh_gt"].compute_vertex_normals()
    runtime_vis["obj_mesh"].vertices = o3d.utility.Vector3dVector(obj_verts_curr)
    runtime_vis["obj_mesh"].triangles = o3d.utility.Vector3iVector(obj_faces_cur)
    runtime_vis["obj_mesh"].vertex_colors = o3d.utility.Vector3dVector(
        np.array([[1.0, 1.0, 0.0]] * len(obj_verts_curr))
    )
    runtime_vis["obj_mesh"].compute_vertex_normals()
    runtime_vis["obj_mesh_gt"].vertices = o3d.utility.Vector3dVector(obj_verts_gt)
    runtime_vis["obj_mesh_gt"].triangles = o3d.utility.Vector3iVector(obj_faces_cur)
    runtime_vis["obj_mesh_gt"].vertex_colors = o3d.utility.Vector3dVector(
        np.array([[1.0, 0.0, 0.0]] * len(obj_verts_gt))
    )
    runtime_vis["obj_mesh_gt"].compute_vertex_normals()
    runtime_vis["window"].update_geometry(runtime_vis["hand_mesh"])
    runtime_vis["window"].update_geometry(runtime_vis["hand_mesh_gt"])
    runtime_vis["window"].update_geometry(runtime_vis["obj_mesh"])
    runtime_vis["window"].update_geometry(runtime_vis["obj_mesh_gt"])
    runtime_vis["window"].poll_events()
    runtime_vis["window"].update_renderer()
    runtime_vis["window"].reset_view_point(True)
