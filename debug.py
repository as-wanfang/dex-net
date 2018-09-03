import dexnet
from dexnet.database.mesh_processor import MeshProcessor

dexnet_api = dexnet.DexNet()

dexnet_api.open_database('example.hdf5', create_db=True)

existing_datasets = [d.name for d in dexnet_api.database.datasets]
if len(existing_datasets) == 1:
    dataset_name = existing_datasets[0]
    dexnet_api.open_dataset(dataset_name)

dexnet_api.delete_object('YellowSaltCube2_5k_tex')
dexnet_api.add_object('YellowSaltCube2_5k_tex.obj') #generate stable pose in this step
dexnet_api.display_stable_poses('YellowSaltCube2_5k_tex')
dexnet_api.dataset.object_keys

#dexnet_api.compute_simulation_data('YellowSaltCube')

dexnet_api.sample_grasps(object_name='YellowSaltCube2_5k_tex', gripper_name='yumi_metal_spline')

# test mesh_processor.generate_graspable function in dexnet_api.add_object
config = dexnet_api._get_config()
mesh_processor = MeshProcessor('YellowSaltCube2_5k_tex.obj', './')
#mesh_processor.generate_graspable(config)

# test mesh_processor.generate_graspable function
mesh_processor._load_mesh()
mesh_processor.mesh_.density = config['obj_density']
mesh_processor._clean_mesh(config['obj_target_scale'], config['obj_scaling_mode'], config['use_uniform_com'], rescale_mesh=config['rescale_objects'])
mesh_processor._generate_sdf(config['path_to_sdfgen'], config['sdf_dim'], config['sdf_padding'])
mesh_processor._generate_stable_poses(config['stp_min_prob'])
mesh_processor.stable_poses_

dexnet_api.dataset.stable_poses('bar_clamp')

dexnet_api.display_stable_poses('YellowSaltCube2_5k_tex_proc')

dexnet_api.compute_metrics(metric_name='robust_ferrari_canny',
                           object_name='YellowSaltCube2_5k_tex',
                           gripper_name='yumi_metal_spline')



dexnet_api.display_grasps('YellowSaltCube2_5k_tex', 'yumi_metal_spline', 'robust_ferrari_canny')
dexnet_api.display_grasps('bar_clamp', 'yumi_metal_spline', 'robust_ferrari_canny')


###############################
# for original obj file, _clean_mesh will result in empty stable pose
# for processed obj file, not
# mesh_processor.mesh_ after _clean_mesh is different from the newly loaded processed obj, the _clean_mesh function may not be well written and need meshlabserver do something
# success = load (meshlabserver) + _clean_mesh + load (meshlabserver) + _clean_mesh
# success = load (meshlabserver) + _clean_mesh + load (meshlabserver)
# fail = load + load + clean / load　+ clean
mesh_processor = MeshProcessor('YellowSaltCube2_5k_tex.obj', './')
mesh_processor._load_mesh()
mesh_processor.mesh_.density = config['obj_density']
mesh_processor._clean_mesh(config['obj_target_scale'], config['obj_scaling_mode'], config['use_uniform_com'], rescale_mesh=config['rescale_objects'])
mesh_processor._generate_sdf(config['path_to_sdfgen'], config['sdf_dim'], config['sdf_padding'])
#mesh_processor._generate_stable_poses(config['stp_min_prob'])
#mesh_processor.mesh_.vertices
#mesh_processor.mesh_.center_of_mass
#mesh_processor.mesh_._compute_com_uniform()
#mesh_processor.mesh_.center_of_mass = mesh_processor.mesh_.bb_center_

#center_of_mass = array([ 2.35758592, 32.46072777, -0.33946854]) mesh_processor.mesh_._compute_com_uniform()
#centroid_ = array([ 2.62808954,  1.16603092, 11.39582063])

mesh_processor1 = MeshProcessor('YellowSaltCube2_5k_tex_proc.obj', './')
mesh_processor1._load_mesh()
mesh_processor1.mesh_.density = config['obj_density']
mesh_processor1._clean_mesh(config['obj_target_scale'], config['obj_scaling_mode'], config['use_uniform_com'], rescale_mesh=config['rescale_objects'])
mesh_processor1._generate_sdf(config['path_to_sdfgen'], config['sdf_dim'], config['sdf_padding'])
mesh_processor1._generate_stable_poses(config['stp_min_prob'])
